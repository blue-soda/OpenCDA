import argparse
import csv
import os

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.tools.offline_inference import (
    apply_selective_sharing_baseline,
    apply_sgcp_constraint,
    apply_resource_overrides,
    clone_grid_selection,
    build_channel_model,
    estimate_grid_selection_payload_bytes,
    estimate_parallel_comm_time_ms,
    load_protocol,
    run_pcs_rounds_with_deadline,
    trim_grid_selection_to_deadline,
)
from opencda.core.common.offline_replay import OfflineCavWorld, clear_sgcp_globals
from opencda.core.clustering.algorithms.clustering.naive_cluster import NaiveCluster
from opencda.core.clustering.algorithms.resource_allocation import build_resource_allocator


def chunk_link(timestamp, source_id, target_id, payload_bytes, upload_type,
               pkt_id, sc_start, sc_num=1, chunk_bytes=10000):
    rows = []
    remaining = int(payload_bytes)
    while remaining > 0:
        size = min(int(chunk_bytes), remaining)
        rows.append({
            'timestamp': timestamp,
            'area_id': '',
            'source_id': int(source_id),
            'target_id': int(target_id),
            'bytes': size,
            'upload_type': upload_type,
            'pkt_id': pkt_id,
            'sc_start': sc_start,
            'sc_num': sc_num,
            'slot_index': '',
            'stage': '',
            'scheduled_delay_ms': '',
        })
        pkt_id += 1
        remaining -= size
    return rows, pkt_id


def set_round_fields(rows, slot_index, scheduled_delay_ms):
    for row in rows:
        row['slot_index'] = slot_index
        row['scheduled_delay_ms'] = '%.6f' % float(scheduled_delay_ms)


def write_plan(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        'timestamp',
        'area_id',
        'source_id',
        'target_id',
        'bytes',
        'upload_type',
        'pkt_id',
        'sc_start',
        'sc_num',
        'slot_index',
        'stage',
        'scheduled_delay_ms',
    ]
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def payloads_from_frame_items(timestamp, frame_items, mode, channel_policy):
    rows = []
    pkt_id = 1
    next_channel = 0
    for eval_frame, metadata in frame_items:
        if not metadata:
            continue
        receiver_id = int(metadata['receiver_id'])
        channel_allocation = metadata.get('channel_allocation', {}) or {}
        channel_sc_nums = metadata.get('channel_sc_nums', {}) or {}
        for source_id, cav in sorted(eval_frame.items(),
                                     key=lambda item: int(item[0])):
            source_id = int(source_id)
            if source_id == receiver_id:
                continue
            payload_bytes = int(cav['lidar_np'].nbytes)
            if payload_bytes <= 0:
                continue
            link = (source_id, receiver_id)
            if channel_policy == 'metadata':
                if link not in channel_allocation:
                    continue
                sc_start = int(channel_allocation[link])
                sc_num = int(channel_sc_nums.get(link, 1))
            elif channel_policy == 'round_robin':
                sc_start = next_channel % 10
                sc_num = 1
                next_channel += 1
            else:
                raise ValueError(channel_policy)
            chunk_rows, pkt_id = chunk_link(
                timestamp,
                source_id,
                receiver_id,
                payload_bytes,
                mode,
                pkt_id,
                sc_start,
                sc_num=sc_num)
            rows.extend(chunk_rows)
    return rows


def pcs_round_upload_rows(frame, protocol, ego_cav_id, timestamp,
                          channel_model):
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=ego_cav_id,
        protocol=protocol,
        density_threshold=None)
    clusters = NaiveCluster(world, all_in_one=False).run()
    allocator = build_resource_allocator('fullperception_pcs', world)
    apply_resource_overrides(
        allocator,
        world,
        num_channels=10,
        bandwidth_mhz=40,
        channel_model=channel_model)
    allocator.set_clusters(clusters)

    # Run the same admission routine once to recover the accepted round times.
    round_metadata = run_pcs_rounds_with_deadline(
        allocator,
        world,
        max_rounds=6,
        deadline_ms=60,
        channel_model=channel_model)
    accepted = int(round_metadata.get('pcs_rounds_accepted', 0))
    round_times = list(round_metadata.get('pcs_round_comm_time_ms', []))

    excluded_receiver_grids = {}
    rows = []
    pkt_id = 1
    scheduled_delay_ms = 0.0
    for round_index in range(accepted):
        allocator.clear_resource_allocation_strategy()
        allocator.excluded_receiver_grids = {
            int(receiver_id): set(grid_ids)
            for receiver_id, grid_ids in excluded_receiver_grids.items()
        }
        allocator.main()
        round_selection = clone_grid_selection(
            getattr(allocator, 'grid_selection', {}))
        payload_bytes, link_bytes = estimate_grid_selection_payload_bytes(
            world,
            round_selection)
        round_time_ms = estimate_parallel_comm_time_ms(
            link_bytes,
            getattr(allocator, 'resource_sc_nums', {}),
            40,
            10,
            channel_model=channel_model)
        remaining_ms = 60.0 - scheduled_delay_ms
        if round_time_ms > remaining_ms:
            round_selection = trim_grid_selection_to_deadline(
                world,
                round_selection,
                getattr(allocator, 'resource_sc_nums', {}),
                40,
                10,
                remaining_ms,
                channel_model=channel_model)
        _, link_bytes = estimate_grid_selection_payload_bytes(
            world,
            round_selection)
        channel_allocation = getattr(allocator, 'resource_strategy', {}) or {}
        channel_sc_nums = getattr(allocator, 'resource_sc_nums', {}) or {}
        round_rows = []
        for link, payload_bytes in sorted(link_bytes.items()):
            if payload_bytes <= 0 or link not in channel_allocation:
                continue
            source_id, target_id = int(link[0]), int(link[1])
            chunk_rows, pkt_id = chunk_link(
                timestamp,
                source_id,
                target_id,
                payload_bytes,
                'pcs_rounds6_d60_seq',
                pkt_id,
                int(channel_allocation[link]),
                sc_num=int(channel_sc_nums.get(link, 1)))
            round_rows.extend(chunk_rows)
        set_round_fields(round_rows, round_index, scheduled_delay_ms)
        rows.extend(round_rows)
        for receiver_id, sender_grids in round_selection.items():
            excluded_receiver_grids.setdefault(receiver_id, set())
            for grid_ids in sender_grids.values():
                excluded_receiver_grids[receiver_id].update(grid_ids)
        if round_index < len(round_times):
            scheduled_delay_ms += float(round_times[round_index])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default=r'D:\Data\Carla')
    parser.add_argument('--scenario-id', default='2026_07_15_01_26_56')
    parser.add_argument('--ego-cav-id', default='1')
    parser.add_argument('--timestamp', default='000060')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    dataset = OPV2VFrameDataset(args.dataset_root)
    protocol = load_protocol(dataset, args.scenario_id)
    frame = dataset.load_frame(
        args.scenario_id,
        args.timestamp,
        ego_cav_id=args.ego_cav_id)
    channel_model = build_channel_model(
        mode='ns3',
        bandwidth_mhz=40,
        num_channels=10,
        frame_deadline_s=0.1,
        ns3_tb_size_bytes=899,
        ns3_slot_duration_ms=0.5,
        ns3_subchannel_prbs=10,
        ns3_symbols_per_slot=12,
        ns3_mcs=28)

    pcs_items = apply_sgcp_constraint(
        frame,
        protocol,
        args.ego_cav_id,
        'fullperception_pcs',
        'all-cavs',
        clustering='singleton',
        num_channels=10,
        bandwidth_mhz=40,
        timestamp=args.timestamp,
        pcs_frame_rounds=6,
        pcs_frame_deadline_ms=60,
        channel_model=channel_model)
    pcs_rows = payloads_from_frame_items(
        args.timestamp,
        pcs_items,
        'pcs_rounds6_d60',
        'metadata')
    write_plan(os.path.join(args.output_dir, 'pcs_upload_plan.csv'), pcs_rows)
    pcs_seq_rows = pcs_round_upload_rows(
        frame,
        protocol,
        args.ego_cav_id,
        args.timestamp,
        channel_model)
    write_plan(os.path.join(args.output_dir, 'pcs_sequential_upload_plan.csv'),
               pcs_seq_rows)

    edge_items = apply_selective_sharing_baseline(
        frame,
        protocol,
        args.ego_cav_id,
        'edgecooper_global',
        'all-cavs',
        3,
        117,
        None,
        'singleton',
        None,
        None,
        timestamp=args.timestamp,
        num_channels=10,
        bandwidth_mhz=40,
        channel_model=channel_model,
        selective_frame_deadline_ms=60)
    edge_rows = payloads_from_frame_items(
        args.timestamp,
        edge_items,
        'edgecooper_global_d60',
        'round_robin')
    write_plan(os.path.join(args.output_dir, 'edgecooper_upload_plan.csv'),
               edge_rows)

    for name, rows in [('pcs', pcs_rows),
                       ('pcs_sequential', pcs_seq_rows),
                       ('edgecooper', edge_rows)]:
        total_bytes = sum(int(row['bytes']) for row in rows)
        links = set((row['source_id'], row['target_id']) for row in rows)
        print('%s chunks=%d links=%d bytes=%d' %
              (name, len(rows), len(links), total_bytes))


if __name__ == '__main__':
    main()
