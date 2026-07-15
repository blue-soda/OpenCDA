# -*- coding: utf-8 -*-
"""
Run OpenCOOD inference from an OPV2V-style data dump.
"""

import argparse
import csv
import math
import os
from collections import defaultdict

from omegaconf import OmegaConf
import yaml

from opencda.core.common.offline_dataset import OPV2VFrameDataset
from opencda.core.common.offline_replay import (
    OfflineCavWorld,
    apply_cluster_state,
    build_constrained_frame,
    clear_sgcp_globals,
    select_sgcp_receiver_id,
)
from opencda.core.clustering.algorithms.clustering.coalition_game import (
    CoalitionGame,
)
from opencda.core.clustering.algorithms.clustering.naive_cluster import (
    NaiveCluster,
)
from opencda.core.clustering.utils import common
from opencda.core.clustering.algorithms.resource_allocation import (
    build_resource_allocator,
)
from opencda.core.ml_libs.opencood_manager import OpenCOODManager


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run OpenCOOD inference from dumped OPV2V-style data.')
    parser.add_argument('--dataset-root', required=True,
                        help='Root folder containing scenario subfolders.')
    parser.add_argument('--scenario-id', default=None,
                        help='Scenario folder name. Defaults to the first one.')
    parser.add_argument('--timestamp', default=None,
                        help='Frame timestamp. Defaults to the first frame.')
    parser.add_argument('--ego-cav-id', default=None,
                        help='Ego CAV folder id. Defaults to the first CAV.')
    parser.add_argument('--fusion-method', default=None,
                        help='Override coperception fusion method.')
    parser.add_argument('--coperception-yaml', default=None,
                        help='Path to enable_coperception.yaml.')
    parser.add_argument('--max-frames', type=int, default=1,
                        help='Number of frames to test. Use 0 for all frames.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Frame index to start from within the scenario.')
    parser.add_argument('--sgcp-constrained', action='store_true',
                        help='Run SGCP clustering/resource allocation and evaluate the constrained uploaded frame.')
    parser.add_argument('--resource-allocation', default='potential_game',
                        help='SGCP resource allocation algorithm for constrained inference.')
    parser.add_argument('--clustering', default='coalition_game',
                        choices=['coalition_game', 'singleton', 'all_in_one'],
                        help='Clustering algorithm for SGCP constrained inference.')
    parser.add_argument('--sgcp-receiver-policy',
                        choices=['ego', 'ego-cluster-head',
                                 'all-cluster-heads'],
                        default='ego-cluster-head',
                        help='Receiver for constrained perception. all-cluster-heads evaluates every cluster head per frame.')
    parser.add_argument('--sgcp-inter-cluster-late-fusion',
                        action='store_true',
                        help='Late-fuse predictions from all SGCP cluster heads into the requested ego pose and submit one AP sample per frame.')
    parser.add_argument('--t-min-stab', type=float, default=None,
                        help='Override CoalitionGame Params.T_min_stab in seconds. Use 0 for no stability window.')
    parser.add_argument('--n-max', type=int, default=None,
                        help='Override CoalitionGame Params.N_max.')
    parser.add_argument('--rho-th', type=float, default=None,
                        help='Override lidar density_threshold / rho_th.')
    parser.add_argument('--cav-count', type=int, default=None,
                        help='Use the first N CAVs in numeric order, keeping ego included.')
    parser.add_argument('--cav-ids', default=None,
                        help='Comma-separated CAV ids to evaluate, e.g. 1,2,3.')
    parser.add_argument('--num-channels', type=int, default=None,
                        help='Override SGCP resource allocation channel count.')
    parser.add_argument('--bandwidth-mhz', type=float, default=None,
                        help='Override SGCP total bandwidth in MHz.')
    parser.add_argument('--selective-sharing-baseline', default=None,
                        choices=['nearest', 'density',
                                 'communication_aware'],
                        help='Run a CAV-only selective-sharing baseline instead of SGCP PPS.')
    parser.add_argument('--selective-member-budget', type=int, default=2,
                        help='Maximum uploaded non-head members per receiver for selective baseline.')
    parser.add_argument('--selective-grid-budget', type=int, default=87,
                        help='Maximum selected grids per receiver for selective baseline.')
    parser.add_argument('--ns3-link-quality-csv', default=None,
                        help='Optional rlc_by_request.csv from ns3_log_eval. '
                             'When set, communication_aware selective sharing '
                             'uses per-link RLC complete ratio instead of only '
                             'a distance proxy.')
    return parser.parse_args()


def load_coperception_params(yaml_path, fusion_method=None):
    if yaml_path is None:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
        yaml_path = os.path.join(
            repo_root,
            'opencda/scenario_testing/config_yaml/enable_coperception.yaml')

    params = OmegaConf.load(yaml_path)['enable_coperception']['coperception']
    params = OmegaConf.to_container(params, resolve=True)
    if fusion_method is not None:
        params['fusion_method'] = fusion_method
    return params


def load_protocol(dataset, scenario_id):
    protocol_path = os.path.join(
        dataset.scenarios[scenario_id]['path'],
        'data_protocol.yaml')
    if not os.path.exists(protocol_path):
        return {}
    with open(protocol_path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.Loader)


def cav_sort_key(cav_id):
    try:
        return (0, int(cav_id))
    except ValueError:
        return (1, str(cav_id))


def select_cav_ids(dataset, scenario_id, ego_cav_id=None, cav_count=None,
                   cav_ids=None):
    scenario_cav_ids = sorted(
        dataset.scenarios[scenario_id]['cav_ids'],
        key=cav_sort_key)
    if cav_ids:
        selected = [item.strip() for item in cav_ids.split(',')
                    if item.strip()]
    elif cav_count is not None:
        if cav_count <= 0:
            raise ValueError('--cav-count must be positive')
        selected = scenario_cav_ids[:cav_count]
    else:
        return None

    if ego_cav_id is not None:
        ego_id = str(ego_cav_id)
        if ego_id not in selected:
            selected = [ego_id] + [item for item in selected
                                   if item != ego_id]
            if cav_count is not None:
                selected = selected[:cav_count]
    return selected


def extract_lidar_density_threshold(protocol):
    try:
        return float(
            protocol['vehicle_base']['sensing']['perception']['lidar'].get(
                'density_threshold', 2.0))
    except (AttributeError, KeyError, TypeError):
        return 2.0


def apply_resource_overrides(resource_allocator, world, num_channels=None,
                             bandwidth_mhz=None):
    if num_channels is not None:
        if num_channels <= 0:
            raise ValueError('--num-channels must be positive')
        world.network_manager.subchannel_num = int(num_channels)
    if not hasattr(resource_allocator, 'p'):
        return
    if num_channels is not None:
        resource_allocator.p.num_channels = int(num_channels)
    if bandwidth_mhz is not None:
        if bandwidth_mhz <= 0:
            raise ValueError('--bandwidth-mhz must be positive')
        resource_allocator.p.bandwidth_all = float(bandwidth_mhz) * (10 ** 6)
    if num_channels is not None or bandwidth_mhz is not None:
        resource_allocator.p.bandwidth_per_channel = (
            resource_allocator.p.bandwidth_all /
            resource_allocator.p.num_channels)


def apply_sgcp_constraint(frame, protocol, ego_cav_id, resource_allocation,
                          receiver_policy, t_min_stab=None,
                          clustering='coalition_game', n_max=None,
                          rho_th=None, num_channels=None,
                          bandwidth_mhz=None):
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=ego_cav_id,
        protocol=protocol,
        density_threshold=rho_th)
    if clustering == 'coalition_game':
        clustering_algorithm = CoalitionGame(world)
    elif clustering == 'singleton':
        clustering_algorithm = NaiveCluster(world, all_in_one=False)
    elif clustering == 'all_in_one':
        clustering_algorithm = NaiveCluster(world, all_in_one=True)
    else:
        raise ValueError('Unknown clustering algorithm: %s' % clustering)
    if t_min_stab is not None and hasattr(clustering_algorithm, 'p'):
        clustering_algorithm.p.T_min_stab = t_min_stab
    if n_max is not None and hasattr(clustering_algorithm, 'p'):
        clustering_algorithm.p.N_max = n_max
    clusters = clustering_algorithm.run()
    apply_cluster_state(world, clusters)
    allocator = build_resource_allocator(resource_allocation, world)
    apply_resource_overrides(
        allocator,
        world,
        num_channels=num_channels,
        bandwidth_mhz=bandwidth_mhz)
    allocator.set_clusters(clusters)
    allocator.run()
    if receiver_policy == 'all-cluster-heads':
        receiver_ids = sorted(int(cluster.head_id) for cluster in clusters)
    else:
        receiver_ids = [select_sgcp_receiver_id(
            world,
            ego_cav_id=ego_cav_id,
            receiver_policy=receiver_policy)]

    constrained_items = []
    for receiver_id in receiver_ids:
        constrained_frame, metadata = build_constrained_frame(
            frame,
            world,
            receiver_id)
        metadata['cluster_count'] = len(clusters)
        metadata['resource_allocation'] = resource_allocation
        metadata['clustering'] = clustering
        metadata['receiver_policy'] = receiver_policy
        metadata['t_min_stab'] = (
            common.Params().T_min_stab if t_min_stab is None else t_min_stab)
        metadata['n_max'] = (
            common.Params().N_max if n_max is None else n_max)
        metadata['rho_th'] = (
            extract_lidar_density_threshold(protocol)
            if rho_th is None else rho_th)
        metadata['num_channels'] = (
            getattr(allocator.p, 'num_channels', None)
            if hasattr(allocator, 'p')
            else world.network_manager.subchannel_num)
        metadata['bandwidth_mhz'] = (
            getattr(allocator.p, 'bandwidth_all', 0.0) / (10 ** 6)
            if hasattr(allocator, 'p') else None)
        constrained_items.append((constrained_frame, metadata))
    return constrained_items


def vehicle_distance(vm_a, vm_b):
    pos_a = vm_a.v2x_manager.get_ego_pos().location
    pos_b = vm_b.v2x_manager.get_ego_pos().location
    return math.sqrt(
        (pos_a.x - pos_b.x) ** 2 +
        (pos_a.y - pos_b.y) ** 2)


def load_ns3_link_quality(path):
    if not path:
        return None
    exact = defaultdict(list)
    pair = defaultdict(list)
    with open(path, newline='') as stream:
        for row in csv.DictReader(stream):
            try:
                source = int(row['source_node'])
                target = int(row['target_node'])
            except (KeyError, TypeError, ValueError):
                continue
            timestamp = str(row.get('timestamp', ''))
            complete = int(float(row.get('rlc_complete', 0) or 0))
            exact[(timestamp, source, target)].append(complete)
            pair[(source, target)].append(complete)
    return {
        'exact': {
            key: sum(values) / float(len(values))
            for key, values in exact.items()
        },
        'pair': {
            key: sum(values) / float(len(values))
            for key, values in pair.items()
        },
        'path': path,
    }


def ns3_link_quality(link_quality, timestamp, source_id, target_id):
    if not link_quality:
        return None
    exact_key = (str(timestamp), int(source_id), int(target_id))
    if exact_key in link_quality['exact']:
        return link_quality['exact'][exact_key]
    return link_quality['pair'].get((int(source_id), int(target_id)))


def candidate_grids_for_sender(head_vm, sender_vm):
    head_lidar = head_vm.perception_manager.lidar
    sender_lidar = sender_vm.perception_manager.lidar
    weak_head_grids = head_lidar.req_grids - head_lidar.high_density_grids
    candidates = sender_lidar.sens_grids & weak_head_grids
    if not candidates:
        candidates = sender_lidar.sens_grids
    return candidates


def select_baseline_members(world, cluster, baseline_name, member_budget,
                            link_quality=None, timestamp=None):
    head_id = int(cluster.head_id)
    head_vm = world.get_vehicle_manager(head_id)
    members = [
        int(member_id) for member_id in sorted(cluster.members)
        if int(member_id) != head_id
    ]
    if member_budget <= 0 or not members:
        return []

    if baseline_name == 'nearest':
        scored = [
            (vehicle_distance(head_vm, world.get_vehicle_manager(member_id)),
             member_id)
            for member_id in members
        ]
        return [member_id for _, member_id in sorted(scored)[:member_budget]]

    if baseline_name in ['density', 'communication_aware']:
        scored = []
        for member_id in members:
            sender_vm = world.get_vehicle_manager(member_id)
            candidate_grids = candidate_grids_for_sender(head_vm, sender_vm)
            density_sum = sum(
                sender_vm.perception_manager.lidar.get_grid_density(grid_id)
                for grid_id in candidate_grids)
            if baseline_name == 'communication_aware':
                distance = vehicle_distance(head_vm, sender_vm)
                quality = ns3_link_quality(
                    link_quality,
                    timestamp,
                    member_id,
                    head_id)
                if quality is None:
                    density_sum = density_sum / (1.0 + distance / 100.0)
                else:
                    density_sum = (
                        density_sum * quality / (1.0 + distance / 100.0))
            scored.append((-density_sum, member_id))
        return [member_id for _, member_id in sorted(scored)[:member_budget]]

    raise ValueError('Unknown selective baseline: %s' % baseline_name)


def assign_selective_grid_selection(world, cluster, baseline_name,
                                    member_budget, grid_budget,
                                    link_quality=None, timestamp=None):
    head_id = int(cluster.head_id)
    head_vm = world.get_vehicle_manager(head_id)
    selected_members = select_baseline_members(
        world,
        cluster,
        baseline_name,
        member_budget,
        link_quality=link_quality,
        timestamp=timestamp)
    if grid_budget <= 0 or not selected_members:
        return

    per_member_budget = max(
        1,
        int(math.ceil(grid_budget / float(len(selected_members)))))
    grid_selection = {}
    remaining = int(grid_budget)
    for member_id in selected_members:
        if remaining <= 0:
            break
        sender_vm = world.get_vehicle_manager(member_id)
        candidate_grids = candidate_grids_for_sender(head_vm, sender_vm)
        grids = sorted(
            candidate_grids,
            key=lambda grid_id: sender_vm.perception_manager.lidar.
            get_grid_density(grid_id),
            reverse=True)
        selected = grids[:min(per_member_budget, remaining)]
        if selected:
            grid_selection[member_id] = selected
            remaining -= len(selected)
    head_vm.perception_manager.co_manager.set_grid_selection(grid_selection)


def apply_selective_sharing_baseline(frame, protocol, ego_cav_id,
                                     baseline_name, receiver_policy,
                                     member_budget, grid_budget,
                                     t_min_stab=None, clustering='coalition_game',
                                     n_max=None, rho_th=None,
                                     link_quality=None, timestamp=None):
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=ego_cav_id,
        protocol=protocol,
        density_threshold=rho_th)
    if clustering == 'coalition_game':
        clustering_algorithm = CoalitionGame(world)
    elif clustering == 'singleton':
        clustering_algorithm = NaiveCluster(world, all_in_one=False)
    elif clustering == 'all_in_one':
        clustering_algorithm = NaiveCluster(world, all_in_one=True)
    else:
        raise ValueError('Unknown clustering algorithm: %s' % clustering)
    if t_min_stab is not None and hasattr(clustering_algorithm, 'p'):
        clustering_algorithm.p.T_min_stab = t_min_stab
    if n_max is not None and hasattr(clustering_algorithm, 'p'):
        clustering_algorithm.p.N_max = n_max
    clusters = clustering_algorithm.run()
    apply_cluster_state(world, clusters)
    for cluster in clusters:
        assign_selective_grid_selection(
            world,
            cluster,
            baseline_name,
            member_budget,
            grid_budget,
            link_quality=link_quality,
            timestamp=timestamp)

    if receiver_policy == 'all-cluster-heads':
        receiver_ids = sorted(int(cluster.head_id) for cluster in clusters)
    else:
        receiver_ids = [select_sgcp_receiver_id(
            world,
            ego_cav_id=ego_cav_id,
            receiver_policy=receiver_policy)]

    constrained_items = []
    for receiver_id in receiver_ids:
        constrained_frame, metadata = build_constrained_frame(
            frame,
            world,
            receiver_id)
        metadata['cluster_count'] = len(clusters)
        metadata['resource_allocation'] = (
            'selective_%s' % baseline_name)
        metadata['clustering'] = clustering
        metadata['receiver_policy'] = receiver_policy
        metadata['t_min_stab'] = (
            common.Params().T_min_stab if t_min_stab is None else t_min_stab)
        metadata['n_max'] = (
            common.Params().N_max if n_max is None else n_max)
        metadata['rho_th'] = (
            extract_lidar_density_threshold(protocol)
            if rho_th is None else rho_th)
        metadata['selective_member_budget'] = member_budget
        metadata['selective_grid_budget'] = grid_budget
        metadata['ns3_link_quality_csv'] = (
            link_quality['path'] if link_quality else '')
        constrained_items.append((constrained_frame, metadata))
    return constrained_items


def run_opencood_inference(manager, frame, ego_lidar_pose):
    reformat_data_dict = manager.opencood_dataset.get_item_test(
        frame,
        ego_lidar_pose)
    output_dict = manager.opencood_dataset.collate_batch_test(
        [reformat_data_dict])
    batch_data = manager.to_device(output_dict)
    return manager.inference(
        batch_data,
        with_stats=False,
        return_object_ids=manager.fusion_method != 'late')


def is_empty_pillar_error(error):
    return (
        isinstance(error, RuntimeError) and
        'input.numel() == 0' in str(error))


def main():
    args = parse_args()
    dataset = OPV2VFrameDataset(args.dataset_root)

    coperception_params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    manager = OpenCOODManager(coperception_params)
    sgcp_summaries = []
    ns3_link_quality = load_ns3_link_quality(args.ns3_link_quality_csv)

    if args.timestamp is not None:
        if args.scenario_id is None:
            scenario_id = next(iter(dataset.scenarios.keys()))
        else:
            scenario_id = args.scenario_id
        frames = [(scenario_id, args.timestamp)]
    else:
        if args.scenario_id is None:
            scenario_id = next(iter(dataset.scenarios.keys()))
        else:
            scenario_id = args.scenario_id
        timestamps = dataset.scenarios[scenario_id]['timestamps']
        if args.max_frames == 0:
            selected = timestamps[args.start_index:]
        else:
            selected = timestamps[args.start_index:
                                  args.start_index + args.max_frames]
        frames = [(scenario_id, timestamp) for timestamp in selected]

    for index, (scenario_id, timestamp) in enumerate(frames, start=1):
        frame = dataset.load_frame(
            scenario_id,
            timestamp,
            ego_cav_id=args.ego_cav_id,
            cav_ids=select_cav_ids(
                dataset,
                scenario_id,
                ego_cav_id=args.ego_cav_id,
                cav_count=args.cav_count,
                cav_ids=args.cav_ids))
        frame_items = [(frame, None)]
        if args.selective_sharing_baseline is not None:
            protocol = load_protocol(dataset, scenario_id)
            frame_items = apply_selective_sharing_baseline(
                frame,
                protocol,
                args.ego_cav_id,
                args.selective_sharing_baseline,
                'all-cluster-heads' if args.sgcp_inter_cluster_late_fusion
                else args.sgcp_receiver_policy,
                args.selective_member_budget,
                args.selective_grid_budget,
                args.t_min_stab,
                args.clustering,
                args.n_max,
                args.rho_th,
                link_quality=ns3_link_quality,
                timestamp=timestamp)
        elif args.sgcp_constrained:
            protocol = load_protocol(dataset, scenario_id)
            frame_items = apply_sgcp_constraint(
                frame,
                protocol,
                args.ego_cav_id,
                args.resource_allocation,
                'all-cluster-heads' if args.sgcp_inter_cluster_late_fusion
                else args.sgcp_receiver_policy,
                args.t_min_stab,
                args.clustering,
                args.n_max,
                args.rho_th,
                args.num_channels,
                args.bandwidth_mhz)
        if args.sgcp_inter_cluster_late_fusion:
            original_ego = next(cav for cav in frame.values() if cav['ego'])
            target_ego_lidar_pose = original_ego['params']['lidar_pose']
            pred_tensors = []
            pred_scores = []
            gt_tensors = []
            for receiver_index, (eval_frame, sgcp_metadata) in enumerate(
                    frame_items,
                    start=1):
                try:
                    ret = run_opencood_inference(
                        manager,
                        eval_frame,
                        target_ego_lidar_pose)
                except RuntimeError as error:
                    if not is_empty_pillar_error(error):
                        raise
                    if sgcp_metadata is not None:
                        sgcp_summaries.append(sgcp_metadata)
                    print('frame=%s/%s late_source=%s/%s scenario=%s '
                          'timestamp=%s receiver=%s cavs=%s skipped=%s '
                          'comm_bytes=%s' % (
                              index,
                              len(frames),
                              receiver_index,
                              len(frame_items),
                              scenario_id,
                              timestamp,
                              sgcp_metadata['receiver_id'],
                              list(eval_frame.keys()),
                              'empty_pillars',
                              sgcp_metadata['communication_bytes']))
                    continue
                pred_box_tensor, pred_score, gt_box_tensor = ret[0:3]
                if pred_box_tensor is not None and pred_score is not None:
                    pred_tensors.append(pred_box_tensor)
                    pred_scores.append(pred_score)
                if gt_box_tensor is not None:
                    gt_tensors.append(gt_box_tensor)
                if sgcp_metadata is not None:
                    sgcp_summaries.append(sgcp_metadata)
                print('frame=%s/%s late_source=%s/%s scenario=%s '
                      'timestamp=%s receiver=%s cavs=%s pred_boxes=%s '
                      'gt_boxes=%s comm_bytes=%s' % (
                          index,
                          len(frames),
                          receiver_index,
                          len(frame_items),
                          scenario_id,
                          timestamp,
                          sgcp_metadata['receiver_id'],
                          list(eval_frame.keys()),
                          0 if pred_box_tensor is None else
                          pred_box_tensor.shape[0],
                          0 if gt_box_tensor is None else
                          gt_box_tensor.shape[0],
                          sgcp_metadata['communication_bytes']))

            fused_pred, fused_score = manager.naive_late_fusion(
                pred_tensors,
                pred_scores)
            fused_gt, _ = manager.naive_late_fusion(gt_tensors, None)
            print('sgcp_late_fusion frame=%s/%s scenario=%s timestamp=%s '
                  'sources=%s fused_pred_boxes=%s fused_gt_boxes=%s' % (
                      index,
                      len(frames),
                      scenario_id,
                      timestamp,
                      len(frame_items),
                      0 if fused_pred is None else fused_pred.shape[0],
                      0 if fused_gt is None else fused_gt.shape[0]))
            manager.submit_results(
                fused_pred,
                fused_score,
                fused_gt,
                with_stats=True,
                force=True)
            continue
        for receiver_index, (eval_frame, sgcp_metadata) in enumerate(
                frame_items,
                start=1):
            ego = next(cav for cav in eval_frame.values() if cav['ego'])
            ego_lidar_pose = ego['params']['lidar_pose']

            ret = run_opencood_inference(manager, eval_frame, ego_lidar_pose)

            pred_box_tensor, pred_score, gt_box_tensor = ret[0:3]
            pred_count = (
                0 if pred_box_tensor is None else pred_box_tensor.shape[0])
            gt_count = 0 if gt_box_tensor is None else gt_box_tensor.shape[0]
            print('frame=%s/%s receiver_sample=%s/%s scenario=%s '
                  'timestamp=%s cavs=%s' % (
                      index,
                      len(frames),
                      receiver_index,
                      len(frame_items),
                      scenario_id,
                      timestamp,
                      list(eval_frame.keys())))
            if sgcp_metadata is not None:
                sgcp_summaries.append(sgcp_metadata)
                print('sgcp_constrained receiver=%s policy=%s sources=%s '
                      'clusters=%s ra=%s comm_bytes=%s selected_grids=%s' % (
                          sgcp_metadata['receiver_id'],
                          sgcp_metadata['receiver_policy'],
                          sgcp_metadata['source_cav_ids'],
                          sgcp_metadata['cluster_count'],
                          sgcp_metadata['resource_allocation'],
                          sgcp_metadata['communication_bytes'],
                          sgcp_metadata['selected_grid_counts']))
            print('fusion_method=%s pred_boxes=%s gt_boxes=%s' %
                  (coperception_params['fusion_method'], pred_count, gt_count))
            if pred_score is not None:
                print('pred_scores_shape=%s' % (tuple(pred_score.shape),))
            manager.submit_results(
                pred_box_tensor,
                pred_score,
                gt_box_tensor,
                with_stats=True,
                force=True)

    if len(frames) > 1:
        manager.evaluate_final_average_precision()
        if sgcp_summaries:
            total_comm = sum(item['communication_bytes']
                             for item in sgcp_summaries)
            avg_comm = total_comm / float(len(sgcp_summaries))
            avg_sources = sum(len(item['source_cav_ids'])
                              for item in sgcp_summaries) / \
                float(len(sgcp_summaries))
            avg_selected_grids = sum(
                sum(item.get('selected_grid_counts', {}).values())
                for item in sgcp_summaries) / float(len(sgcp_summaries))
            print('sgcp_summary frames=%s avg_comm_bytes=%.2f '
                  'total_comm_bytes=%s avg_source_cavs=%.2f '
                  'avg_selected_grids=%.2f' % (
                      len(sgcp_summaries),
                      avg_comm,
                      total_comm,
                      avg_sources,
                      avg_selected_grids))


if __name__ == '__main__':
    main()
