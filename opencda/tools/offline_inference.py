# -*- coding: utf-8 -*-
"""
Run OpenCOOD inference from an OPV2V-style data dump.
"""

import argparse
import os

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
    parser.add_argument('--sgcp-receiver-policy',
                        choices=['ego', 'ego-cluster-head',
                                 'all-cluster-heads'],
                        default='ego-cluster-head',
                        help='Receiver for constrained perception. all-cluster-heads evaluates every cluster head per frame.')
    parser.add_argument('--sgcp-inter-cluster-late-fusion',
                        action='store_true',
                        help='Late-fuse predictions from all SGCP cluster heads into the requested ego pose and submit one AP sample per frame.')
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


def apply_sgcp_constraint(frame, protocol, ego_cav_id, resource_allocation,
                          receiver_policy):
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=ego_cav_id,
        protocol=protocol)
    clusters = CoalitionGame(world).run()
    apply_cluster_state(world, clusters)
    allocator = build_resource_allocator(resource_allocation, world)
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
        metadata['receiver_policy'] = receiver_policy
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
        return_object_ids=True)


def main():
    args = parse_args()
    dataset = OPV2VFrameDataset(args.dataset_root)

    coperception_params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    manager = OpenCOODManager(coperception_params)
    sgcp_summaries = []

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
            ego_cav_id=args.ego_cav_id)
        frame_items = [(frame, None)]
        if args.sgcp_constrained:
            protocol = load_protocol(dataset, scenario_id)
            frame_items = apply_sgcp_constraint(
                frame,
                protocol,
                args.ego_cav_id,
                args.resource_allocation,
                'all-cluster-heads' if args.sgcp_inter_cluster_late_fusion
                else args.sgcp_receiver_policy)
        if args.sgcp_inter_cluster_late_fusion:
            original_ego = next(cav for cav in frame.values() if cav['ego'])
            target_ego_lidar_pose = original_ego['params']['lidar_pose']
            pred_tensors = []
            pred_scores = []
            gt_tensors = []
            for receiver_index, (eval_frame, sgcp_metadata) in enumerate(
                    frame_items,
                    start=1):
                ret = run_opencood_inference(
                    manager,
                    eval_frame,
                    target_ego_lidar_pose)
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
            print('sgcp_summary frames=%s avg_comm_bytes=%.2f '
                  'total_comm_bytes=%s avg_source_cavs=%.2f' % (
                      len(sgcp_summaries),
                      avg_comm,
                      total_comm,
                      avg_sources))


if __name__ == '__main__':
    main()
