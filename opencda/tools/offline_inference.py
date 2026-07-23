# -*- coding: utf-8 -*-
"""
Run OpenCOOD inference from an OPV2V-style data dump.
"""

import argparse
import copy
import csv
import math
import os
import json
import random
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from omegaconf import OmegaConf
import yaml

from opencood.utils import common_utils
from opencood.utils.transformation_utils import x1_to_x2

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
from opencda.core.clustering.algorithms.clustering.cov_coalition_game import (
    COVCoalitionGame,
)
from opencda.core.clustering.algorithms.clustering.paper_baselines import (
    build_paper_baseline_clusters,
)
from opencda.core.clustering.algorithms.clustering.naive_cluster import (
    NaiveCluster,
)
from opencda.core.clustering.utils import common
from opencda.core.clustering.utils.channel_model import build_channel_model
from opencda.core.clustering.algorithms.resource_allocation import (
    build_resource_allocator,
)
from opencda.core.ml_libs.opencood_manager import OpenCOODManager


EDGECOOPER_GLOBAL_COMM_RANGE_M = 35.0
DEFAULT_COPERCEPTION_YAML = (
    'docs/doc_workspace/SGCP/artifacts/'
    'early_from_late_checkpoint_20260719/'
    'enable_coperception_early_from_attentive.yaml')


def repo_root():
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))


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
    parser.add_argument('--coperception-yaml',
                        default=DEFAULT_COPERCEPTION_YAML,
                        help='Path to enable_coperception.yaml. Defaults to '
                             'the SGCP attentive current-protocol config.')
    parser.add_argument('--max-frames', type=int, default=1,
                        help='Number of frames to test. Use 0 for all frames.')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Frame index to start from within the scenario.')
    parser.add_argument('--sgcp-constrained', action='store_true',
                        help='Run SGCP clustering/resource allocation and evaluate the constrained uploaded frame.')
    parser.add_argument('--resource-allocation', default='potential_game',
                        help='SGCP resource allocation algorithm for constrained inference.')
    parser.add_argument('--clustering', default='coalition_game',
                        choices=['coalition_game', 'cov_coalition_game',
                                 'fixed_first_frame',
                                 'singleton', 'all_in_one',
                                 'random_balanced',
                                 'distance_greedy',
                                 'density_greedy_cluster',
                                 'mobility_stability_greedy',
                                 'seac_social_adaptive',
                                 'hho_vanet'],
                        help='Clustering algorithm for SGCP constrained inference.')
    parser.add_argument('--sgcp-receiver-policy',
                        choices=['ego', 'ego-cluster-head',
                                 'all-cluster-heads',
                                 'all-scheduled-receivers',
                                 'all-cavs'],
                        default='ego-cluster-head',
                        help='Receiver for constrained perception. all-cluster-heads evaluates every cluster head per frame; all-scheduled-receivers evaluates receivers that actually receive scheduled uploads; all-cavs evaluates every CAV as a potential receiver.')
    parser.add_argument('--sgcp-inter-cluster-late-fusion',
                        action='store_true',
                        help='Late-fuse predictions from all SGCP cluster heads into the requested ego pose and submit one AP sample per frame.')
    parser.add_argument('--sgcp-late-nms-thresh', type=float, default=0.15,
                        help='NMS IoU threshold for SGCP inter-cluster late '
                             'fusion. Defaults to the previous value 0.15.')
    parser.add_argument('--sgcp-upload-mode', default='grid',
                        choices=['grid', 'head_only', 'full_cluster'],
                        help='Upload mode for SGCP constrained replay. grid uses PPS-selected grids; head_only keeps only each receiver; full_cluster uploads all cluster member point clouds for protocol probes.')
    parser.add_argument('--sgcp-grid-selection-mode', default='utility',
                        choices=['utility', 'random', 'spatial_diverse',
                                 'object_clustered'],
                        help='Grid selection mode for SGCP scheduled links. '
                             'random/spatial_diverse/object_clustered keep '
                             'scheduled links and grid counts but replace '
                             'grids with deterministic candidate choices.')
    parser.add_argument('--sgcp-grid-score-mode', default='utility',
                        choices=['utility', 'raw_density',
                                 'density_distance'],
                        help='Grid scoring mode used by potential_game before optional grid selection replacement.')
    parser.add_argument('--sgcp-coverage-fallback', default='none',
                        choices=['none', 'persistent',
                                 'quality_persistent'],
                        help='Optional SGCP member-coverage fallback probe. '
                             'persistent uses coverage history only; '
                             'quality_persistent additionally requires the '
                             'candidate member to have comparable historical '
                             'detector-quality proxy. Both reuse the same '
                             'subchannel and grid budget. Defaults to none.')
    parser.add_argument('--sgcp-routing-hints-csv', default=None,
                        help='Optional oracle/debug CSV from '
                             'sgcp_object_point_association. When set, '
                             'SGCP replaces at most N existing scheduled '
                             'links per frame with hinted target-to-head '
                             'routes under the same RB budget.')
    parser.add_argument('--sgcp-routing-hints-max-per-frame', type=int,
                        default=1,
                        help='Maximum diagnostic routing-hint replacements '
                             'per frame. Defaults to 1.')
    parser.add_argument('--t-min-stab', type=float, default=None,
                        help='Override CoalitionGame stability prediction '
                             'window in seconds. Defaults to the inferred '
                             'perception frame interval; use 0 for no '
                             'stability window.')
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
    parser.add_argument('--channel-estimator', default='ns3',
                        choices=['logical', 'ns3'],
                        help='Shared channel estimator for all schedulers. '
                             'Defaults to ns3, the SGCP current-protocol '
                             'calibrated TB-size-per-slot service rate; '
                             'logical preserves bandwidth/num_channels.')
    parser.add_argument('--communication-deadline-ms', type=float,
                        default=None,
                        help='Communication budget per perception frame in '
                             'milliseconds. Defaults to the scenario network '
                             'time_slot, typically 100 ms.')
    parser.add_argument('--sgcp-frame-mbps-budget', type=float, default=None,
                        help='Optional raw-LiDAR payload budget in Mbps per '
                             '100 ms perception frame for SGCP grid uploads. '
                             'This caps selected grid payload after scheduling '
                             'without changing the scheduler objective.')
    parser.add_argument('--ns3-tb-size-bytes', type=int, default=899,
                        help='NS3-calibrated transport block bytes per '
                             'subchannel grant. Defaults to 899 for the SGCP '
                             '40 MHz / 10 target-subchannel protocol.')
    parser.add_argument('--ns3-slot-duration-ms', type=float, default=0.5,
                        help='NR slot duration used by the NS3 estimator. '
                             'Defaults to 0.5 ms for numerology 1.')
    parser.add_argument('--ns3-subchannel-prbs', type=int, default=10,
                        help='NS3 sidelink PRBs per logical subchannel.')
    parser.add_argument('--ns3-symbols-per-slot', type=int, default=12,
                        help='PSSCH symbols per slot used by NS3 manual '
                             'scheduler TB sizing.')
    parser.add_argument('--ns3-mcs', type=int, default=28,
                        help='NS3 sidelink MCS used by manual scheduler. '
                             'Defaults to SGCP current-protocol MCS 28.')
    parser.add_argument('--head-rb-budget', type=int, default=None,
                        help='Override PotentialGame per-head RB budget B_h. '
                             'Defaults to 1 to preserve the original SGCP '
                             'protocol.')
    parser.add_argument('--pcs-blind-spot-min-division', type=int,
                        default=None,
                        help='Override FullPerception PCS blind-spot '
                             'division granularity. This changes PCS blind '
                             'spot units, not network bandwidth.')
    parser.add_argument('--pcs-min-overlap-grids', type=int, default=None,
                        help='Override minimum sender/receiver blind-spot '
                             'grid overlap for FullPerception PCS candidate '
                             'links.')
    parser.add_argument('--pcs-blind-spot-radius', type=int, default=None,
                        help='Override PCS blind-spot connected-neighborhood '
                             'radius in grid cells. This changes blind-spot '
                             'unitization only, not bandwidth.')
    parser.add_argument('--pcs-min-spot-grids', type=int, default=None,
                        help='Override minimum grids per PCS blind-spot unit. '
                             'Use to avoid unrealistically tiny blind spots.')
    parser.add_argument('--pcs-communication-range-m', type=float,
                        default=None,
                        help='Override PCS sender-receiver communication '
                             'range in meters. This changes the physical '
                             'candidate-link range only, not the PCS '
                             'scheduling mechanism.')
    parser.add_argument('--pcs-frame-rounds', type=int, default=1,
                        help='Maximum number of repeated PCS scheduling '
                             'rounds inside one perception frame. Each round '
                             'excludes receiver grids already accepted in '
                             'previous rounds. Defaults to 1.')
    parser.add_argument('--pcs-frame-deadline-ms', type=float, default=None,
                        help='Optional per-frame PCS raw-LiDAR communication '
                             'deadline in milliseconds. Repeated PCS rounds '
                             'stop before exceeding this deadline.')
    parser.add_argument('--max-upload-points-per-source', type=int,
                        default=None,
                        help='Optional deterministic point budget for each '
                             'uploaded source CAV after grid/full-cluster '
                             'selection. This keeps scheduling semantics '
                             'unchanged while probing payload/AP tradeoff.')
    parser.add_argument('--selective-sharing-baseline', default=None,
                        choices=['random', 'nearest', 'density',
                                 'greedy_density', 'communication_aware',
                                 'pacp_lidar',
                                 'global_selective_proxy',
                                 'cluster_local_selective_proxy',
                                 'edgecooper',
                                 'edgecooper_global',
                                 'edgecooper_global_hd'],
                        help='Run a selective-sharing or RSU/edge-assisted baseline instead of SGCP PPS.')
    parser.add_argument('--selective-member-budget', type=int, default=2,
                        help='Maximum uploaded non-head members per receiver for selective baseline.')
    parser.add_argument('--selective-grid-budget', type=int, default=87,
                        help='Maximum selected grids per receiver for selective baseline.')
    parser.add_argument('--selective-frame-deadline-ms', type=float,
                        default=None,
                        help='Optional deadline-aware trimming for selective '
                             'baselines using the shared channel estimator. '
                             'Defaults to disabled to preserve fixed-budget '
                             'baseline protocols.')
    parser.add_argument('--edgecooper-global-comm-range-m', type=float,
                        default=EDGECOOPER_GLOBAL_COMM_RANGE_M,
                        help='Communication range for edgecooper_global '
                             'candidate V2V links. Defaults to 35 m to '
                             'preserve existing baseline results.')
    parser.add_argument('--max-senders-per-receiver', type=int, default=1,
                        help='Receiver-side concurrent inbound-link capacity '
                             'for orthogonal resources. Defaults to 1 to '
                             'preserve the endpoint-disjoint Table 1 PCS and '
                             'EdgeCooper baselines; K=2 is a protocol '
                             'capability sensitivity.')
    parser.add_argument('--ns3-link-quality-csv', default=None,
                        help='Optional rlc_by_request.csv from ns3_log_eval. '
                             'When set, communication_aware selective sharing '
                             'uses per-link RLC complete ratio instead of only '
                             'a distance proxy.')
    parser.add_argument('--sgcp-trace-output', default=None,
                        help='Optional CSV path for per-receiver SGCP protocol trace.')
    parser.add_argument('--object-diagnostics-output', default=None,
                        help='Optional CSV path for per-GT object miss diagnostics. '
                             'When set, each evaluated sample is compared '
                             'against a full-20CAV reference from the same '
                             'frame.')
    parser.add_argument('--eval-stats-output', default=None,
                        help='Optional CSV path for per-evaluated-sample '
                             'TP/FP/GT deltas. This supports frame/sample '
                             'bootstrap uncertainty estimates without '
                             'changing inference behavior.')
    parser.add_argument('--object-diagnostics-iou', type=float, default=0.5,
                        help='IoU threshold used by object diagnostics. '
                             'Defaults to 0.5.')
    parser.add_argument('--collapse-to-ego-pointcloud', action='store_true',
                        help='Before OpenCOOD inference, project all CAV '
                             'point clouds in the evaluated frame into the '
                             'ego/receiver lidar pose and collapse them into '
                             'one ego CAV. This keeps the SGCP point-cloud '
                             'communication plan but lets single-agent or '
                             'intermediate checkpoints act as merged point '
                             'cloud detectors.')
    parser.add_argument('--debug-opencood-output', action='store_true',
                        help='Print compact model-output/postprocess '
                             'diagnostics for each OpenCOOD inference call. '
                             'This is intended for checkpoint adaptation '
                             'smoke tests and is off by default.')
    parser.add_argument('--postprocess-score-threshold', type=float,
                        default=None,
                        help='Temporarily override the OpenCOOD '
                             'postprocessor target score_threshold. Useful '
                             'for detector checkpoint calibration probes.')
    parser.add_argument('--postprocess-nms-thresh', type=float, default=None,
                        help='Temporarily override the OpenCOOD '
                             'postprocessor nms_thresh for detector '
                             'checkpoint calibration probes.')
    return parser.parse_args()


def load_coperception_params(yaml_path, fusion_method=None):
    if yaml_path is None:
        yaml_path = os.path.join(
            repo_root(),
            'opencda/scenario_testing/config_yaml/enable_coperception.yaml')
    elif not os.path.isabs(yaml_path):
        candidate = os.path.join(repo_root(), yaml_path)
        if os.path.exists(candidate):
            yaml_path = candidate

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


def fixed_delta_from_protocol(protocol, fallback=0.05):
    try:
        return float(protocol['world']['fixed_delta_seconds'])
    except (KeyError, TypeError, ValueError):
        return fallback


def frame_interval_seconds(timestamps, fixed_delta_seconds):
    if len(timestamps) >= 2:
        try:
            return ((int(timestamps[1]) - int(timestamps[0])) *
                    fixed_delta_seconds)
        except ValueError:
            pass
    return fixed_delta_seconds


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


def build_cli_channel_model(args, world=None):
    time_slot = 0.1
    if world is not None:
        time_slot = float(getattr(world.network_manager, 'time_slot', 0.1))
    if args.communication_deadline_ms is not None:
        if args.communication_deadline_ms <= 0:
            raise ValueError('--communication-deadline-ms must be positive')
        time_slot = float(args.communication_deadline_ms) / 1000.0
    return build_channel_model(
        mode=args.channel_estimator,
        bandwidth_mhz=args.bandwidth_mhz or 20.0,
        num_channels=args.num_channels or (
            getattr(world.network_manager, 'subchannel_num', 10)
            if world is not None else 10),
        frame_deadline_s=time_slot,
        ns3_tb_size_bytes=args.ns3_tb_size_bytes,
        ns3_slot_duration_ms=args.ns3_slot_duration_ms,
        ns3_subchannel_prbs=args.ns3_subchannel_prbs,
        ns3_symbols_per_slot=args.ns3_symbols_per_slot,
        ns3_mcs=args.ns3_mcs)


def apply_resource_overrides(resource_allocator, world, num_channels=None,
                             bandwidth_mhz=None, head_rb_budget=None,
                             pcs_blind_spot_min_division=None,
                             pcs_min_overlap_grids=None,
                             pcs_blind_spot_radius=None,
                             pcs_min_spot_grids=None,
                             pcs_communication_range_m=None,
                             channel_model=None,
                             max_senders_per_receiver=None):
    if num_channels is not None:
        if num_channels <= 0:
            raise ValueError('--num-channels must be positive')
        world.network_manager.subchannel_num = int(num_channels)
        if hasattr(resource_allocator, 'lambda_subchannels'):
            resource_allocator.lambda_subchannels = int(num_channels)
    if bandwidth_mhz is not None and hasattr(resource_allocator, 'bandwidth_all'):
        if bandwidth_mhz <= 0:
            raise ValueError('--bandwidth-mhz must be positive')
        resource_allocator.bandwidth_all = float(bandwidth_mhz) * (10 ** 6)
    if pcs_blind_spot_min_division is not None:
        if pcs_blind_spot_min_division <= 0:
            raise ValueError('--pcs-blind-spot-min-division must be positive')
        if hasattr(resource_allocator, 'blind_spot_min_division'):
            resource_allocator.blind_spot_min_division = int(
                pcs_blind_spot_min_division)
    if pcs_min_overlap_grids is not None:
        if pcs_min_overlap_grids < 0:
            raise ValueError('--pcs-min-overlap-grids cannot be negative')
        if hasattr(resource_allocator, 'min_overlap_grids'):
            resource_allocator.min_overlap_grids = int(pcs_min_overlap_grids)
    if pcs_blind_spot_radius is not None:
        if pcs_blind_spot_radius <= 0:
            raise ValueError('--pcs-blind-spot-radius must be positive')
        if hasattr(resource_allocator, 'blind_spot_adjacency_radius'):
            resource_allocator.blind_spot_adjacency_radius = int(
                pcs_blind_spot_radius)
    if pcs_min_spot_grids is not None:
        if pcs_min_spot_grids <= 0:
            raise ValueError('--pcs-min-spot-grids must be positive')
        if hasattr(resource_allocator, 'blind_spot_min_grids'):
            resource_allocator.blind_spot_min_grids = int(
                pcs_min_spot_grids)
    if pcs_communication_range_m is not None:
        if pcs_communication_range_m <= 0:
            raise ValueError('--pcs-communication-range-m must be positive')
        if hasattr(resource_allocator, 'communication_range_m'):
            resource_allocator.communication_range_m = float(
                pcs_communication_range_m)
    if max_senders_per_receiver is not None:
        if max_senders_per_receiver <= 0:
            raise ValueError('--max-senders-per-receiver must be positive')
        if hasattr(resource_allocator, 'max_senders_per_receiver'):
            resource_allocator.max_senders_per_receiver = int(
                max_senders_per_receiver)
    if hasattr(resource_allocator, 'time_slot'):
        resource_allocator.time_slot = float(
            getattr(world.network_manager, 'time_slot', 0.1))
    if channel_model is not None:
        resource_allocator.channel_model = channel_model
        if hasattr(resource_allocator, 'time_slot'):
            resource_allocator.time_slot = float(channel_model.frame_deadline_s)
        if hasattr(resource_allocator, 'lambda_subchannels'):
            resource_allocator.lambda_subchannels = channel_model.num_channels
        if hasattr(resource_allocator, 'bandwidth_all'):
            resource_allocator.bandwidth_all = channel_model.bandwidth_bps
    if not hasattr(resource_allocator, 'p'):
        return
    if num_channels is not None:
        resource_allocator.p.num_channels = int(num_channels)
    if bandwidth_mhz is not None:
        if bandwidth_mhz <= 0:
            raise ValueError('--bandwidth-mhz must be positive')
        resource_allocator.p.bandwidth_all = float(bandwidth_mhz) * (10 ** 6)
    if head_rb_budget is not None:
        if head_rb_budget <= 0:
            raise ValueError('--head-rb-budget must be positive')
        resource_allocator.p.head_rb_budget = int(head_rb_budget)
    if num_channels is not None or bandwidth_mhz is not None:
        resource_allocator.p.bandwidth_per_channel = (
            resource_allocator.p.bandwidth_all /
            resource_allocator.p.num_channels)
    if channel_model is not None:
        resource_allocator.p.channel_model = channel_model
        resource_allocator.p.T_ddl = float(channel_model.frame_deadline_s)
        resource_allocator.p.num_channels = channel_model.num_channels
        resource_allocator.p.bandwidth_all = channel_model.bandwidth_bps
        resource_allocator.p.bandwidth_per_channel = (
            channel_model.bandwidth_bps / channel_model.num_channels)


def clone_grid_selection(grid_selection):
    return {
        int(receiver_id): {
            int(sender_id): set(grid_ids)
            for sender_id, grid_ids in sender_grids.items()
        }
        for receiver_id, sender_grids in (grid_selection or {}).items()
    }


def merge_grid_selection(target, source):
    for receiver_id, sender_grids in clone_grid_selection(source).items():
        target.setdefault(receiver_id, {})
        for sender_id, grid_ids in sender_grids.items():
            target[receiver_id].setdefault(sender_id, set())
            target[receiver_id][sender_id].update(grid_ids)


def estimate_grid_selection_payload_bytes(world, grid_selection):
    total_bytes = 0
    link_bytes = {}
    for receiver_id, sender_grids in clone_grid_selection(
            grid_selection).items():
        for sender_id, grid_ids in sender_grids.items():
            sender_vm = world.get_vehicle_manager(sender_id)
            if sender_vm is None:
                continue
            selected_points = sender_vm.perception_manager.lidar.\
                get_local_points_by_grid_ids(grid_ids)
            if selected_points is None or selected_points.size == 0:
                payload_bytes = 0
            else:
                payload_bytes = int(selected_points.nbytes)
            link_bytes[(sender_id, receiver_id)] = payload_bytes
            total_bytes += payload_bytes
    return total_bytes, link_bytes


def estimate_parallel_comm_time_ms(link_bytes, resource_sc_nums,
                                   bandwidth_mhz, num_channels,
                                   channel_model=None):
    """Estimate raw-LiDAR transfer time for one conflict-free round."""
    if not link_bytes:
        return 0.0
    channel_model = channel_model or build_channel_model(
        mode='logical',
        bandwidth_mhz=bandwidth_mhz or 20.0,
        num_channels=num_channels or 1)
    max_seconds = 0.0
    for link, payload_bytes in link_bytes.items():
        if payload_bytes <= 0:
            continue
        sc_num = max(int((resource_sc_nums or {}).get(link, 1)), 1)
        max_seconds = max(
            max_seconds,
            channel_model.payload_time_ms(payload_bytes, sc_num) / 1000.0)
    return max_seconds * 1000.0


def trim_grid_selection_to_deadline(world, grid_selection, resource_sc_nums,
                                    bandwidth_mhz, num_channels,
                                    deadline_ms, channel_model=None):
    """Trim PCS-selected grids so every parallel link fits the remaining time."""
    if deadline_ms is None:
        return clone_grid_selection(grid_selection)
    channel_model = channel_model or build_channel_model(
        mode='logical',
        bandwidth_mhz=bandwidth_mhz or 20.0,
        num_channels=num_channels or 1)
    trimmed = {}
    for receiver_id, sender_grids in clone_grid_selection(
            grid_selection).items():
        for sender_id, grid_ids in sender_grids.items():
            sender_vm = world.get_vehicle_manager(sender_id)
            if sender_vm is None:
                continue
            link = (int(sender_id), int(receiver_id))
            sc_num = max(int((resource_sc_nums or {}).get(link, 1)), 1)
            budget_bytes = channel_model.payload_budget_bytes(
                deadline_ms=deadline_ms,
                subchannels=sc_num)
            if budget_bytes <= 0:
                continue
            lidar = sender_vm.perception_manager.lidar

            def grid_score(grid_id):
                grid_index = grid_index_from_id(grid_id)
                if grid_index is None:
                    grid_index = (999999, 999999)
                return (
                    lidar.get_grid_density(grid_id),
                    -grid_index[0],
                    -grid_index[1],
                    str(grid_id))

            selected = []
            used_bytes = 0
            for grid_id in sorted(grid_ids, key=grid_score, reverse=True):
                points = lidar.get_local_points_by_grid_ids([grid_id])
                grid_bytes = 0 if points is None else int(points.nbytes)
                if grid_bytes <= 0:
                    continue
                if used_bytes + grid_bytes > budget_bytes:
                    continue
                selected.append(grid_id)
                used_bytes += grid_bytes
            if selected:
                trimmed.setdefault(int(receiver_id), {})[int(sender_id)] = set(
                    selected)
    return trimmed


def trim_grid_selection_to_payload_budget(world, grid_selection,
                                          max_payload_bytes,
                                          strategies=None):
    """Deterministically admit scheduled grids under a frame payload budget."""
    selection = clone_grid_selection(grid_selection)
    original_bytes, _ = estimate_grid_selection_payload_bytes(world, selection)
    if max_payload_bytes is None:
        return selection, original_bytes, original_bytes
    max_payload_bytes = int(max_payload_bytes)
    if max_payload_bytes <= 0:
        return {}, original_bytes, 0

    scheduled_order = []
    seen = set()
    if strategies:
        for receiver_id in sorted(strategies):
            for sender_id, _, _, grid_ids in strategies.get(receiver_id, []):
                receiver_id = int(receiver_id)
                sender_id = int(sender_id)
                available = selection.get(receiver_id, {}).get(sender_id, set())
                for grid_id in grid_ids:
                    key = (receiver_id, sender_id, grid_id)
                    if grid_id in available and key not in seen:
                        scheduled_order.append(key)
                        seen.add(key)

    leftovers = []
    for receiver_id, sender_grids in selection.items():
        for sender_id, grid_ids in sender_grids.items():
            sender_vm = world.get_vehicle_manager(sender_id)
            lidar = None if sender_vm is None else sender_vm.perception_manager.lidar
            for grid_id in grid_ids:
                key = (int(receiver_id), int(sender_id), grid_id)
                if key in seen:
                    continue
                density = 0.0 if lidar is None else lidar.get_grid_density(grid_id)
                leftovers.append((density, str(grid_id), key))
    leftovers.sort(reverse=True)

    trimmed = {}
    used_bytes = 0
    for receiver_id, sender_id, grid_id in (
            scheduled_order + [item[2] for item in leftovers]):
        sender_vm = world.get_vehicle_manager(sender_id)
        if sender_vm is None:
            continue
        points = sender_vm.perception_manager.lidar.\
            get_local_points_by_grid_ids([grid_id])
        grid_bytes = 0 if points is None else int(points.nbytes)
        if grid_bytes <= 0:
            continue
        if used_bytes + grid_bytes > max_payload_bytes:
            continue
        trimmed.setdefault(receiver_id, {}).setdefault(sender_id, set()).add(
            grid_id)
        used_bytes += grid_bytes
    return trimmed, original_bytes, used_bytes


def apply_grid_selection_to_world(world, grid_selection,
                                  channel_allocation=None):
    """Replace offline receiver grid selections and matching channel records."""
    channel_allocation = dict(channel_allocation or {})
    for vm in world.get_vehicle_managers().values():
        vm.perception_manager.co_manager.clear_grid_selection()
        scheduler = getattr(vm.v2x_manager, 'scheduler', None)
        if scheduler is not None and hasattr(scheduler, 'clear_strategies'):
            scheduler.clear_strategies()

    for receiver_id, sender_grids in clone_grid_selection(
            grid_selection).items():
        receiver_vm = world.get_vehicle_manager(receiver_id)
        if receiver_vm is None:
            continue
        clean_selection = {
            int(sender_id): set(grid_ids)
            for sender_id, grid_ids in sender_grids.items()
            if grid_ids
        }
        if not clean_selection:
            continue
        receiver_vm.perception_manager.co_manager.set_grid_selection(
            clean_selection)
        scheduler = getattr(receiver_vm.v2x_manager, 'scheduler', None)
        if scheduler is None:
            continue
        schedule = {}
        for offset, sender_id in enumerate(sorted(clean_selection)):
            link = (int(sender_id), int(receiver_id))
            schedule[link] = channel_allocation.get(link, offset)
        scheduler.set_strategies(schedule)


def estimate_eval_frame_comm_time_ms(eval_frame, metadata, bandwidth_mhz,
                                     num_channels, channel_model=None):
    """Estimate transfer time for an evaluated frame sample."""
    metadata = metadata or {}
    receiver_id = int(metadata.get('receiver_id'))
    channel_model = channel_model or build_channel_model(
        mode='logical',
        bandwidth_mhz=bandwidth_mhz or 20.0,
        num_channels=num_channels or 1)
    channel_allocation = metadata.get('channel_allocation', {}) or {}
    uploaded_sources = [
        int(source_id)
        for source_id in metadata.get('source_cav_ids', [])
        if int(source_id) != receiver_id
    ]
    if not uploaded_sources:
        return 0.0
    if channel_allocation:
        max_seconds = 0.0
        for source_id in uploaded_sources:
            cav = eval_frame.get(source_id)
            if cav is None:
                continue
            payload_bytes = int(cav['lidar_np'].nbytes)
            max_seconds = max(
                max_seconds,
                channel_model.payload_time_ms(payload_bytes, 1) / 1000.0)
        return max_seconds * 1000.0
    total_bytes = int(metadata.get('communication_bytes', 0) or 0)
    return channel_model.payload_time_ms(total_bytes, channel_model.num_channels)


def run_pcs_rounds_with_deadline(allocator, world, max_rounds=1,
                                 deadline_ms=None, channel_model=None):
    """Run repeated PCS scheduling rounds within one frame deadline."""
    max_rounds = max(int(max_rounds or 1), 1)
    if max_rounds == 1 and deadline_ms is None:
        allocator.run()
        total_bytes, link_bytes = estimate_grid_selection_payload_bytes(
            world,
            getattr(allocator, 'grid_selection', {}))
        frame_time_ms = estimate_parallel_comm_time_ms(
            link_bytes,
            getattr(allocator, 'resource_sc_nums', {}),
            getattr(allocator, 'bandwidth_all', 20.0 * (10 ** 6)) / (10 ** 6),
            getattr(allocator, 'lambda_subchannels', 1),
            channel_model=channel_model)
        return {
            'pcs_rounds_requested': 1,
            'pcs_rounds_accepted': 1 if total_bytes > 0 else 0,
            'pcs_frame_comm_time_ms': frame_time_ms,
            'pcs_round_comm_time_ms': [frame_time_ms] if total_bytes > 0 else [],
            'pcs_round_comm_bytes': [total_bytes] if total_bytes > 0 else [],
        }

    deadline_ms = None if deadline_ms is None else float(deadline_ms)
    union_grid_selection = {}
    union_strategy = {}
    union_sc_nums = {}
    excluded_receiver_grids = {}
    round_times = []
    round_bytes = []
    accepted_rounds = 0

    for _round_index in range(max_rounds):
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
            getattr(allocator, 'bandwidth_all', 20.0 * (10 ** 6)) / (10 ** 6),
            getattr(allocator, 'lambda_subchannels', 1),
            channel_model=channel_model)
        if payload_bytes <= 0 or round_time_ms <= 0:
            break
        if deadline_ms is not None:
            remaining_ms = deadline_ms - sum(round_times)
            if remaining_ms <= 0:
                break
            if round_time_ms > remaining_ms:
                round_selection = trim_grid_selection_to_deadline(
                    world,
                    round_selection,
                    getattr(allocator, 'resource_sc_nums', {}),
                    getattr(allocator, 'bandwidth_all',
                            20.0 * (10 ** 6)) / (10 ** 6),
                    getattr(allocator, 'lambda_subchannels', 1),
                    remaining_ms,
                    channel_model=channel_model)
                payload_bytes, link_bytes = (
                    estimate_grid_selection_payload_bytes(
                        world,
                        round_selection))
                round_time_ms = estimate_parallel_comm_time_ms(
                    link_bytes,
                    getattr(allocator, 'resource_sc_nums', {}),
                    getattr(allocator, 'bandwidth_all',
                            20.0 * (10 ** 6)) / (10 ** 6),
                    getattr(allocator, 'lambda_subchannels', 1),
                    channel_model=channel_model)
                if payload_bytes <= 0 or round_time_ms <= 0:
                    break
                if sum(round_times) + round_time_ms > deadline_ms + 1e-6:
                    break

        merge_grid_selection(union_grid_selection, round_selection)
        selected_links = set()
        for receiver_id, sender_grids in round_selection.items():
            for sender_id, grid_ids in sender_grids.items():
                if grid_ids:
                    selected_links.add((int(sender_id), int(receiver_id)))
        for link, start_idx in getattr(allocator, 'resource_strategy',
                                       {}).items():
            normalized_link = (int(link[0]), int(link[1]))
            if normalized_link in selected_links:
                union_strategy[normalized_link] = start_idx
        for link, sc_num in getattr(allocator, 'resource_sc_nums',
                                    {}).items():
            normalized_link = (int(link[0]), int(link[1]))
            if normalized_link in selected_links:
                union_sc_nums[normalized_link] = max(
                    int(sc_num),
                    int(union_sc_nums.get(normalized_link, 1)))
        for receiver_id, sender_grids in round_selection.items():
            excluded_receiver_grids.setdefault(receiver_id, set())
            for grid_ids in sender_grids.values():
                excluded_receiver_grids[receiver_id].update(grid_ids)
        round_times.append(round_time_ms)
        round_bytes.append(payload_bytes)
        accepted_rounds += 1

    allocator.clear_resource_allocation_strategy()
    allocator.grid_selection = union_grid_selection
    allocator.resource_strategy = union_strategy
    allocator.resource_sc_nums = union_sc_nums
    allocator.excluded_receiver_grids = {}
    allocator.update_resource_allocation_strategy()
    return {
        'pcs_rounds_requested': max_rounds,
        'pcs_rounds_accepted': accepted_rounds,
        'pcs_frame_comm_time_ms': sum(round_times),
        'pcs_round_comm_time_ms': round_times,
        'pcs_round_comm_bytes': round_bytes,
    }


def candidate_grids_for_sender(head_vm, sender_vm):
    head_lidar = head_vm.perception_manager.lidar
    sender_lidar = sender_vm.perception_manager.lidar
    weak_head_grids = head_lidar.req_grids - head_lidar.high_density_grids
    candidates = sender_lidar.sens_grids & weak_head_grids
    if not candidates:
        candidates = sender_lidar.sens_grids
    return candidates


def receiver_blind_grids(head_vm):
    head_lidar = head_vm.perception_manager.lidar
    blind_grids = head_lidar.req_grids - head_lidar.high_density_grids
    if not blind_grids:
        blind_grids = head_lidar.req_grids
    return blind_grids


def randomize_scheduled_grid_selection(world, clusters, timestamp):
    """Replace selected grids while preserving SGCP scheduled links/counts."""
    for cluster in clusters:
        head_id = int(cluster.head_id)
        head_vm = world.get_vehicle_manager(head_id)
        if head_vm is None:
            continue
        co_manager = head_vm.perception_manager.co_manager
        current_selection = getattr(co_manager, 'grid_selection', {}) or {}
        randomized = {}
        for sender_id, grid_ids in current_selection.items():
            sender_id = int(sender_id)
            sender_vm = world.get_vehicle_manager(sender_id)
            if sender_vm is None:
                continue
            candidates = sorted(candidate_grids_for_sender(head_vm, sender_vm))
            if not candidates:
                continue
            count = min(len(grid_ids), len(candidates))
            rng = random.Random('%s-%s-%s' % (timestamp, head_id, sender_id))
            randomized[sender_id] = rng.sample(candidates, count)
        co_manager.clear_grid_selection()
        co_manager.set_grid_selection(randomized)


def grid_center_from_id(grid_id, grid_size):
    try:
        x_idx, y_idx = [int(item) for item in str(grid_id).split('_')]
    except (TypeError, ValueError):
        return None
    return (
        x_idx * grid_size + grid_size / 2.0,
        y_idx * grid_size + grid_size / 2.0,
    )


def squared_distance(point_a, point_b):
    return (
        (point_a[0] - point_b[0]) ** 2 +
        (point_a[1] - point_b[1]) ** 2)


def select_spatially_diverse_grids(head_vm, sender_vm, candidates, count):
    if count <= 0 or not candidates:
        return []
    lidar = sender_vm.perception_manager.lidar
    grid_size = lidar.grid_size
    remaining = set(candidates)
    selected = []

    def density(grid_id):
        return lidar.get_grid_density(grid_id)

    first = max(
        remaining,
        key=lambda grid_id: (density(grid_id), str(grid_id)))
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < count:
        selected_centers = [
            grid_center_from_id(grid_id, grid_size)
            for grid_id in selected
        ]
        selected_centers = [
            center for center in selected_centers if center is not None
        ]

        def diversity_score(grid_id):
            center = grid_center_from_id(grid_id, grid_size)
            if center is None or not selected_centers:
                min_distance = 0.0
            else:
                min_distance = min(
                    squared_distance(center, selected_center)
                    for selected_center in selected_centers)
            return (density(grid_id) + 1e-6) * (1.0 + min_distance / 10000.0)

        best_grid = max(
            remaining,
            key=lambda grid_id: (diversity_score(grid_id), str(grid_id)))
        selected.append(best_grid)
        remaining.remove(best_grid)
    return selected


def grid_index_from_id(grid_id):
    try:
        return tuple(int(item) for item in str(grid_id).split('_'))
    except (TypeError, ValueError):
        return None


def grid_l1_distance(grid_a, grid_b):
    index_a = grid_index_from_id(grid_a)
    index_b = grid_index_from_id(grid_b)
    if index_a is None or index_b is None:
        return 999999
    return abs(index_a[0] - index_b[0]) + abs(index_a[1] - index_b[1])


def select_object_clustered_grids(head_vm, sender_vm, candidates, count):
    """Select compact high-density grid patches as object-level proxies."""
    if count <= 0 or not candidates:
        return []
    lidar = sender_vm.perception_manager.lidar
    remaining = set(candidates)
    selected = []

    def density(grid_id):
        return lidar.get_grid_density(grid_id)

    while remaining and len(selected) < count:
        if not selected:
            best_grid = max(
                remaining,
                key=lambda grid_id: (density(grid_id), str(grid_id)))
        else:
            best_grid = max(
                remaining,
                key=lambda grid_id: (
                    density(grid_id) /
                    (1.0 + min(grid_l1_distance(grid_id, selected_grid)
                               for selected_grid in selected)),
                    density(grid_id),
                    str(grid_id)))
        selected.append(best_grid)
        remaining.remove(best_grid)
    return selected


def diversify_scheduled_grid_selection(world, clusters):
    """Replace selected grids with deterministic density-aware spatial cover."""
    for cluster in clusters:
        head_id = int(cluster.head_id)
        head_vm = world.get_vehicle_manager(head_id)
        if head_vm is None:
            continue
        co_manager = head_vm.perception_manager.co_manager
        current_selection = getattr(co_manager, 'grid_selection', {}) or {}
        diversified = {}
        for sender_id, grid_ids in current_selection.items():
            sender_id = int(sender_id)
            sender_vm = world.get_vehicle_manager(sender_id)
            if sender_vm is None:
                continue
            candidates = sorted(candidate_grids_for_sender(head_vm, sender_vm))
            if not candidates:
                continue
            count = min(len(grid_ids), len(candidates))
            diversified[sender_id] = select_spatially_diverse_grids(
                head_vm,
                sender_vm,
                candidates,
                count)
        co_manager.clear_grid_selection()
        co_manager.set_grid_selection(diversified)


def cluster_scheduled_grid_selection(world, clusters):
    """Replace selected grids with compact high-density object proxies."""
    for cluster in clusters:
        head_id = int(cluster.head_id)
        head_vm = world.get_vehicle_manager(head_id)
        if head_vm is None:
            continue
        co_manager = head_vm.perception_manager.co_manager
        current_selection = getattr(co_manager, 'grid_selection', {}) or {}
        clustered = {}
        for sender_id, grid_ids in current_selection.items():
            sender_id = int(sender_id)
            sender_vm = world.get_vehicle_manager(sender_id)
            if sender_vm is None:
                continue
            candidates = sorted(candidate_grids_for_sender(head_vm, sender_vm))
            if not candidates:
                continue
            count = min(len(grid_ids), len(candidates))
            clustered[sender_id] = select_object_clustered_grids(
                head_vm,
                sender_vm,
                candidates,
                count)
        co_manager.clear_grid_selection()
        co_manager.set_grid_selection(clustered)


def quality_ratio(stat):
    gt_sum = float(stat.get('quality_gt_sum', 0))
    if gt_sum <= 0:
        return None
    return float(stat.get('quality_pred_sum', 0)) / gt_sum


def apply_persistent_coverage_fallback(world, clusters, coverage_state,
                                       quality_aware=False):
    """Swap in repeatedly unscheduled members without changing link count.

    This is a diagnostic/algorithm probe for the 10ch case: global channel
    capacity is already saturated, so a coverage repair must reuse an existing
    subchannel rather than adding another request.
    """
    if coverage_state is None:
        return 0
    replacements = 0
    for cluster in clusters:
        head_id = int(cluster.head_id)
        head_vm = world.get_vehicle_manager(head_id)
        if head_vm is None:
            continue
        co_manager = head_vm.perception_manager.co_manager
        current_selection = {
            int(sender_id): list(grid_ids)
            for sender_id, grid_ids in (
                getattr(co_manager, 'grid_selection', {}) or {}).items()
        }
        if not current_selection:
            continue
        scheduler = head_vm.v2x_manager.scheduler
        channel_allocation = getattr(scheduler, 'channel_allocation', {})
        non_head_members = [
            int(member_id) for member_id in sorted(cluster.members)
            if int(member_id) != head_id
        ]
        unscheduled = [
            member_id for member_id in non_head_members
            if member_id not in current_selection
        ]
        if not unscheduled:
            continue

        def member_deficit(member_id):
            stat = coverage_state.get(member_id, {})
            return (
                int(stat.get('unscheduled_frames', 0)) -
                int(stat.get('uploaded_frames', 0)),
                int(stat.get('unscheduled_frames', 0)),
                -int(stat.get('uploaded_frames', 0)),
                -member_id,
            )

        candidate_id = max(unscheduled, key=member_deficit)
        candidate_deficit = member_deficit(candidate_id)
        if candidate_deficit[0] < 2:
            continue

        def scheduled_score(member_id):
            stat = coverage_state.get(member_id, {})
            grid_count = len(current_selection.get(member_id, []))
            return (
                int(stat.get('uploaded_frames', 0)) -
                int(stat.get('unscheduled_frames', 0)),
                grid_count,
                -member_id,
            )

        replaced_id = max(current_selection.keys(), key=scheduled_score)
        if member_deficit(replaced_id)[0] >= candidate_deficit[0]:
            continue
        if quality_aware:
            candidate_stat = coverage_state.get(candidate_id, {})
            replaced_stat = coverage_state.get(replaced_id, {})
            candidate_quality = quality_ratio(candidate_stat)
            replaced_quality = quality_ratio(replaced_stat)
            if (candidate_quality is None or
                    int(candidate_stat.get('quality_rows', 0)) < 2):
                continue
            if (replaced_quality is not None and
                    candidate_quality < 0.9 * replaced_quality):
                continue
            if candidate_quality < 0.25:
                continue
        replaced_grids = current_selection.get(replaced_id, [])
        if not replaced_grids:
            continue
        candidate_vm = world.get_vehicle_manager(candidate_id)
        if candidate_vm is None:
            continue
        candidates = sorted(candidate_grids_for_sender(head_vm, candidate_vm))
        if not candidates:
            continue
        grid_count = min(len(replaced_grids), len(candidates))
        new_grids = select_spatially_diverse_grids(
            head_vm,
            candidate_vm,
            candidates,
            grid_count)
        if not new_grids:
            continue
        replaced_vm = world.get_vehicle_manager(replaced_id)
        replaced_density = 0.0
        if replaced_vm is not None:
            replaced_density = sum(
                replaced_vm.perception_manager.lidar.get_grid_density(grid_id)
                for grid_id in replaced_grids)
        candidate_density = sum(
            candidate_vm.perception_manager.lidar.get_grid_density(grid_id)
            for grid_id in new_grids)
        if (replaced_density > 0 and
                candidate_density < 0.8 * replaced_density):
            continue
        old_channel = channel_allocation.pop((replaced_id, head_id), None)
        if old_channel is None:
            continue
        channel_allocation[(candidate_id, head_id)] = old_channel
        current_selection.pop(replaced_id, None)
        current_selection[candidate_id] = new_grids
        co_manager.clear_grid_selection()
        co_manager.set_grid_selection(current_selection)
        replacements += 1
    return replacements


def load_sgcp_routing_hints(path):
    """Load diagnostic target-to-head routing hints.

    This is intentionally an offline/debug probe. It consumes diagnostics that
    may include GT-derived object ids or full-reference comparisons, so it must
    not be reported as a deployable algorithm.
    """
    if not path:
        return None
    hints = defaultdict(list)
    with open(path, newline='') as stream:
        for row in csv.DictReader(stream):
            timestamp = str(row.get('timestamp', ''))
            object_grid = str(row.get('object_grid_id', '')).strip()
            if not timestamp or not object_grid:
                continue
            try:
                receiver_id = int(float(row.get('nearest_head', '')))
            except (TypeError, ValueError):
                continue
            sender_value = (
                row.get('best_raw_cav_id_m0p0') or
                row.get('best_raw_cav_id_m2p0') or
                row.get('nearest_cav'))
            try:
                sender_id = int(float(sender_value))
            except (TypeError, ValueError):
                continue
            try:
                ratio = float(row.get('sgcp_full_box_point_ratio_m0p0', 1.0)
                              or 1.0)
            except (TypeError, ValueError):
                ratio = 1.0
            try:
                full_points = int(float(
                    row.get('full_reference_box_points_m0p0', 0) or 0))
            except (TypeError, ValueError):
                full_points = 0
            try:
                raw_points = int(float(
                    row.get('best_raw_cav_box_points_m0p0', 0) or 0))
            except (TypeError, ValueError):
                raw_points = 0
            hints[timestamp].append({
                'timestamp': timestamp,
                'receiver_id': receiver_id,
                'sender_id': sender_id,
                'object_grid_id': object_grid,
                'object_id': row.get('object_id', ''),
                'ratio': ratio,
                'full_points': full_points,
                'raw_points': raw_points,
                'score': (
                    (1.0 - min(max(ratio, 0.0), 1.0)) *
                    max(full_points, 1) +
                    0.2 * raw_points),
            })
    for timestamp in list(hints.keys()):
        hints[timestamp] = sorted(
            hints[timestamp],
            key=lambda item: (
                item['score'],
                item['full_points'],
                item['raw_points'],
                -item['sender_id']),
            reverse=True)
    return hints


def neighboring_grid_ids(grid_id, radius=1):
    index = grid_index_from_id(grid_id)
    if index is None:
        return [grid_id]
    gx, gy = index
    grids = []
    for dist in range(radius + 1):
        for dx in range(-dist, dist + 1):
            for dy in range(-dist, dist + 1):
                if abs(dx) + abs(dy) != dist:
                    continue
                grids.append('%d_%d' % (gx + dx, gy + dy))
    return grids


def hinted_grid_selection(head_vm, sender_vm, object_grid_id, count):
    sender_lidar = sender_vm.perception_manager.lidar
    candidates = set(candidate_grids_for_sender(head_vm, sender_vm))
    if not candidates:
        candidates = set(sender_lidar.sens_grids)
    selected = []
    for grid_id in neighboring_grid_ids(object_grid_id, radius=2):
        if grid_id in candidates and sender_lidar.get_grid_density(grid_id) > 0:
            selected.append(grid_id)
        if len(selected) >= count:
            return selected
    remaining = [
        grid for grid in candidates
        if grid not in selected and sender_lidar.get_grid_density(grid) > 0
    ]
    remaining = sorted(
        remaining,
        key=lambda grid: (
            -grid_l1_distance(grid, object_grid_id),
            sender_lidar.get_grid_density(grid),
            str(grid)),
        reverse=True)
    for grid_id in remaining:
        selected.append(grid_id)
        if len(selected) >= count:
            break
    return selected


def merge_hinted_grid_selection(head_vm, sender_vm, object_grid_id,
                                existing_grids):
    """Preserve detector context while forcing a small target-grid hint."""
    count = len(existing_grids)
    if count <= 0:
        return []
    sender_lidar = sender_vm.perception_manager.lidar
    candidates = set(candidate_grids_for_sender(head_vm, sender_vm))
    if not candidates:
        candidates = set(sender_lidar.sens_grids)
    hint_grids = []
    for grid_id in neighboring_grid_ids(object_grid_id, radius=1):
        if grid_id in candidates and sender_lidar.get_grid_density(grid_id) > 0:
            hint_grids.append(grid_id)
        if len(hint_grids) >= min(3, count):
            break
    if not hint_grids:
        return list(existing_grids)
    selected = []
    selected_set = set()
    for grid_id in hint_grids:
        if grid_id not in selected_set:
            selected.append(grid_id)
            selected_set.add(grid_id)
    preserved = sorted(
        [grid for grid in existing_grids if grid not in selected_set],
        key=lambda grid: (
            sender_lidar.get_grid_density(grid),
            str(grid)),
        reverse=True)
    for grid_id in preserved:
        selected.append(grid_id)
        selected_set.add(grid_id)
        if len(selected) >= count:
            return selected
    remaining = sorted(
        [grid for grid in candidates if grid not in selected_set],
        key=lambda grid: (
            sender_lidar.get_grid_density(grid),
            str(grid)),
        reverse=True)
    for grid_id in remaining:
        if sender_lidar.get_grid_density(grid_id) <= 0:
            continue
        selected.append(grid_id)
        if len(selected) >= count:
            break
    return selected


def apply_diagnostic_routing_hints(world, timestamp, routing_hints,
                                   max_per_frame=1):
    """Apply oracle/debug target-to-head route replacements."""
    if not routing_hints or max_per_frame <= 0:
        return 0
    hints = routing_hints.get(str(timestamp), [])
    if not hints:
        return 0
    applied = 0
    used_receivers = set()
    used_senders = set()
    for hint in hints:
        if applied >= max_per_frame:
            break
        receiver_id = int(hint['receiver_id'])
        sender_id = int(hint['sender_id'])
        if receiver_id in used_receivers or sender_id in used_senders:
            continue
        if receiver_id == sender_id:
            continue
        receiver_vm = world.get_vehicle_manager(receiver_id)
        sender_vm = world.get_vehicle_manager(sender_id)
        if receiver_vm is None or sender_vm is None:
            continue
        co_manager = receiver_vm.perception_manager.co_manager
        current_selection = {
            int(src): list(grids)
            for src, grids in (
                getattr(co_manager, 'grid_selection', {}) or {}).items()
        }
        if not current_selection:
            continue
        scheduler = receiver_vm.v2x_manager.scheduler
        channel_allocation = getattr(scheduler, 'channel_allocation', {})
        if sender_id in current_selection:
            new_grids = merge_hinted_grid_selection(
                receiver_vm,
                sender_vm,
                hint['object_grid_id'],
                current_selection[sender_id])
            if not new_grids:
                continue
            current_selection[sender_id] = new_grids
            co_manager.clear_grid_selection()
            co_manager.set_grid_selection(current_selection)
            applied += 1
            used_receivers.add(receiver_id)
            used_senders.add(sender_id)
            continue

        def replace_score(src_id):
            grids = current_selection.get(src_id, [])
            src_vm = world.get_vehicle_manager(src_id)
            if src_vm is None:
                return (float('inf'), src_id)
            density_sum = sum(
                src_vm.perception_manager.lidar.get_grid_density(grid)
                for grid in grids)
            return (density_sum, src_id)

        replaceable = [
            src_id for src_id in current_selection.keys()
            if (src_id, receiver_id) in channel_allocation
        ]
        if not replaceable:
            continue
        replaced_id = min(replaceable, key=replace_score)
        replaced_grids = current_selection.get(replaced_id, [])
        count = max(1, len(replaced_grids))
        new_grids = hinted_grid_selection(
            receiver_vm,
            sender_vm,
            hint['object_grid_id'],
            count)
        if not new_grids:
            continue
        old_channel = channel_allocation.pop((replaced_id, receiver_id), None)
        if old_channel is None:
            continue
        channel_allocation[(sender_id, receiver_id)] = old_channel
        current_selection.pop(replaced_id, None)
        current_selection[sender_id] = new_grids
        co_manager.clear_grid_selection()
        co_manager.set_grid_selection(current_selection)
        applied += 1
        used_receivers.add(receiver_id)
        used_senders.add(sender_id)
    return applied


def update_coverage_state_from_items(coverage_state, constrained_items):
    if coverage_state is None:
        return
    for _, metadata in constrained_items:
        receiver_id = int(metadata.get('receiver_id'))
        members = set(int(item) for item in metadata.get(
            'cluster_member_ids', []))
        sources = set(int(item) for item in metadata.get(
            'source_cav_ids', []))
        uploaded = sources - {receiver_id}
        for member_id in members:
            stat = coverage_state.setdefault(member_id, {
                'uploaded_frames': 0,
                'unscheduled_frames': 0,
                'fused_frames': 0,
                'quality_rows': 0,
                'quality_pred_sum': 0,
                'quality_gt_sum': 0,
            })
            if member_id in sources:
                stat['fused_frames'] += 1
            if member_id in uploaded:
                stat['uploaded_frames'] += 1
            elif member_id != receiver_id:
                stat['unscheduled_frames'] += 1


def update_coverage_quality_state(coverage_state, metadata, pred_count,
                                  gt_count):
    if coverage_state is None or metadata is None:
        return
    if gt_count is None or gt_count <= 0:
        return
    receiver_id = int(metadata.get('receiver_id'))
    source_ids = set(int(item) for item in metadata.get(
        'source_cav_ids', []))
    uploaded_ids = source_ids - {receiver_id}
    for source_id in uploaded_ids:
        stat = coverage_state.setdefault(source_id, {
            'uploaded_frames': 0,
            'unscheduled_frames': 0,
            'fused_frames': 0,
            'quality_rows': 0,
            'quality_pred_sum': 0,
            'quality_gt_sum': 0,
        })
        stat['quality_rows'] += 1
        stat['quality_pred_sum'] += int(pred_count or 0)
        stat['quality_gt_sum'] += int(gt_count or 0)


def cluster_templates_from_clusters(clusters):
    templates = []
    for cluster in clusters:
        templates.append({
            'head_id': int(cluster.head_id),
            'member_ids': sorted(int(member_id)
                                 for member_id in cluster.members),
        })
    return templates


def build_fixed_clusters(world, cluster_templates):
    common.Vehicle_Grid.initialize(world)
    clusters = []
    for template in cluster_templates:
        member_ids = [
            int(member_id)
            for member_id in template['member_ids']
            if world.get_vehicle_manager(int(member_id)) is not None
        ]
        if not member_ids:
            continue
        cluster = common.Cluster(set(member_ids))
        fixed_head = int(template['head_id'])
        if world.get_vehicle_manager(fixed_head) is not None:
            cluster.head_id = fixed_head
            cluster.grid_bits = cluster.compute_grid_bits()
        clusters.append(cluster)
    return clusters


def _vehicle_xy(vehicle_id):
    vehicle = common.global_vehicles[int(vehicle_id)]
    location = vehicle.get_position().location
    return float(location.x), float(location.y)


def _vehicle_distance(first_id, second_id):
    x1, y1 = _vehicle_xy(first_id)
    x2, y2 = _vehicle_xy(second_id)
    return math.hypot(x1 - x2, y1 - y2)


def _vehicle_velocity_xy(vehicle_id):
    vehicle = common.global_vehicles[int(vehicle_id)]
    speed = float(vehicle.get_speed())
    direction = vehicle.get_direction()
    return speed * float(direction[0]), speed * float(direction[1])


def _relative_speed(first_id, second_id):
    vx1, vy1 = _vehicle_velocity_xy(first_id)
    vx2, vy2 = _vehicle_velocity_xy(second_id)
    return math.hypot(vx1 - vx2, vy1 - vy2)


def _center_head_id(member_ids):
    member_ids = [int(item) for item in member_ids]
    if not member_ids:
        return None
    positions = {vid: _vehicle_xy(vid) for vid in member_ids}
    center_x = sum(item[0] for item in positions.values()) / len(member_ids)
    center_y = sum(item[1] for item in positions.values()) / len(member_ids)
    return min(
        member_ids,
        key=lambda vid: (
            math.hypot(positions[vid][0] - center_x,
                       positions[vid][1] - center_y),
            vid))


def _cluster_density_score(vehicle_id):
    vehicle = common.global_vehicles[int(vehicle_id)]
    density_dict = getattr(vehicle, 'grid_density_dict', {}) or {}
    high_density = getattr(vehicle, 'high_density_grids', set()) or set()
    sens_grids = getattr(vehicle, 'sens_grids', set()) or set()
    density_sum = sum(float(density_dict.get(grid_id, 0.0))
                      for grid_id in high_density)
    return (
        density_sum,
        len(high_density),
        len(sens_grids),
        -int(vehicle_id))


def _make_offline_cluster(member_ids):
    cluster = common.Cluster(set(int(item) for item in member_ids))
    head_id = _center_head_id(cluster.members)
    if head_id is not None:
        cluster.head_id = int(head_id)
        cluster.grid_bits = cluster.compute_grid_bits()
    return cluster


def build_heuristic_clusters(world, clustering, n_max=None, timestamp=None):
    """Build deterministic clustering baselines for offline ablation.

    These baselines intentionally share the same center-nearest head election
    rule as the SGCP coalition implementation; only membership formation is
    replaced.
    """
    common.Vehicle_Grid.initialize(world)
    vehicle_ids = sorted(int(item) for item in common.global_vehicles)
    if not vehicle_ids:
        return []
    capacity = int(n_max or common.Params().N_max)
    capacity = max(1, capacity)
    if clustering == 'random_balanced':
        seed = '%s:%s' % (
            timestamp or 'all',
            ','.join(str(item) for item in vehicle_ids))
        shuffled = list(vehicle_ids)
        random.Random(seed).shuffle(shuffled)
        return [
            _make_offline_cluster(shuffled[index:index + capacity])
            for index in range(0, len(shuffled), capacity)
        ]

    unassigned = set(vehicle_ids)
    clusters = []
    while unassigned:
        if clustering == 'distance_greedy':
            head_id = min(
                unassigned,
                key=lambda vid: (
                    sum(_vehicle_distance(vid, other)
                        for other in unassigned if other != vid),
                    vid))
            members = [head_id]
            candidates = sorted(
                (vid for vid in unassigned if vid != head_id),
                key=lambda vid: (_vehicle_distance(head_id, vid), vid))
            members.extend(candidates[:capacity - 1])
        elif clustering == 'density_greedy_cluster':
            head_id = max(unassigned, key=_cluster_density_score)
            members = [head_id]
            covered = set(common.global_vehicles[head_id].sens_grids)
            while len(members) < capacity:
                candidates = [vid for vid in unassigned if vid not in members]
                if not candidates:
                    break
                best_vid = max(
                    candidates,
                    key=lambda vid: (
                        len(common.global_vehicles[vid].sens_grids - covered),
                        _cluster_density_score(vid),
                        -_vehicle_distance(head_id, vid),
                        -vid))
                members.append(best_vid)
                covered |= common.global_vehicles[best_vid].sens_grids
        elif clustering == 'mobility_stability_greedy':
            head_id = min(
                unassigned,
                key=lambda vid: (
                    sum(_vehicle_distance(vid, other)
                        for other in unassigned if other != vid),
                    common.global_vehicles[vid].get_speed(),
                    vid))
            members = [head_id]
            covered = set(common.global_vehicles[head_id].sens_grids)
            while len(members) < capacity:
                candidates = [vid for vid in unassigned if vid not in members]
                if not candidates:
                    break
                best_vid = min(
                    candidates,
                    key=lambda vid: (
                        _relative_speed(head_id, vid),
                        _vehicle_distance(head_id, vid),
                        -len(common.global_vehicles[vid].sens_grids - covered),
                        vid))
                members.append(best_vid)
                covered |= common.global_vehicles[best_vid].sens_grids
        else:
            raise ValueError('Unknown heuristic clustering: %s' % clustering)
        unassigned -= set(members)
        clusters.append(_make_offline_cluster(members))
    return clusters


def is_offline_constructed_clustering(clustering):
    return clustering in [
        'random_balanced',
        'distance_greedy',
        'density_greedy_cluster',
        'mobility_stability_greedy',
        'seac_social_adaptive',
        'hho_vanet',
    ]


def build_offline_constructed_clusters(world, clustering, n_max=None,
                                       timestamp=None):
    if clustering in ['seac_social_adaptive', 'hho_vanet']:
        return build_paper_baseline_clusters(
            world,
            clustering,
            n_max=n_max,
            timestamp=timestamp)
    return build_heuristic_clusters(
        world,
        clustering,
        n_max=n_max,
        timestamp=timestamp)


def scheduled_receiver_ids(world, fallback_clusters=None):
    receiver_ids = set()
    for vm in world.get_vehicle_managers().values():
        scheduler = getattr(vm.v2x_manager, 'scheduler', None)
        channel_allocation = getattr(scheduler, 'channel_allocation', {})
        for link in channel_allocation.keys():
            try:
                _, target_id = link
            except (TypeError, ValueError):
                continue
            receiver_ids.add(int(target_id))
    if not receiver_ids and fallback_clusters:
        receiver_ids.update(int(cluster.head_id) for cluster in fallback_clusters)
    return sorted(receiver_ids)


def apply_sgcp_constraint(frame, protocol, ego_cav_id, resource_allocation,
                          receiver_policy, t_min_stab=None,
                          clustering='coalition_game', n_max=None,
                          rho_th=None, num_channels=None,
                          bandwidth_mhz=None, upload_mode='grid',
                          grid_selection_mode='utility',
                          grid_score_mode='utility',
                          timestamp=None,
                          fixed_cluster_templates=None,
                          head_rb_budget=None,
                          pcs_blind_spot_min_division=None,
                          pcs_min_overlap_grids=None,
                          pcs_blind_spot_radius=None,
                          pcs_min_spot_grids=None,
                          pcs_communication_range_m=None,
                          pcs_frame_rounds=1,
                          pcs_frame_deadline_ms=None,
                          coverage_fallback='none',
                          coverage_state=None,
                          max_upload_points_per_source=None,
                          routing_hints=None,
                          routing_hints_max_per_frame=1,
                          channel_model=None,
                          max_senders_per_receiver=1,
                          sgcp_frame_mbps_budget=None):
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=ego_cav_id,
        protocol=protocol,
        density_threshold=rho_th)
    if clustering in ['coalition_game', 'fixed_first_frame']:
        clustering_algorithm = CoalitionGame(world)
    elif clustering == 'cov_coalition_game':
        clustering_algorithm = COVCoalitionGame(world)
    elif clustering == 'singleton':
        clustering_algorithm = NaiveCluster(world, all_in_one=False)
    elif clustering == 'all_in_one':
        clustering_algorithm = NaiveCluster(world, all_in_one=True)
    elif is_offline_constructed_clustering(clustering):
        clustering_algorithm = None
    else:
        raise ValueError('Unknown clustering algorithm: %s' % clustering)
    if (clustering_algorithm is not None and t_min_stab is not None and
            hasattr(clustering_algorithm, 'p')):
        clustering_algorithm.p.T_min_stab = t_min_stab
    if (clustering_algorithm is not None and n_max is not None and
            hasattr(clustering_algorithm, 'p')):
        clustering_algorithm.p.N_max = n_max
    if clustering == 'fixed_first_frame' and fixed_cluster_templates:
        clusters = build_fixed_clusters(world, fixed_cluster_templates)
    elif is_offline_constructed_clustering(clustering):
        clusters = build_offline_constructed_clusters(
            world,
            clustering,
            n_max=n_max,
            timestamp=timestamp)
    else:
        clusters = clustering_algorithm.run()
        if (clustering == 'fixed_first_frame' and
                fixed_cluster_templates is not None and
                not fixed_cluster_templates):
            fixed_cluster_templates.extend(
                cluster_templates_from_clusters(clusters))
    apply_cluster_state(world, clusters)
    if channel_model is None:
        channel_model = build_channel_model(
            mode='logical',
            bandwidth_mhz=bandwidth_mhz or 20.0,
            num_channels=num_channels or world.network_manager.subchannel_num,
            frame_deadline_s=getattr(world.network_manager, 'time_slot', 0.1))
    allocator = build_resource_allocator(resource_allocation, world)
    if hasattr(allocator, 'grid_score_mode'):
        allocator.grid_score_mode = grid_score_mode
    apply_resource_overrides(
        allocator,
        world,
        num_channels=num_channels,
        bandwidth_mhz=bandwidth_mhz,
        head_rb_budget=head_rb_budget,
        pcs_blind_spot_min_division=pcs_blind_spot_min_division,
        pcs_min_overlap_grids=pcs_min_overlap_grids,
        pcs_blind_spot_radius=pcs_blind_spot_radius,
        pcs_min_spot_grids=pcs_min_spot_grids,
        pcs_communication_range_m=pcs_communication_range_m,
        channel_model=channel_model,
        max_senders_per_receiver=max_senders_per_receiver)
    allocator.set_clusters(clusters)
    pcs_round_metadata = {}
    if resource_allocation == 'fullperception_pcs':
        pcs_round_metadata = run_pcs_rounds_with_deadline(
            allocator,
            world,
            max_rounds=pcs_frame_rounds,
            deadline_ms=pcs_frame_deadline_ms,
            channel_model=channel_model)
    else:
        allocator.run()
    if grid_selection_mode == 'random':
        randomize_scheduled_grid_selection(world, clusters, timestamp)
    elif grid_selection_mode == 'spatial_diverse':
        diversify_scheduled_grid_selection(world, clusters)
    elif grid_selection_mode == 'object_clustered':
        cluster_scheduled_grid_selection(world, clusters)
    coverage_fallback_replacements = 0
    if coverage_fallback in ['persistent', 'quality_persistent']:
        coverage_fallback_replacements = apply_persistent_coverage_fallback(
            world,
            clusters,
            coverage_state,
            quality_aware=coverage_fallback == 'quality_persistent')
    routing_hint_replacements = apply_diagnostic_routing_hints(
        world,
        timestamp,
        routing_hints,
        max_per_frame=routing_hints_max_per_frame)
    frame_budget_bytes = ''
    frame_budget_original_bytes = ''
    frame_budget_admitted_bytes = ''
    if sgcp_frame_mbps_budget is not None:
        if sgcp_frame_mbps_budget <= 0:
            raise ValueError('--sgcp-frame-mbps-budget must be positive')
        frame_budget_bytes = int(
            float(sgcp_frame_mbps_budget) * 1e6 * 0.1 / 8.0)
        current_selection = collect_receiver_grid_selection(world, clusters)
        current_channel_allocation = {}
        for vm in world.get_vehicle_managers().values():
            scheduler = getattr(vm.v2x_manager, 'scheduler', None)
            current_channel_allocation.update(
                getattr(scheduler, 'channel_allocation', {}) or {})
        trimmed_selection, frame_budget_original_bytes, \
            frame_budget_admitted_bytes = trim_grid_selection_to_payload_budget(
                world,
                current_selection,
                frame_budget_bytes,
                strategies=getattr(allocator, 'strategies', None))
        apply_grid_selection_to_world(
            world,
            trimmed_selection,
            channel_allocation=current_channel_allocation)
    if receiver_policy == 'all-cluster-heads':
        receiver_ids = sorted(int(cluster.head_id) for cluster in clusters)
    elif receiver_policy == 'all-scheduled-receivers':
        receiver_ids = scheduled_receiver_ids(world, clusters)
    elif receiver_policy == 'all-cavs':
        receiver_ids = sorted(int(cav_id)
                              for cav_id in world.get_vehicle_managers())
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
            receiver_id,
            upload_mode=upload_mode,
            max_upload_points_per_source=max_upload_points_per_source)
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
        metadata.update(channel_model.to_metadata())
        metadata['head_rb_budget'] = (
            getattr(allocator.p, 'head_rb_budget', None)
            if hasattr(allocator, 'p') else None)
        metadata['bandwidth_mhz'] = (
            getattr(allocator.p, 'bandwidth_all', 0.0) / (10 ** 6)
            if hasattr(allocator, 'p') else channel_model.bandwidth_mhz)
        metadata['upload_mode'] = upload_mode
        metadata['grid_selection_mode'] = grid_selection_mode
        metadata['grid_score_mode'] = grid_score_mode
        metadata['coverage_fallback'] = coverage_fallback
        metadata['coverage_fallback_replacements'] = (
            coverage_fallback_replacements)
        metadata['routing_hint_replacements'] = routing_hint_replacements
        metadata['routing_hints_csv'] = (
            '' if not routing_hints else 'enabled')
        metadata['max_upload_points_per_source'] = (
            max_upload_points_per_source or '')
        metadata['max_senders_per_receiver'] = max(
            1,
            int(max_senders_per_receiver or 1))
        metadata['sgcp_frame_mbps_budget'] = (
            '' if sgcp_frame_mbps_budget is None
            else sgcp_frame_mbps_budget)
        metadata['sgcp_frame_budget_bytes'] = frame_budget_bytes
        metadata['sgcp_frame_budget_original_bytes'] = (
            frame_budget_original_bytes)
        metadata['sgcp_frame_budget_admitted_bytes'] = (
            frame_budget_admitted_bytes)
        metadata['pcs_rounds_requested'] = (
            pcs_round_metadata.get('pcs_rounds_requested', ''))
        metadata['pcs_rounds_accepted'] = (
            pcs_round_metadata.get('pcs_rounds_accepted', ''))
        metadata['frame_comm_time_ms'] = (
            pcs_round_metadata.get('pcs_frame_comm_time_ms', ''))
        metadata['pcs_round_comm_time_ms'] = (
            pcs_round_metadata.get('pcs_round_comm_time_ms', []))
        metadata['pcs_round_comm_bytes'] = (
            pcs_round_metadata.get('pcs_round_comm_bytes', []))
        if metadata.get('frame_comm_time_ms', '') == '':
            metadata['frame_comm_time_ms'] = estimate_eval_frame_comm_time_ms(
                constrained_frame,
                metadata,
                metadata.get('bandwidth_mhz') or bandwidth_mhz,
                metadata.get('num_channels') or num_channels,
                channel_model=channel_model)
        constrained_items.append((constrained_frame, metadata))
    update_coverage_state_from_items(coverage_state, constrained_items)
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


def candidate_member_ids(world, cluster, baseline_name):
    head_id = int(cluster.head_id)
    if baseline_name in ['global_selective_proxy', 'edgecooper']:
        return [
            int(member_id)
            for member_id in sorted(world.get_vehicle_managers().keys())
            if int(member_id) != head_id
        ]
    if baseline_name in ['edgecooper_global', 'edgecooper_global_hd']:
        head_vm = world.get_vehicle_manager(head_id)
        receiver_ids = set(getattr(
            world,
            '_edgecooper_global_receiver_ids',
            set()) or set())
        feasible_members = []
        for member_id in sorted(world.get_vehicle_managers().keys()):
            member_id = int(member_id)
            if member_id == head_id:
                continue
            if baseline_name == 'edgecooper_global_hd' and member_id in receiver_ids:
                continue
            sender_vm = world.get_vehicle_manager(member_id)
            comm_range_m = float(getattr(
                world,
                '_edgecooper_global_comm_range_m',
                EDGECOOPER_GLOBAL_COMM_RANGE_M))
            if vehicle_distance(head_vm, sender_vm) <= comm_range_m:
                feasible_members.append(member_id)
        return feasible_members
    return [
        int(member_id) for member_id in sorted(cluster.members)
        if int(member_id) != head_id
    ]


def density_score_for_member(head_vm, sender_vm):
    candidate_grids = candidate_grids_for_sender(head_vm, sender_vm)
    return sum(
        sender_vm.perception_manager.lidar.get_grid_density(grid_id)
        for grid_id in candidate_grids)


def edgecooper_candidate_grids(head_vm, sender_vm):
    sender_lidar = sender_vm.perception_manager.lidar
    return sender_lidar.sens_grids & receiver_blind_grids(head_vm)


def edgecooper_grid_score(head_vm, sender_vm, grid_id, covered_grids=None):
    sender_lidar = sender_vm.perception_manager.lidar
    head_lidar = head_vm.perception_manager.lidar
    blind_grids = receiver_blind_grids(head_vm)
    sender_density = sender_lidar.get_grid_density(grid_id)
    head_density = head_lidar.get_grid_density(grid_id)
    blind_bonus = 1.0 if grid_id in blind_grids else 0.25
    novelty_bonus = 1.0
    if covered_grids is not None and grid_id in covered_grids:
        novelty_bonus = 0.35
    redundancy_penalty = min(sender_density, head_density)
    return (
        sender_density * blind_bonus * novelty_bonus -
        0.25 * redundancy_penalty)


def pacp_lidar_candidate_grids(head_vm, sender_vm):
    """LiDAR adaptation of PACP BEV-match priority over raw point grids."""
    sender_lidar = sender_vm.perception_manager.lidar
    head_lidar = head_vm.perception_manager.lidar
    blind_grids = receiver_blind_grids(head_vm)
    candidates = sender_lidar.sens_grids & (blind_grids | head_lidar.req_grids)
    if not candidates:
        candidates = sender_lidar.sens_grids
    return candidates


def pacp_lidar_grid_score(head_vm, sender_vm, grid_id, covered_grids=None):
    sender_lidar = sender_vm.perception_manager.lidar
    head_lidar = head_vm.perception_manager.lidar
    covered_grids = set() if covered_grids is None else set(covered_grids)
    sender_density = sender_lidar.get_grid_density(grid_id)
    head_density = head_lidar.get_grid_density(grid_id)
    blind_bonus = 1.0 if grid_id in receiver_blind_grids(head_vm) else 0.35
    overlap_match = min(sender_density, head_density)
    complementarity = max(0.0, sender_density - head_density)
    novelty = 1.0 if grid_id not in covered_grids else 0.25
    # PACP's RGB BEV-match prefers perceptually consistent agents; in LiDAR,
    # overlap_match is the BEV occupancy agreement and complementarity keeps
    # blind-spot recovery from becoming pure redundancy.
    return novelty * (
        0.55 * overlap_match +
        0.90 * blind_bonus * complementarity +
        0.20 * blind_bonus * sender_density)


def pacp_lidar_member_score(head_vm, sender_vm, covered_grids=None,
                            link_quality=None, timestamp=None):
    covered_grids = set() if covered_grids is None else set(covered_grids)
    candidates = set(pacp_lidar_candidate_grids(head_vm, sender_vm))
    if not candidates:
        return 0.0, candidates
    head_id = int(head_vm.vehicle_id)
    sender_id = int(sender_vm.vehicle_id)
    bev_match = 0.0
    complementarity = 0.0
    for grid_id in candidates:
        sender_density = sender_vm.perception_manager.lidar.get_grid_density(
            grid_id)
        head_density = head_vm.perception_manager.lidar.get_grid_density(
            grid_id)
        if grid_id in head_vm.perception_manager.lidar.req_grids:
            bev_match += min(sender_density, head_density)
        if grid_id not in covered_grids:
            complementarity += max(0.0, sender_density - head_density)
    distance = vehicle_distance(head_vm, sender_vm)
    quality = ns3_link_quality(link_quality, timestamp, sender_id, head_id)
    if quality is None:
        quality = 1.0 / (1.0 + distance / 100.0)
    score = (0.60 * bev_match + 1.00 * complementarity) * quality
    score = score / (1.0 + len(candidates) / 200.0)
    return score, candidates


def select_pacp_lidar_members(world, head_vm, members, member_budget,
                              link_quality=None, timestamp=None):
    selected = []
    covered = set()
    remaining = set(members)
    while remaining and len(selected) < member_budget:
        best = None
        for member_id in sorted(remaining):
            sender_vm = world.get_vehicle_manager(member_id)
            score, candidates = pacp_lidar_member_score(
                head_vm,
                sender_vm,
                covered_grids=covered,
                link_quality=link_quality,
                timestamp=timestamp)
            distance = vehicle_distance(head_vm, sender_vm)
            item = (-score, distance, member_id, candidates)
            if best is None or item < best:
                best = item
        if best is None or best[0] >= 0.0:
            break
        _, _, member_id, candidates = best
        selected.append(member_id)
        covered.update(candidates)
        remaining.remove(member_id)
    return selected


def select_pacp_lidar_grids(head_vm, sender_vm, candidates, count,
                            covered_grids=None):
    if count <= 0 or not candidates:
        return []
    covered_grids = set() if covered_grids is None else set(covered_grids)
    remaining = set(candidates)
    selected = []
    while remaining and len(selected) < count:
        best = max(
            remaining,
            key=lambda grid_id: (
                pacp_lidar_grid_score(
                    head_vm,
                    sender_vm,
                    grid_id,
                    covered_grids=covered_grids | set(selected)),
                sender_vm.perception_manager.lidar.get_grid_density(grid_id),
                str(grid_id)))
        selected.append(best)
        remaining.remove(best)
    return selected


def edgecooper_global_sender_capacity(world, member_budget):
    vehicle_count = max(1, len(world.get_vehicle_managers()))
    cluster_count = max(1, int(getattr(
        world,
        '_edgecooper_global_cluster_count',
        len(world.get_vehicle_managers()))))
    total_slots = max(1, cluster_count * max(1, member_budget))
    return max(1, int(math.ceil(total_slots / float(vehicle_count))))


def select_edgecooper_members(world, head_vm, members, member_budget,
                              global_sender_loads=None,
                              sender_capacity=None):
    selected = []
    covered = set()
    remaining = set(members)
    while remaining and len(selected) < member_budget:
        best = None
        for member_id in sorted(remaining):
            sender_load = 0
            if global_sender_loads is not None:
                sender_load = int(global_sender_loads.get(member_id, 0))
                if sender_capacity is not None and sender_load >= sender_capacity:
                    continue
            sender_vm = world.get_vehicle_manager(member_id)
            candidate_grids = set(edgecooper_candidate_grids(
                head_vm,
                sender_vm))
            if not candidate_grids:
                continue
            complementarity = sum(
                max(0.0, edgecooper_grid_score(
                    head_vm,
                    sender_vm,
                    grid_id,
                    covered_grids=covered))
                for grid_id in candidate_grids)
            redundancy = sum(
                sender_vm.perception_manager.lidar.get_grid_density(grid_id)
                for grid_id in candidate_grids & covered)
            distance = vehicle_distance(head_vm, sender_vm)
            load_penalty = 1.0 + float(sender_load)
            score = (
                complementarity / ((1.0 + distance / 50.0) * load_penalty) -
                0.35 * redundancy)
            item = (-score, distance, member_id, candidate_grids)
            if best is None or item < best:
                best = item
        if best is None:
            break
        _, _, member_id, candidate_grids = best
        selected.append(member_id)
        if global_sender_loads is not None:
            global_sender_loads[member_id] = (
                int(global_sender_loads.get(member_id, 0)) + 1)
        covered.update(candidate_grids)
        remaining.remove(member_id)
    return selected


def select_edgecooper_grids(head_vm, sender_vm, candidates, count,
                            covered_grids=None):
    if count <= 0 or not candidates:
        return []
    covered_grids = set() if covered_grids is None else set(covered_grids)
    remaining = set(candidates)
    selected = []
    while remaining and len(selected) < count:
        best = max(
            remaining,
            key=lambda grid_id: (
                edgecooper_grid_score(
                    head_vm,
                    sender_vm,
                    grid_id,
                    covered_grids=covered_grids | set(selected)),
                sender_vm.perception_manager.lidar.get_grid_density(grid_id),
                str(grid_id)))
        selected.append(best)
        remaining.remove(best)
    return selected


def select_baseline_members(world, cluster, baseline_name, member_budget,
                            link_quality=None, timestamp=None):
    head_id = int(cluster.head_id)
    head_vm = world.get_vehicle_manager(head_id)
    members = candidate_member_ids(world, cluster, baseline_name)
    if member_budget <= 0 or not members:
        return []

    if baseline_name == 'random':
        rng = random.Random('%s_%s_%s' % (timestamp, head_id, member_budget))
        shuffled = list(members)
        rng.shuffle(shuffled)
        return shuffled[:member_budget]

    if baseline_name == 'nearest':
        scored = [
            (vehicle_distance(head_vm, world.get_vehicle_manager(member_id)),
             member_id)
            for member_id in members
        ]
        return [member_id for _, member_id in sorted(scored)[:member_budget]]

    if baseline_name in ['edgecooper', 'edgecooper_global',
                         'edgecooper_global_hd']:
        sender_loads = None
        sender_capacity = None
        if baseline_name in ['edgecooper_global', 'edgecooper_global_hd']:
            sender_loads = getattr(
                world,
                '_edgecooper_global_sender_loads',
                None)
            if sender_loads is None:
                sender_loads = {}
                world._edgecooper_global_sender_loads = sender_loads
            sender_capacity = edgecooper_global_sender_capacity(
                world,
                member_budget)
        return select_edgecooper_members(
            world,
            head_vm,
            members,
            member_budget,
            global_sender_loads=sender_loads,
            sender_capacity=sender_capacity)

    if baseline_name == 'pacp_lidar':
        return select_pacp_lidar_members(
            world,
            head_vm,
            members,
            member_budget,
            link_quality=link_quality,
            timestamp=timestamp)

    if baseline_name in ['density', 'greedy_density', 'communication_aware',
                         'global_selective_proxy',
                         'cluster_local_selective_proxy']:
        scored = []
        for member_id in members:
            sender_vm = world.get_vehicle_manager(member_id)
            density_sum = density_score_for_member(head_vm, sender_vm)
            if baseline_name in ['communication_aware',
                                 'cluster_local_selective_proxy']:
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
            elif baseline_name == 'global_selective_proxy':
                distance = vehicle_distance(head_vm, sender_vm)
                density_sum = density_sum / (1.0 + distance / 200.0)
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
    covered_edge_grids = set()
    for member_id in selected_members:
        if remaining <= 0:
            break
        sender_vm = world.get_vehicle_manager(member_id)
        if baseline_name in ['edgecooper', 'edgecooper_global',
                             'edgecooper_global_hd']:
            candidate_grids = edgecooper_candidate_grids(head_vm, sender_vm)
        elif baseline_name == 'pacp_lidar':
            candidate_grids = pacp_lidar_candidate_grids(head_vm, sender_vm)
        else:
            candidate_grids = candidate_grids_for_sender(head_vm, sender_vm)
        if baseline_name == 'random':
            grids = list(candidate_grids)
            rng = random.Random('%s_%s_%s_%s' % (
                timestamp,
                head_id,
                member_id,
                grid_budget))
            rng.shuffle(grids)
        elif baseline_name in ['edgecooper', 'edgecooper_global',
                               'edgecooper_global_hd']:
            grids = select_edgecooper_grids(
                head_vm,
                sender_vm,
                candidate_grids,
                min(per_member_budget, remaining),
                covered_grids=covered_edge_grids)
        elif baseline_name == 'pacp_lidar':
            grids = select_pacp_lidar_grids(
                head_vm,
                sender_vm,
                candidate_grids,
                min(per_member_budget, remaining),
                covered_grids=covered_edge_grids)
        else:
            grids = sorted(
                candidate_grids,
                key=lambda grid_id: sender_vm.perception_manager.lidar.
                get_grid_density(grid_id),
                reverse=True)
        selected = grids[:min(per_member_budget, remaining)]
        if selected:
            grid_selection[member_id] = selected
            if baseline_name in ['edgecooper', 'edgecooper_global',
                                 'edgecooper_global_hd', 'pacp_lidar']:
                covered_edge_grids.update(selected)
            remaining -= len(selected)
    head_vm.perception_manager.co_manager.set_grid_selection(grid_selection)


def collect_receiver_grid_selection(world, clusters):
    selection = {}
    for cluster in clusters:
        receiver_id = int(cluster.head_id)
        receiver_vm = world.get_vehicle_manager(receiver_id)
        if receiver_vm is None:
            continue
        grid_selection = getattr(
            receiver_vm.perception_manager.co_manager,
            'grid_selection',
            {}) or {}
        normalized = {}
        for sender_id, grid_ids in grid_selection.items():
            grid_ids = list(grid_ids or [])
            if grid_ids:
                normalized[int(sender_id)] = grid_ids
        if normalized:
            selection[receiver_id] = normalized
    return selection


def apply_receiver_grid_selection(world, clusters, selection):
    for cluster in clusters:
        receiver_id = int(cluster.head_id)
        receiver_vm = world.get_vehicle_manager(receiver_id)
        if receiver_vm is None:
            continue
        co_manager = receiver_vm.perception_manager.co_manager
        co_manager.clear_grid_selection()
        co_manager.set_grid_selection(selection.get(receiver_id, {}))


def trim_selective_grid_selection_to_global_deadline(
        world, clusters, baseline_name, deadline_ms, channel_model,
        max_senders_per_receiver=1):
    """Admit selective-sharing grids under one shared per-frame budget.

    The older selective deadline trim was applied independently to each
    receiver. For protocol-native all-CAV baselines this let 20 receivers each
    consume a full deadline budget, while NS3 sees one shared V2V frame. This
    routine makes the budget global and admits high-priority grids in a fair
    round-robin order across scheduled links.
    """
    original = collect_receiver_grid_selection(world, clusters)
    budget_bytes = channel_model.payload_budget_bytes(
        deadline_ms=deadline_ms,
        subchannels=channel_model.num_channels)
    if deadline_ms is None or budget_bytes <= 0:
        return {
            'budget_bytes': budget_bytes,
            'admitted_bytes': 0,
            'candidate_bytes': 0,
            'admitted_links': 0,
            'candidate_links': 0,
        }

    link_entries = []
    candidate_bytes = 0
    for receiver_id, sender_grids in original.items():
        receiver_vm = world.get_vehicle_manager(receiver_id)
        if receiver_vm is None:
            continue
        for sender_id, grid_ids in sender_grids.items():
            sender_vm = world.get_vehicle_manager(sender_id)
            if sender_vm is None:
                continue
            entries = []
            for grid_id in grid_ids:
                points = sender_vm.perception_manager.lidar.\
                    get_local_points_by_grid_ids([grid_id])
                grid_bytes = 0 if points is None else int(points.nbytes)
                if grid_bytes <= 0:
                    continue
                candidate_bytes += grid_bytes
                if baseline_name in ['edgecooper', 'edgecooper_global',
                                     'edgecooper_global_hd']:
                    score = edgecooper_grid_score(
                        receiver_vm,
                        sender_vm,
                        grid_id)
                elif baseline_name == 'pacp_lidar':
                    score = pacp_lidar_grid_score(
                        receiver_vm,
                        sender_vm,
                        grid_id)
                else:
                    score = sender_vm.perception_manager.lidar.\
                        get_grid_density(grid_id)
                entries.append((
                    -float(score),
                    -float(sender_vm.perception_manager.lidar.
                           get_grid_density(grid_id)),
                    str(grid_id),
                    grid_id,
                    grid_bytes))
            entries.sort()
            if entries:
                link_entries.append({
                    'receiver_id': int(receiver_id),
                    'sender_id': int(sender_id),
                    'entries': entries,
                    'cursor': 0,
                })

    # Higher-priority links get earlier turns, but every selected link can
    # advance one grid per pass before a link receives a second grid.
    link_entries.sort(key=lambda item: item['entries'][0][:3])
    original_link_count = len(link_entries)
    if baseline_name in ['edgecooper_global', 'edgecooper_global_hd']:
        matched_links = []
        occupied_senders = set()
        receiver_loads = defaultdict(int)
        max_inbound = max(1, int(max_senders_per_receiver or 1))
        for item in link_entries:
            sender_id = item['sender_id']
            receiver_id = item['receiver_id']
            sender_blocked = (
                sender_id in occupied_senders or
                receiver_loads.get(sender_id, 0) > 0)
            receiver_blocked = (
                receiver_id in occupied_senders or
                receiver_loads.get(receiver_id, 0) >= max_inbound)
            if sender_blocked or receiver_blocked:
                continue
            matched_links.append(item)
            occupied_senders.add(sender_id)
            receiver_loads[receiver_id] += 1
            if len(matched_links) >= channel_model.num_channels:
                break
        link_entries = matched_links

    admitted = {}
    admitted_bytes = 0
    while True:
        advanced = False
        for item in link_entries:
            cursor = item['cursor']
            if cursor >= len(item['entries']):
                continue
            _, _, _, grid_id, grid_bytes = item['entries'][cursor]
            item['cursor'] += 1
            advanced = True
            if admitted_bytes + grid_bytes > budget_bytes:
                continue
            receiver_id = item['receiver_id']
            sender_id = item['sender_id']
            admitted.setdefault(receiver_id, {}).setdefault(
                sender_id, []).append(grid_id)
            admitted_bytes += grid_bytes
            if admitted_bytes >= budget_bytes:
                break
        if not advanced or admitted_bytes >= budget_bytes:
            break

    apply_receiver_grid_selection(world, clusters, admitted)
    admitted_links = sum(
        1 for sender_grids in admitted.values()
        for grid_ids in sender_grids.values() if grid_ids)
    candidate_links = len(link_entries)
    return {
        'budget_bytes': int(budget_bytes),
        'admitted_bytes': int(admitted_bytes),
        'candidate_bytes': int(candidate_bytes),
        'admitted_links': int(admitted_links),
        'candidate_links': int(candidate_links),
        'pre_matching_candidate_links': int(original_link_count),
        'max_senders_per_receiver': max(
            1,
            int(max_senders_per_receiver or 1)),
    }


def apply_selective_sharing_baseline(frame, protocol, ego_cav_id,
                                     baseline_name, receiver_policy,
                                     member_budget, grid_budget,
                                     t_min_stab=None, clustering='coalition_game',
                                     n_max=None, rho_th=None,
                                     link_quality=None, timestamp=None,
                                     num_channels=None,
                                     bandwidth_mhz=None,
                                     max_upload_points_per_source=None,
                                     channel_model=None,
                                     selective_frame_deadline_ms=None,
                                     edgecooper_global_comm_range_m=None,
                                     max_senders_per_receiver=1):
    clear_sgcp_globals()
    world = OfflineCavWorld(
        frame,
        ego_id=ego_cav_id,
        protocol=protocol,
        density_threshold=rho_th)
    if clustering == 'coalition_game':
        clustering_algorithm = CoalitionGame(world)
    elif clustering == 'cov_coalition_game':
        clustering_algorithm = COVCoalitionGame(world)
    elif clustering == 'singleton':
        clustering_algorithm = NaiveCluster(world, all_in_one=False)
    elif clustering == 'all_in_one':
        clustering_algorithm = NaiveCluster(world, all_in_one=True)
    elif is_offline_constructed_clustering(clustering):
        clustering_algorithm = None
    else:
        raise ValueError('Unknown clustering algorithm: %s' % clustering)
    if (clustering_algorithm is not None and t_min_stab is not None and
            hasattr(clustering_algorithm, 'p')):
        clustering_algorithm.p.T_min_stab = t_min_stab
    if (clustering_algorithm is not None and n_max is not None and
            hasattr(clustering_algorithm, 'p')):
        clustering_algorithm.p.N_max = n_max
    if is_offline_constructed_clustering(clustering):
        clusters = build_offline_constructed_clusters(
            world,
            clustering,
            n_max=n_max,
            timestamp=timestamp)
    else:
        clusters = clustering_algorithm.run()
    apply_cluster_state(world, clusters)
    if channel_model is None:
        channel_model = build_channel_model(
            mode='logical',
            bandwidth_mhz=bandwidth_mhz or 20.0,
            num_channels=num_channels or world.network_manager.subchannel_num,
            frame_deadline_s=getattr(world.network_manager, 'time_slot', 0.1))
    if baseline_name in ['edgecooper_global', 'edgecooper_global_hd']:
        world._edgecooper_global_sender_loads = {}
        world._edgecooper_global_cluster_count = len(clusters)
        world._edgecooper_global_receiver_ids = set(
            int(cluster.head_id) for cluster in clusters)
        world._edgecooper_global_comm_range_m = (
            EDGECOOPER_GLOBAL_COMM_RANGE_M
            if edgecooper_global_comm_range_m is None
            else float(edgecooper_global_comm_range_m))
    for cluster in clusters:
        assign_selective_grid_selection(
            world,
            cluster,
            baseline_name,
            member_budget,
            grid_budget,
            link_quality=link_quality,
            timestamp=timestamp)
    global_admission = None
    if selective_frame_deadline_ms is not None:
        global_admission = trim_selective_grid_selection_to_global_deadline(
            world,
            clusters,
            baseline_name,
            selective_frame_deadline_ms,
            channel_model,
            max_senders_per_receiver=max_senders_per_receiver)

    if receiver_policy == 'all-cluster-heads':
        receiver_ids = sorted(int(cluster.head_id) for cluster in clusters)
    elif receiver_policy == 'all-scheduled-receivers':
        receiver_ids = scheduled_receiver_ids(world, clusters)
    elif receiver_policy == 'all-cavs':
        receiver_ids = sorted(int(cav_id)
                              for cav_id in world.get_vehicle_managers())
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
            receiver_id,
            max_upload_points_per_source=max_upload_points_per_source)
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
        metadata['selective_frame_deadline_ms'] = (
            '' if selective_frame_deadline_ms is None
            else selective_frame_deadline_ms)
        metadata['edgecooper_global_comm_range_m'] = (
            getattr(world, '_edgecooper_global_comm_range_m', '')
            if baseline_name in ['edgecooper_global',
                                 'edgecooper_global_hd'] else '')
        metadata['max_senders_per_receiver'] = max(
            1,
            int(max_senders_per_receiver or 1))
        if global_admission is not None:
            metadata['selective_global_budget_bytes'] = (
                global_admission['budget_bytes'])
            metadata['selective_global_admitted_bytes'] = (
                global_admission['admitted_bytes'])
            metadata['selective_global_candidate_bytes'] = (
                global_admission['candidate_bytes'])
            metadata['selective_global_admitted_links'] = (
                global_admission['admitted_links'])
            metadata['selective_global_candidate_links'] = (
                global_admission['candidate_links'])
        metadata['num_channels'] = num_channels or world.network_manager.subchannel_num
        metadata['bandwidth_mhz'] = bandwidth_mhz or 20.0
        metadata.update(channel_model.to_metadata())
        metadata['ns3_link_quality_csv'] = (
            link_quality['path'] if link_quality else '')
        metadata['max_upload_points_per_source'] = (
            max_upload_points_per_source or '')
        metadata['frame_comm_time_ms'] = estimate_eval_frame_comm_time_ms(
            constrained_frame,
            metadata,
            metadata['bandwidth_mhz'],
            metadata['num_channels'],
            channel_model=channel_model)
        constrained_items.append((constrained_frame, metadata))
    return constrained_items


def summarize_opencood_output(output_dict, dataset, batch_data=None,
                              prefix='opencood_debug'):
    if not output_dict:
        print('%s no_output' % prefix)
        return
    score_threshold = None
    try:
        score_threshold = float(
            dataset.post_processor.params['target_args']['score_threshold'])
    except (AttributeError, KeyError, TypeError, ValueError):
        pass

    for cav_id, preds in output_dict.items():
        if not isinstance(preds, dict) or 'psm' not in preds:
            print('%s cav=%s keys=%s' %
                  (prefix, cav_id, sorted(preds.keys())
                   if isinstance(preds, dict) else type(preds).__name__))
            continue
        prob = torch.sigmoid(preds['psm'].detach()).reshape(-1)
        quantiles = torch.quantile(
            prob,
            torch.tensor([0.5, 0.9, 0.99, 0.999],
                         device=prob.device)).detach().cpu().numpy()
        above_counts = {}
        for threshold in [0.001, 0.003, 0.005, 0.01, 0.05, 0.10,
                          0.20, 0.30, 0.50]:
            above_counts[threshold] = int((prob > threshold).sum().item())
        shape_text = ','.join(str(item) for item in preds['psm'].shape)
        rm_shape_text = (
            ','.join(str(item) for item in preds['rm'].shape)
            if 'rm' in preds else 'none')
        print(
            '%s cav=%s psm_shape=%s rm_shape=%s score_threshold=%s '
            'prob_min=%.6f prob_max=%.6f prob_mean=%.6f '
            'q50=%.6f q90=%.6f q99=%.6f q999=%.6f '
            'above_0001=%d above_0003=%d above_0005=%d '
            'above_001=%d above_005=%d above_010=%d '
            'above_020=%d above_030=%d above_050=%d' %
            (prefix, cav_id, shape_text, rm_shape_text,
             '' if score_threshold is None else '%.4f' % score_threshold,
             float(prob.min().item()), float(prob.max().item()),
             float(prob.mean().item()),
             float(quantiles[0]), float(quantiles[1]),
             float(quantiles[2]), float(quantiles[3]),
             above_counts[0.001], above_counts[0.003],
             above_counts[0.005], above_counts[0.01],
             above_counts[0.05], above_counts[0.10],
             above_counts[0.20], above_counts[0.30],
             above_counts[0.50]))
        if batch_data is None or score_threshold is None:
            continue
        try:
            cav_content = batch_data[cav_id]
            anchor_box = cav_content['anchor_box']
            transformation_matrix = cav_content['transformation_matrix']
            reg = preds['rm']
            batch_box3d = dataset.post_processor.delta_to_boxes3d(
                reg,
                anchor_box)
            mask = (prob > score_threshold).view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)
            boxes3d = torch.masked_select(
                batch_box3d[0],
                mask_reg[0]).view(-1, 7)
            scores = torch.masked_select(prob, mask.view(-1))
            if len(boxes3d) == 0:
                print('%s cav=%s post_candidates=0' % (prefix, cav_id))
                continue
            from opencood.utils import box_utils
            boxes3d_corner = box_utils.boxes_to_corners_3d(
                boxes3d,
                order=dataset.post_processor.params['order'])
            projected_boxes3d = box_utils.project_box3d(
                boxes3d_corner,
                transformation_matrix)
            keep_large = box_utils.remove_large_pred_bbx(projected_boxes3d)
            keep_z = box_utils.remove_bbx_abnormal_z(projected_boxes3d)
            keep_range = box_utils.get_mask_for_boxes_within_range_torch(
                projected_boxes3d)
            keep_all = torch.logical_and(
                torch.logical_and(keep_large, keep_z),
                keep_range)
            filtered_boxes = projected_boxes3d[
                torch.logical_and(keep_large, keep_z)]
            filtered_scores = scores[
                torch.logical_and(keep_large, keep_z)]
            nms_keep_count = 0
            nms_range_count = 0
            if filtered_boxes.shape[0] > 0:
                nms_keep = box_utils.nms_rotated(
                    filtered_boxes,
                    filtered_scores,
                    dataset.post_processor.params['nms_thresh'])
                nms_keep_count = int(len(nms_keep))
                if nms_keep_count > 0:
                    nms_boxes = filtered_boxes[nms_keep]
                    nms_range_count = int(
                        box_utils.get_mask_for_boxes_within_range_torch(
                            nms_boxes).sum().item())
            print('%s cav=%s post_candidates=%d keep_large=%d keep_z=%d '
                  'keep_range=%d keep_all_pre_nms=%d nms_keep=%d '
                  'nms_keep_in_range=%d score_max=%.6f' %
                  (prefix, cav_id, int(boxes3d.shape[0]),
                   int(keep_large.sum().item()),
                   int(keep_z.sum().item()),
                   int(keep_range.sum().item()),
                   int(keep_all.sum().item()),
                   nms_keep_count,
                   nms_range_count,
                   float(scores.max().item())))
        except Exception as error:
            print('%s cav=%s post_debug_error=%s' %
                  (prefix, cav_id, str(error)))


def run_opencood_inference(manager, frame, ego_lidar_pose,
                           debug_output=False):
    reformat_data_dict = manager.opencood_dataset.get_item_test(
        frame,
        ego_lidar_pose)
    output_dict = manager.opencood_dataset.collate_batch_test(
        [reformat_data_dict])
    batch_data = manager.to_device(output_dict)
    if debug_output:
        from opencood.tools import inference_utils
        with torch.no_grad():
            if manager.fusion_method == 'late':
                ret = inference_utils.inference_late_fusion(
                    batch_data,
                    manager.model,
                    manager.opencood_dataset,
                    return_output=True,
                    return_object_ids=False)
            elif manager.fusion_method == 'early':
                ret = inference_utils.inference_early_fusion(
                    batch_data,
                    manager.model,
                    manager.opencood_dataset,
                    return_output=True,
                    return_object_ids=False)
            elif manager.fusion_method.startswith('intermediate'):
                ret = inference_utils.inference_intermediate_fusion(
                    batch_data,
                    manager.model,
                    manager.opencood_dataset,
                    return_output=True,
                    return_object_ids=False)
            else:
                raise NotImplementedError(
                    'Only early, late and intermediate fusion is supported.')
        summarize_opencood_output(
            ret[-1],
            manager.opencood_dataset,
            batch_data=batch_data,
            prefix='opencood_debug')
        return ret[:-1]
    return manager.inference(
        batch_data,
        with_stats=False,
        return_object_ids=manager.fusion_method != 'late')


def project_points_by_matrix(points, transformation_matrix):
    if points is None or points.size == 0:
        return points
    xyz1 = np.hstack((
        points[:, :3],
        np.ones((points.shape[0], 1), dtype=points.dtype)))
    projected = np.dot(xyz1, np.asarray(transformation_matrix).T)[:, :3]
    output = points.copy()
    output[:, :3] = projected
    return output


def collapse_frame_to_receiver_pointcloud(frame, receiver_lidar_pose):
    """Collapse an evaluated multi-CAV frame into one merged ego CAV."""
    ego_id = next(cav_id for cav_id, cav in frame.items() if cav['ego'])
    ego_cav = frame[ego_id]
    collapsed = OrderedDict()
    merged_points = []
    merged_vehicles = {}
    for cav in frame.values():
        cav_pose = cav['params']['lidar_pose']
        transformation_matrix = x1_to_x2(
            cav_pose,
            receiver_lidar_pose)
        merged_points.append(
            project_points_by_matrix(cav['lidar_np'], transformation_matrix))
        merged_vehicles.update(cav['params'].get('vehicles', {}))

    cloned = OrderedDict()
    cloned['ego'] = True
    cloned['time_delay'] = ego_cav.get('time_delay', 0)
    cloned['params'] = copy.deepcopy(ego_cav['params'])
    cloned['params']['vehicles'] = merged_vehicles
    cloned['params']['transformation_matrix'] = x1_to_x2(
        receiver_lidar_pose,
        receiver_lidar_pose)
    cloned['params']['gt_transformation_matrix'] = (
        cloned['params']['transformation_matrix'])
    cloned['params']['spatial_correction_matrix'] = (
        cloned['params']['transformation_matrix'])
    cloned['lidar_np'] = np.vstack(merged_points).astype(np.float32)
    collapsed[ego_id] = cloned
    return collapsed


def is_empty_pillar_error(error):
    return (
        isinstance(error, RuntimeError) and
        'input.numel() == 0' in str(error))


def format_channel_allocation(channel_allocation):
    items = []
    for link, subchannel in sorted(channel_allocation.items()):
        try:
            source_id, target_id = link
            items.append('%s>%s:%s' % (source_id, target_id, subchannel))
        except (TypeError, ValueError):
            items.append('%s:%s' % (link, subchannel))
    return ';'.join(items)


def trace_row(scenario_id, timestamp, metadata, eval_frame,
              pred_count=None, gt_count=None, skipped=''):
    source_ids = [int(cav_id) for cav_id in metadata.get('source_cav_ids', [])]
    receiver_id = int(metadata.get('receiver_id'))
    selected_grid_counts = {
        int(key): int(value)
        for key, value in metadata.get('selected_grid_counts', {}).items()
    }
    channel_allocation = metadata.get('channel_allocation', {}) or {}
    scheduled_sources = set()
    for link in channel_allocation.keys():
        try:
            source_id, target_id = link
        except (TypeError, ValueError):
            continue
        if int(target_id) == receiver_id:
            scheduled_sources.add(int(source_id))
    uploaded_sources = [source_id for source_id in source_ids
                        if source_id != receiver_id]
    missing_channel_sources = [
        source_id for source_id in uploaded_sources
        if source_id not in scheduled_sources
    ]
    point_counts = {
        int(cav_id): int(cav['lidar_np'].shape[0])
        for cav_id, cav in eval_frame.items()
    }
    return {
        'scenario_id': scenario_id,
        'timestamp': timestamp,
        'receiver_id': receiver_id,
        'receiver_policy': metadata.get('receiver_policy', ''),
        'coperception_yaml': metadata.get('coperception_yaml', ''),
        'resource_allocation': metadata.get('resource_allocation', ''),
        'upload_mode': metadata.get('upload_mode', ''),
        'grid_selection_mode': metadata.get('grid_selection_mode', ''),
        'grid_score_mode': metadata.get('grid_score_mode', ''),
        'coverage_fallback': metadata.get('coverage_fallback', ''),
        'coverage_fallback_replacements': metadata.get(
            'coverage_fallback_replacements', ''),
        'routing_hint_replacements': metadata.get(
            'routing_hint_replacements', ''),
        'routing_hints_csv': metadata.get('routing_hints_csv', ''),
        'clustering': metadata.get('clustering', ''),
        'cluster_count': metadata.get('cluster_count', ''),
        'selective_frame_deadline_ms': metadata.get(
            'selective_frame_deadline_ms',
            ''),
        'cluster_member_ids': ';'.join(
            str(item) for item in metadata.get('cluster_member_ids', [])),
        'source_cav_ids': ';'.join(str(item) for item in source_ids),
        'uploaded_source_ids': ';'.join(str(item) for item in uploaded_sources),
        'selected_grid_counts_json': json.dumps(
            selected_grid_counts, sort_keys=True),
        'point_counts_json': json.dumps(point_counts, sort_keys=True),
        'communication_bytes': metadata.get('communication_bytes', 0),
        'frame_comm_time_ms': metadata.get('frame_comm_time_ms', ''),
        'num_channels': metadata.get('num_channels', ''),
        'bandwidth_mhz': metadata.get('bandwidth_mhz', ''),
        'communication_deadline_ms': metadata.get(
            'communication_deadline_ms', ''),
        'sgcp_frame_mbps_budget': metadata.get(
            'sgcp_frame_mbps_budget', ''),
        'sgcp_frame_budget_bytes': metadata.get(
            'sgcp_frame_budget_bytes', ''),
        'sgcp_frame_budget_original_bytes': metadata.get(
            'sgcp_frame_budget_original_bytes', ''),
        'sgcp_frame_budget_admitted_bytes': metadata.get(
            'sgcp_frame_budget_admitted_bytes', ''),
        'max_senders_per_receiver': metadata.get(
            'max_senders_per_receiver', ''),
        'channel_estimator': metadata.get('channel_estimator', ''),
        'ns3_tb_size_bytes': metadata.get('ns3_tb_size_bytes', ''),
        'ns3_slot_duration_ms': metadata.get('ns3_slot_duration_ms', ''),
        'ns3_subchannel_prbs': metadata.get('ns3_subchannel_prbs', ''),
        'ns3_symbols_per_slot': metadata.get('ns3_symbols_per_slot', ''),
        'ns3_mcs': metadata.get('ns3_mcs', ''),
        'pcs_rounds_requested': metadata.get('pcs_rounds_requested', ''),
        'pcs_rounds_accepted': metadata.get('pcs_rounds_accepted', ''),
        'pcs_round_comm_time_ms_json': json.dumps(
            metadata.get('pcs_round_comm_time_ms', [])),
        'pcs_round_comm_bytes_json': json.dumps(
            metadata.get('pcs_round_comm_bytes', [])),
        'channel_allocation': format_channel_allocation(channel_allocation),
        'missing_channel_sources': ';'.join(
            str(item) for item in missing_channel_sources),
        'pred_boxes': '' if pred_count is None else pred_count,
        'gt_boxes': '' if gt_count is None else gt_count,
        'skipped': skipped,
    }


def write_trace_csv(path, rows):
    if not path or not rows:
        return
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fieldnames = [
        'scenario_id',
        'timestamp',
        'receiver_id',
        'receiver_policy',
        'coperception_yaml',
        'resource_allocation',
        'upload_mode',
        'grid_selection_mode',
        'grid_score_mode',
        'coverage_fallback',
        'coverage_fallback_replacements',
        'routing_hint_replacements',
        'routing_hints_csv',
        'clustering',
        'cluster_count',
        'selective_frame_deadline_ms',
        'cluster_member_ids',
        'source_cav_ids',
        'uploaded_source_ids',
        'selected_grid_counts_json',
        'point_counts_json',
        'communication_bytes',
        'frame_comm_time_ms',
        'num_channels',
        'bandwidth_mhz',
        'communication_deadline_ms',
        'sgcp_frame_mbps_budget',
        'sgcp_frame_budget_bytes',
        'sgcp_frame_budget_original_bytes',
        'sgcp_frame_budget_admitted_bytes',
        'max_senders_per_receiver',
        'channel_estimator',
        'ns3_tb_size_bytes',
        'ns3_slot_duration_ms',
        'ns3_subchannel_prbs',
        'ns3_symbols_per_slot',
        'ns3_mcs',
        'pcs_rounds_requested',
        'pcs_rounds_accepted',
        'pcs_round_comm_time_ms_json',
        'pcs_round_comm_bytes_json',
        'channel_allocation',
        'missing_channel_sources',
        'pred_boxes',
        'gt_boxes',
        'skipped',
    ]
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def tensor_to_numpy(value):
    if value is None:
        return None
    return common_utils.torch_tensor_to_numpy(value)


def normalize_object_ids(object_ids, count):
    if object_ids is None:
        return [''] * count
    normalized = [str(item) for item in object_ids]
    if len(normalized) < count:
        normalized.extend([''] * (count - len(normalized)))
    return normalized[:count]


def box_center(box):
    if box is None or len(box) == 0:
        return ('', '', '')
    center = np.mean(box[:, :3], axis=0)
    return ('%.4f' % center[0], '%.4f' % center[1], '%.4f' % center[2])


def match_predictions_to_gt(gt_boxes, pred_boxes, pred_scores, iou_thresh):
    """Return best and greedy matched prediction info for each canonical GT."""
    if gt_boxes is None:
        return []
    gt_boxes_np = tensor_to_numpy(gt_boxes)
    gt_count = gt_boxes_np.shape[0]
    if gt_count == 0:
        return []
    matches = [
        {
            'matched': False,
            'best_iou': 0.0,
            'best_score': '',
            'matched_score': '',
        }
        for _ in range(gt_count)
    ]
    if pred_boxes is None or pred_scores is None:
        return matches

    pred_boxes_np = tensor_to_numpy(pred_boxes)
    pred_scores_np = tensor_to_numpy(pred_scores)
    if pred_boxes_np is None or pred_scores_np is None:
        return matches
    if pred_boxes_np.shape[0] == 0:
        return matches

    pred_polygons = list(common_utils.convert_format(pred_boxes_np))
    gt_polygons = list(common_utils.convert_format(gt_boxes_np))
    score_order = np.argsort(-pred_scores_np)
    unmatched_gt = set(range(gt_count))

    for pred_index in score_order:
        if not unmatched_gt:
            break
        pred_polygon = pred_polygons[pred_index]
        ious = common_utils.compute_iou(pred_polygon, gt_polygons)
        if len(ious) == 0:
            continue
        for gt_index, iou in enumerate(ious):
            if iou > matches[gt_index]['best_iou']:
                matches[gt_index]['best_iou'] = float(iou)
                matches[gt_index]['best_score'] = (
                    '%.6f' % float(pred_scores_np[pred_index]))
        candidate_gt = max(unmatched_gt, key=lambda idx: ious[idx])
        candidate_iou = float(ious[candidate_gt])
        if candidate_iou < iou_thresh:
            continue
        matches[candidate_gt]['matched'] = True
        matches[candidate_gt]['matched_score'] = (
            '%.6f' % float(pred_scores_np[pred_index]))
        unmatched_gt.remove(candidate_gt)
    return matches


def object_diagnostic_rows(scenario_id, timestamp, sample_label, metadata,
                           canonical_gt_boxes, canonical_gt_ids,
                           reference_pred_boxes, reference_scores,
                           method_pred_boxes, method_scores, iou_thresh):
    if canonical_gt_boxes is None:
        return []
    gt_boxes_np = tensor_to_numpy(canonical_gt_boxes)
    if gt_boxes_np is None:
        return []
    gt_ids = normalize_object_ids(canonical_gt_ids, gt_boxes_np.shape[0])
    reference_matches = match_predictions_to_gt(
        canonical_gt_boxes,
        reference_pred_boxes,
        reference_scores,
        iou_thresh)
    method_matches = match_predictions_to_gt(
        canonical_gt_boxes,
        method_pred_boxes,
        method_scores,
        iou_thresh)
    rows = []
    metadata = metadata or {}
    for gt_index, gt_box in enumerate(gt_boxes_np):
        center_x, center_y, center_z = box_center(gt_box)
        ref_match = reference_matches[gt_index]
        method_match = method_matches[gt_index]
        rows.append({
            'scenario_id': scenario_id,
            'timestamp': timestamp,
            'sample_label': sample_label,
            'receiver_id': metadata.get('receiver_id', ''),
            'resource_allocation': metadata.get('resource_allocation', ''),
            'clustering': metadata.get('clustering', ''),
            'upload_mode': metadata.get('upload_mode', ''),
            'grid_selection_mode': metadata.get('grid_selection_mode', ''),
            'grid_score_mode': metadata.get('grid_score_mode', ''),
            'num_channels': metadata.get('num_channels', ''),
            'bandwidth_mhz': metadata.get('bandwidth_mhz', ''),
            'communication_deadline_ms': metadata.get(
                'communication_deadline_ms', ''),
            'channel_estimator': metadata.get('channel_estimator', ''),
            'ns3_tb_size_bytes': metadata.get('ns3_tb_size_bytes', ''),
            'ns3_slot_duration_ms': metadata.get('ns3_slot_duration_ms', ''),
            'communication_bytes': metadata.get('communication_bytes', ''),
            'source_cav_ids': ';'.join(
                str(item) for item in metadata.get('source_cav_ids', [])),
            'uploaded_source_ids': ';'.join(
                str(item) for item in metadata.get(
                    'source_cav_ids', [])[1:]),
            'selected_grid_counts_json': json.dumps(
                metadata.get('selected_grid_counts', {}),
                sort_keys=True),
            'gt_index': gt_index,
            'gt_object_id': gt_ids[gt_index],
            'gt_center_x': center_x,
            'gt_center_y': center_y,
            'gt_center_z': center_z,
            'full_reference_matched': int(ref_match['matched']),
            'full_reference_best_iou': '%.6f' % ref_match['best_iou'],
            'full_reference_best_score': ref_match['best_score'],
            'method_matched': int(method_match['matched']),
            'method_best_iou': '%.6f' % method_match['best_iou'],
            'method_best_score': method_match['best_score'],
            'full_detected_method_missed': int(
                ref_match['matched'] and not method_match['matched']),
        })
    return rows


def write_object_diagnostics_csv(path, rows):
    if not path or not rows:
        return
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fieldnames = [
        'scenario_id',
        'timestamp',
        'sample_label',
        'receiver_id',
        'resource_allocation',
        'clustering',
        'upload_mode',
        'grid_selection_mode',
        'grid_score_mode',
        'num_channels',
        'bandwidth_mhz',
        'communication_deadline_ms',
        'channel_estimator',
        'ns3_tb_size_bytes',
        'ns3_slot_duration_ms',
        'communication_bytes',
        'source_cav_ids',
        'uploaded_source_ids',
        'selected_grid_counts_json',
        'gt_index',
        'gt_object_id',
        'gt_center_x',
        'gt_center_y',
        'gt_center_z',
        'full_reference_matched',
        'full_reference_best_iou',
        'full_reference_best_score',
        'method_matched',
        'method_best_iou',
        'method_best_score',
        'full_detected_method_missed',
    ]
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def result_stat_lengths(result_stat):
    return {
        iou: {
            'tp': len(values['tp']),
            'fp': len(values['fp']),
            'gt': values['gt'],
        }
        for iou, values in result_stat.items()
    }


def result_stat_delta(result_stat, before):
    delta = {}
    for iou, values in result_stat.items():
        start_tp = before[iou]['tp']
        start_fp = before[iou]['fp']
        delta[iou] = {
            'tp': list(values['tp'][start_tp:]),
            'fp': list(values['fp'][start_fp:]),
            'gt': values['gt'] - before[iou]['gt'],
        }
    return delta


def append_eval_stats_row(rows,
                          scenario_id,
                          timestamp,
                          sample_label,
                          receiver_id,
                          metadata,
                          before,
                          result_stat):
    if rows is None:
        return
    delta = result_stat_delta(result_stat, before)
    row = {
        'scenario_id': scenario_id,
        'timestamp': timestamp,
        'sample_label': sample_label,
        'receiver_id': receiver_id,
        'resource_allocation': (
            '' if metadata is None else
            metadata.get('resource_allocation', '')),
        'clustering': '' if metadata is None else metadata.get('clustering', ''),
        'communication_bytes': (
            0 if metadata is None else
            metadata.get('communication_bytes', 0)),
    }
    for iou in (0.3, 0.5, 0.7):
        suffix = str(iou).replace('.', '')
        row['tp_%s_json' % suffix] = json.dumps(delta[iou]['tp'])
        row['fp_%s_json' % suffix] = json.dumps(delta[iou]['fp'])
        row['gt_%s' % suffix] = delta[iou]['gt']
    rows.append(row)


def write_eval_stats_csv(path, rows):
    if not path or not rows:
        return
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fieldnames = [
        'scenario_id',
        'timestamp',
        'sample_label',
        'receiver_id',
        'resource_allocation',
        'clustering',
        'communication_bytes',
        'tp_03_json',
        'fp_03_json',
        'gt_03',
        'tp_05_json',
        'fp_05_json',
        'gt_05',
        'tp_07_json',
        'fp_07_json',
        'gt_07',
    ]
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    dataset = OPV2VFrameDataset(args.dataset_root)

    coperception_params = load_coperception_params(
        args.coperception_yaml,
        args.fusion_method)
    manager = OpenCOODManager(coperception_params)
    if args.postprocess_score_threshold is not None:
        manager.opencood_dataset.post_processor.params[
            'target_args']['score_threshold'] = (
                args.postprocess_score_threshold)
        print('postprocess_score_threshold_override=%.6f' %
              args.postprocess_score_threshold)
    if args.postprocess_nms_thresh is not None:
        manager.opencood_dataset.post_processor.params['nms_thresh'] = (
            args.postprocess_nms_thresh)
        print('postprocess_nms_thresh_override=%.6f' %
              args.postprocess_nms_thresh)
    sgcp_summaries = []
    sgcp_trace_rows = []
    object_rows = []
    eval_stats_rows = [] if args.eval_stats_output else None
    ns3_link_quality = load_ns3_link_quality(args.ns3_link_quality_csv)
    sgcp_routing_hints = load_sgcp_routing_hints(
        args.sgcp_routing_hints_csv)
    fixed_cluster_templates = []
    sgcp_coverage_state = defaultdict(dict)

    if args.timestamp is not None:
        if args.scenario_id is None:
            scenario_id = next(iter(dataset.scenarios.keys()))
        else:
            scenario_id = args.scenario_id
        frames = [(scenario_id, args.timestamp)]
        protocol_for_timing = load_protocol(dataset, scenario_id)
        effective_t_min_stab = (
            args.t_min_stab if args.t_min_stab is not None
            else fixed_delta_from_protocol(protocol_for_timing))
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
        protocol_for_timing = load_protocol(dataset, scenario_id)
        interval_source = timestamps[args.start_index:args.start_index + 2]
        effective_t_min_stab = (
            args.t_min_stab if args.t_min_stab is not None
            else frame_interval_seconds(
                interval_source,
                fixed_delta_from_protocol(protocol_for_timing)))

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
        original_ego = next(cav for cav in frame.values() if cav['ego'])
        target_ego_lidar_pose = original_ego['params']['lidar_pose']
        canonical_ret = None
        if args.object_diagnostics_output:
            canonical_ret = run_opencood_inference(
                manager,
                frame,
                target_ego_lidar_pose,
                debug_output=args.debug_opencood_output)
        frame_items = [(frame, None)]
        late_receiver_policy = (
            args.sgcp_receiver_policy
            if args.sgcp_receiver_policy in ['all-scheduled-receivers',
                                             'all-cavs']
            else 'all-cluster-heads')
        if args.selective_sharing_baseline is not None:
            protocol = load_protocol(dataset, scenario_id)
            frame_channel_model = build_cli_channel_model(args)
            frame_items = apply_selective_sharing_baseline(
                frame,
                protocol,
                args.ego_cav_id,
                args.selective_sharing_baseline,
                late_receiver_policy if args.sgcp_inter_cluster_late_fusion
                else args.sgcp_receiver_policy,
                args.selective_member_budget,
                args.selective_grid_budget,
                effective_t_min_stab,
                args.clustering,
                args.n_max,
                args.rho_th,
                link_quality=ns3_link_quality,
                timestamp=timestamp,
                num_channels=args.num_channels,
                bandwidth_mhz=args.bandwidth_mhz,
                max_upload_points_per_source=(
                    args.max_upload_points_per_source),
                channel_model=frame_channel_model,
                selective_frame_deadline_ms=(
                    args.selective_frame_deadline_ms),
                edgecooper_global_comm_range_m=(
                    args.edgecooper_global_comm_range_m),
                max_senders_per_receiver=(
                    args.max_senders_per_receiver))
        elif args.sgcp_constrained:
            protocol = load_protocol(dataset, scenario_id)
            frame_channel_model = build_cli_channel_model(args)
            frame_items = apply_sgcp_constraint(
                frame,
                protocol,
                args.ego_cav_id,
                args.resource_allocation,
                late_receiver_policy if args.sgcp_inter_cluster_late_fusion
                else args.sgcp_receiver_policy,
                effective_t_min_stab,
                args.clustering,
                args.n_max,
                args.rho_th,
                args.num_channels,
                args.bandwidth_mhz,
                args.sgcp_upload_mode,
                args.sgcp_grid_selection_mode,
                args.sgcp_grid_score_mode,
                timestamp,
                fixed_cluster_templates=(
                    fixed_cluster_templates
                    if args.clustering == 'fixed_first_frame' else None),
                head_rb_budget=args.head_rb_budget,
                pcs_blind_spot_min_division=(
                    args.pcs_blind_spot_min_division),
                pcs_min_overlap_grids=args.pcs_min_overlap_grids,
                pcs_blind_spot_radius=args.pcs_blind_spot_radius,
                pcs_min_spot_grids=args.pcs_min_spot_grids,
                pcs_communication_range_m=args.pcs_communication_range_m,
                pcs_frame_rounds=args.pcs_frame_rounds,
                pcs_frame_deadline_ms=args.pcs_frame_deadline_ms,
                coverage_fallback=args.sgcp_coverage_fallback,
                coverage_state=sgcp_coverage_state,
                max_upload_points_per_source=(
                    args.max_upload_points_per_source),
                routing_hints=sgcp_routing_hints,
                routing_hints_max_per_frame=(
                    args.sgcp_routing_hints_max_per_frame),
                channel_model=frame_channel_model,
                max_senders_per_receiver=(
                    args.max_senders_per_receiver),
                sgcp_frame_mbps_budget=args.sgcp_frame_mbps_budget)
        for _, metadata in frame_items:
            if metadata is not None:
                metadata['coperception_yaml'] = args.coperception_yaml
        if args.sgcp_inter_cluster_late_fusion:
            pred_tensors = []
            pred_scores = []
            gt_tensors = []
            for receiver_index, (eval_frame, sgcp_metadata) in enumerate(
                    frame_items,
                    start=1):
                try:
                    inference_frame = (
                        collapse_frame_to_receiver_pointcloud(
                            eval_frame,
                            target_ego_lidar_pose)
                        if args.collapse_to_ego_pointcloud else eval_frame)
                    ret = run_opencood_inference(
                        manager,
                        inference_frame,
                        target_ego_lidar_pose,
                        debug_output=args.debug_opencood_output)
                except RuntimeError as error:
                    if not is_empty_pillar_error(error):
                        raise
                    if sgcp_metadata is not None:
                        sgcp_summaries.append(sgcp_metadata)
                        sgcp_trace_rows.append(trace_row(
                            scenario_id,
                            timestamp,
                            sgcp_metadata,
                            eval_frame,
                            skipped='empty_pillars'))
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
                    update_coverage_quality_state(
                        sgcp_coverage_state,
                        sgcp_metadata,
                        0 if pred_box_tensor is None else
                        pred_box_tensor.shape[0],
                        0 if gt_box_tensor is None else
                        gt_box_tensor.shape[0])
                    sgcp_summaries.append(sgcp_metadata)
                    sgcp_trace_rows.append(trace_row(
                        scenario_id,
                        timestamp,
                        sgcp_metadata,
                        eval_frame,
                        pred_count=(
                            0 if pred_box_tensor is None else
                            pred_box_tensor.shape[0]),
                        gt_count=(
                            0 if gt_box_tensor is None else
                            gt_box_tensor.shape[0])))
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

            fused_pred_ret = manager.naive_late_fusion(
                pred_tensors,
                pred_scores,
                iou_threshold=args.sgcp_late_nms_thresh)
            fused_pred, fused_score = fused_pred_ret[:2]
            fused_gt, _ = manager.naive_late_fusion(
                gt_tensors,
                None,
                iou_threshold=args.sgcp_late_nms_thresh)
            print('sgcp_late_fusion frame=%s/%s scenario=%s timestamp=%s '
                  'sources=%s fused_pred_boxes=%s fused_gt_boxes=%s' % (
                      index,
                      len(frames),
                      scenario_id,
                      timestamp,
                      len(frame_items),
                      0 if fused_pred is None else fused_pred.shape[0],
                      0 if fused_gt is None else fused_gt.shape[0]))
            aggregate_metadata = {
                'receiver_id': args.ego_cav_id,
                'coperception_yaml': args.coperception_yaml,
                'resource_allocation': (
                    'selective_%s' % args.selective_sharing_baseline
                    if args.selective_sharing_baseline is not None
                    else args.resource_allocation),
                'clustering': args.clustering,
                'upload_mode': args.sgcp_upload_mode,
                'grid_selection_mode': args.sgcp_grid_selection_mode,
                'grid_score_mode': args.sgcp_grid_score_mode,
                'num_channels': args.num_channels,
                'bandwidth_mhz': args.bandwidth_mhz,
                'communication_deadline_ms': (
                    '' if args.communication_deadline_ms is None
                    else args.communication_deadline_ms),
                'selective_frame_deadline_ms': (
                    '' if args.selective_frame_deadline_ms is None
                    else args.selective_frame_deadline_ms),
                'channel_estimator': args.channel_estimator,
                'ns3_tb_size_bytes': args.ns3_tb_size_bytes,
                'ns3_slot_duration_ms': args.ns3_slot_duration_ms,
                'ns3_subchannel_prbs': args.ns3_subchannel_prbs,
                'ns3_symbols_per_slot': args.ns3_symbols_per_slot,
                'ns3_mcs': args.ns3_mcs,
                'communication_bytes': sum(
                    int((metadata or {}).get('communication_bytes', 0))
                    for _, metadata in frame_items),
                'frame_comm_time_ms': max(
                    [
                        float((metadata or {}).get(
                            'frame_comm_time_ms', 0) or 0)
                        for _, metadata in frame_items
                    ] or [0.0]),
                'pcs_rounds_requested': max(
                    [
                        int((metadata or {}).get(
                            'pcs_rounds_requested', 0) or 0)
                        for _, metadata in frame_items
                    ] or [0]),
                'pcs_rounds_accepted': max(
                    [
                        int((metadata or {}).get(
                            'pcs_rounds_accepted', 0) or 0)
                        for _, metadata in frame_items
                    ] or [0]),
                'pcs_round_comm_time_ms': [],
                'pcs_round_comm_bytes': [],
                'source_cav_ids': sorted(set(
                    source_id
                    for _, metadata in frame_items
                    for source_id in (
                        (metadata or {}).get('source_cav_ids', [])))),
                'selected_grid_counts': {},
            }
            for _, metadata in frame_items:
                if not metadata:
                    continue
                for key, value in metadata.get(
                        'selected_grid_counts', {}).items():
                    aggregate_metadata['selected_grid_counts'][key] = (
                        aggregate_metadata[
                            'selected_grid_counts'].get(key, 0) +
                        value)
            if canonical_ret is not None:
                object_rows.extend(object_diagnostic_rows(
                    scenario_id,
                    timestamp,
                    'inter_cluster_late_fusion',
                    aggregate_metadata,
                    canonical_ret[2],
                    canonical_ret[3] if len(canonical_ret) > 3 else None,
                    canonical_ret[0],
                    canonical_ret[1],
                    fused_pred,
                    fused_score,
                    args.object_diagnostics_iou))
            before_stats = result_stat_lengths(manager.result_stat)
            manager.submit_results(
                fused_pred,
                fused_score,
                fused_gt,
                with_stats=True,
                force=True)
            append_eval_stats_row(
                eval_stats_rows,
                scenario_id,
                timestamp,
                'inter_cluster_late_fusion',
                'late_fused',
                aggregate_metadata,
                before_stats,
                manager.result_stat)
            continue
        for receiver_index, (eval_frame, sgcp_metadata) in enumerate(
                frame_items,
                start=1):
            ego = next(cav for cav in eval_frame.values() if cav['ego'])
            ego_lidar_pose = ego['params']['lidar_pose']

            inference_frame = (
                collapse_frame_to_receiver_pointcloud(
                    eval_frame,
                    ego_lidar_pose)
                if args.collapse_to_ego_pointcloud else eval_frame)
            ret = run_opencood_inference(
                manager,
                inference_frame,
                ego_lidar_pose,
                debug_output=args.debug_opencood_output)

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
                update_coverage_quality_state(
                    sgcp_coverage_state,
                    sgcp_metadata,
                    pred_count,
                    gt_count)
                sgcp_summaries.append(sgcp_metadata)
                sgcp_trace_rows.append(trace_row(
                    scenario_id,
                    timestamp,
                    sgcp_metadata,
                    eval_frame,
                    pred_count=pred_count,
                    gt_count=gt_count))
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
            if canonical_ret is not None:
                sample_label = (
                    'full_reference'
                    if sgcp_metadata is None
                    else 'receiver_sample')
                object_rows.extend(object_diagnostic_rows(
                    scenario_id,
                    timestamp,
                    sample_label,
                    sgcp_metadata,
                    canonical_ret[2],
                    canonical_ret[3] if len(canonical_ret) > 3 else None,
                    canonical_ret[0],
                    canonical_ret[1],
                    pred_box_tensor,
                    pred_score,
                    args.object_diagnostics_iou))
            before_stats = result_stat_lengths(manager.result_stat)
            manager.submit_results(
                pred_box_tensor,
                pred_score,
                gt_box_tensor,
                with_stats=True,
                force=True)
            append_eval_stats_row(
                eval_stats_rows,
                scenario_id,
                timestamp,
                'receiver_sample',
                '' if sgcp_metadata is None else
                sgcp_metadata.get('receiver_id', ''),
                sgcp_metadata,
                before_stats,
                manager.result_stat)

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
    write_trace_csv(args.sgcp_trace_output, sgcp_trace_rows)
    write_object_diagnostics_csv(
        args.object_diagnostics_output,
        object_rows)
    write_eval_stats_csv(args.eval_stats_output, eval_stats_rows)


if __name__ == '__main__':
    main()
