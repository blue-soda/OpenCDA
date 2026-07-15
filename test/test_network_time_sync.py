# -*- coding: utf-8 -*-
"""Regression tests for OpenCDA/NS3 time-base handling."""

from opencda.core.networking.network_manager import NetworkManager


class DummyCavWorld(object):
    fixed_delta_seconds = 0.05

    def get_vehicle_manager(self, vehicle_id):
        return None


def test_network_time_slot_matches_carla_fixed_delta():
    manager = NetworkManager(
        DummyCavWorld(),
        {
            'subchannel_num': 10,
            'subchannel_bandwidth': 4,
            'time_slot': 0.05,
            'use_ns3': False,
        })

    assert manager.time_slot == 0.05
    assert manager.current_time_slot == 0
    assert manager.current_sim_time == 0.0

    manager.advance_time_slot()

    assert manager.current_time_slot == 1
    assert manager.tick_count == 1
    assert manager.current_sim_time == 0.05
    assert len(manager.history) == 1
    assert manager.history[0]['slot_index'] == 0


def test_multiple_network_slots_track_carla_time():
    manager = NetworkManager(
        DummyCavWorld(),
        {
            'subchannel_num': 10,
            'subchannel_bandwidth': 4,
            'time_slot': 0.05,
            'use_ns3': False,
        })

    for _ in range(10):
        manager.advance_time_slot()

    assert manager.current_time_slot == 10
    assert manager.tick_count == 10
    assert abs(manager.current_sim_time - 0.5) < 1e-9
    assert len(manager.history) == 10
