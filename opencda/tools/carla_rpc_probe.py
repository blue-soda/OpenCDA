# -*- coding: utf-8 -*-
"""Probe CARLA Python RPC readiness before online OpenCDA runs."""

import argparse
import sys
import time

import carla


def parse_args():
    parser = argparse.ArgumentParser(
        description='Wait until CARLA get_world() returns a map name.')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--timeout', type=float, default=30.0,
                        help='Per-RPC timeout in seconds.')
    parser.add_argument('--wait', type=float, default=180.0,
                        help='Total wall-clock wait budget in seconds.')
    parser.add_argument('--interval', type=float, default=5.0,
                        help='Sleep interval between attempts in seconds.')
    parser.add_argument('--expect-map', default='',
                        help='Optional substring expected in map name.')
    return parser.parse_args()


def main():
    args = parse_args()
    deadline = time.time() + args.wait
    last_error = None
    while time.time() < deadline:
        try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(args.timeout)
            world = client.get_world()
            map_name = world.get_map().name
            if args.expect_map and args.expect_map not in map_name:
                print('CARLA_RPC_READY_WRONG_MAP map=%s expected=%s' %
                      (map_name, args.expect_map))
                return 2
            print('CARLA_RPC_READY map=%s' % map_name)
            return 0
        except RuntimeError as error:
            last_error = error
            time.sleep(args.interval)

    print('CARLA_RPC_NOT_READY last_error=%s' % last_error)
    return 1


if __name__ == '__main__':
    sys.exit(main())
