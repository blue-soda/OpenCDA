# LGCP Scheduled Upload Plan Smoke

This run schedules every request into sequential member-to-leader and leader-to-RSU slots. It is a latency proxy and full-plan scheduler input, not an NS3 live replay by itself.

- subchannels per slot: `10`
- scheduled requests: `504 / 504`
- scheduled bytes: `1313568 / 1313568`
- mean frame scheduling latency: `73.636364 ms`
- max frame scheduling latency: `80.000000 ms`

Boundary: this is a scheduling proxy, not final perception performance.
