# LGCP Scheduled Upload Plan Smoke

This run builds a single-slot, capacity-gated NS3 smoke plan from the raw-slice-aware LGCP upload plan. It keeps at most `10` requests per frame and assigns unique `sc_start/sc_num` values.

- reserved leader-to-RSU subchannels: `3`
- scheduled requests: `110 / 504`
- scheduled bytes: `543408 / 1313568`

Boundary: this is a scheduled replay smoke input, not a full multi-slot LGCP scheduler or final performance row.
