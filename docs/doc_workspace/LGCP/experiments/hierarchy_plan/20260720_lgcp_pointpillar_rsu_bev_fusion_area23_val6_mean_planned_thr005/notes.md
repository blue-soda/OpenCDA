# LGCP RSU BEV Attentive Fusion

This run crops member point clouds by LGCP area, projects them into a shared reference/RSU lidar frame, encodes one scatter BEV canvas per area leader, and uses the existing PointPillar attentive backbone for RSU-level feature fusion.

- frames: `6`
- area rows: `138`
- query mode: `mean`
- AP@0.3/AP@0.5/AP@0.7: `0.194570/0.148743/0.084005`
- member upload bytes: `320528`
- leader feature sparse-cell bytes: `3321344`
