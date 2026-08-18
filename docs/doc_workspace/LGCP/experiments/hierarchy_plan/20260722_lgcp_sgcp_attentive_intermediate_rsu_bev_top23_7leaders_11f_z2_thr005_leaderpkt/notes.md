# LGCP RSU BEV Attentive Fusion

This run crops member point clouds by LGCP area, projects them into a shared reference/RSU lidar frame, encodes one scatter BEV canvas per area leader, and uses the existing PointPillar attentive backbone for RSU-level feature fusion.

- frames: `11`
- area rows: `65`
- query mode: `mean`
- packet granularity: `leader`
- AP@0.3/AP@0.5/AP@0.7: `0.663529/0.556226/0.252941`
- member upload bytes: `996368`
- leader feature sparse-cell bytes: `3822336`
