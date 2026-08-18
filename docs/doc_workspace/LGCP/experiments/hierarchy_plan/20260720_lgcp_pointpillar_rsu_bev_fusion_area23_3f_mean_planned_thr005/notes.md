# LGCP RSU BEV Attentive Fusion

This run crops member point clouds by LGCP area, projects them into a shared reference/RSU lidar frame, encodes one scatter BEV canvas per area leader, and uses the existing PointPillar attentive backbone for RSU-level feature fusion.

- frames: `3`
- area rows: `69`
- query mode: `mean`
- AP@0.3/AP@0.5/AP@0.7: `0.166667/0.130019/0.119298`
- member upload bytes: `138304`
- leader feature sparse-cell bytes: `1587712`
