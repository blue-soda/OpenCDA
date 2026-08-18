# LGCP RSU BEV Attentive Fusion

This run crops member point clouds by LGCP area, projects them into a shared reference/RSU lidar frame, encodes one scatter BEV canvas per area leader, and uses the existing PointPillar attentive backbone for RSU-level feature fusion.

- frames: `1`
- area rows: `4`
- query mode: `mean`
- AP@0.3/AP@0.5/AP@0.7: `0.888889/0.763889/0.763889`
- member upload bytes: `4032`
- leader feature sparse-cell bytes: `90240`
