# LGCP RSU BEV Attentive Fusion

This run crops member point clouds by LGCP area, projects them into a shared reference/RSU lidar frame, encodes one scatter BEV canvas per area leader, and uses the existing PointPillar attentive backbone for RSU-level feature fusion.

- frames: `21`
- area rows: `105`
- query mode: `mean`
- packet granularity: `leader`
- AP@0.3/AP@0.5/AP@0.7: `0.618795/0.610993/0.583450`
- member upload bytes: `2034352`
- leader feature sparse-cell bytes: `19707520`
