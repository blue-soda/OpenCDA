# LGCP PointPillar Feature Probe

This probe records PointPillar intermediate tensor shapes and maps LGCP world area cells to leader-local BEV feature index ranges. It does not crop features.

- rows: `5`
- scatter shape: `2x64x200x704`
- fused feature shape: `1x384x100x352`
