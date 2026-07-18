# LGCP PointPillar RSU Head Probe

This run feeds assembled RSU scatter canvases through the PointPillar backbone and detection heads. It also attempts voxel postprocessing without GT/AP.

- rows: `1`
- psm shape: `1x2x100x352`
- rm shape: `1x14x100x352`
- postprocess boxes: `18`
