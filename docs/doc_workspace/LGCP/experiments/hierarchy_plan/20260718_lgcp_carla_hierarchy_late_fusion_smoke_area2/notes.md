# LGCP Box-Level Hierarchy Late Fusion

This run evaluates LGCP hierarchy by running OpenCOOD for each leader area-task group, slicing predictions by LGCP area in world coordinates, and late-fusing leader local predictions into an RSU global result.

It is a model-calling box-level hierarchy ablation, not neural feature tensor slicing.

- frames: `1`
- assignment rows: `2`
- cached group inference calls: `2`
- AP@0.5: `1.000000`
