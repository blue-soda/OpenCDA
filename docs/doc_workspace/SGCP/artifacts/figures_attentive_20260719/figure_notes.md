# Attentive SGCP Figures

Generated: 2026-07-19

Source manifests:

- `../attentive_protocol_20260719/protocol_native_attentive_manifest.csv`
- `../attentive_fusion_ablation_20260719/fusion_scaffold_attentive_manifest.csv`
- `../attentive_scheduler_comparison_20260719/scheduler_comparison_attentive_manifest.csv`

Figures:

- `figure2_protocol_breakdown_attentive.png/.pdf`
- `figure3_fusion_contribution_attentive.png/.pdf`
- `figure4_scheduler_comparison_attentive.png/.pdf`

Claim boundary:

- These figures use the attentive intermediate checkpoint as the detector for SGCP and the comparable baselines.
- They supersede the legacy early-checkpoint breakdown figures for forward SGCP writing, but they do not delete the legacy artifacts.
- Pure late is a prediction-sharing reference. Its plotted communication label is `box 1.37`, the 80 B/box one-hop broadcast estimate from `pure_late_attentive_box_comm_80`.
- FullPerception-PCS is plotted as a paper-faithful PCS scheduling + raw-LiDAR full-sender adaptation: `0.63/0.49/0.17`, `raw 32.1`.
- Full 20-CAV early fusion remains an upper reference, not a baseline algorithm.
- EdgeCooperHD and PACP-LiDAR remain reference/proxy schedulers with stronger information assumptions or higher traffic than SGCP.
