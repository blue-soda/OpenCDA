# Attentive Pareto Notes

Generated: 2026-07-19

Source:

- `pareto_attentive_source.csv`

Figures:

- `figure1_pareto_ap03_attentive.png/.pdf`
- `figure1_pareto_ap07_attentive.png/.pdf`

Key reading:

- `SGCP_PAPG_attentive` reaches `0.87/0.81/0.36` at `62.54 Mbps`.
- `PureLateBroadcast80_attentive` reaches `0.82/0.65/0.28` with `1.37 Mbps` box-broadcast overhead.
- `EdgeCooperHD_attentive` reaches `0.85/0.74/0.35` at `65.40 Mbps`.
- `PACP_LiDAR_attentive` reaches `0.88/0.79/0.37` at `86.56 Mbps`.
- `Full20Early_attentive` is the raw-sharing upper reference: `0.88/0.85/0.45` at `118.71 Mbps`.

Claim boundary:

- For AP@0.3, PACP-LiDAR slightly exceeds SGCP but uses substantially more raw-LiDAR traffic.
- For AP@0.5, SGCP is the best non-upper-reference point in this source set.
- For AP@0.7, Density/Link-aware and PACP-LiDAR can slightly exceed SGCP, but at higher traffic; Full20Early remains the localization upper reference.
- Legacy early-checkpoint Pareto points should be treated as checkpoint-reference artifacts, not as the default forward-writing figure.
