# Detector Checkpoint Sensitivity Manifest

Date: 2026-07-19

This artifact consolidates the detector/checkpoint probes used to protect the SGCP paper narrative. It originally served as a sensitivity manifest; after the full attentive Table/Figure rerun on 2026-07-19, the attentive checkpoint became the current forward-writing candidate. The artifact still records which rows are legacy protocol references, which rows are detector sensitivity, and which rows should be rejected.

Machine-readable source:

```text
docs\doc_workspace\SGCP\artifacts\checkpoint_sensitivity_20260719\detector_checkpoint_sensitivity_manifest.csv
```

## Current Decision

- The legacy SGCP raw-LiDAR protocol used `pointpillar_early_fusion` + PAPG + inter-cluster `naive_late_fusion()`: `0.81/0.78/0.39`, `62.54 Mbps`.
- The legacy Pure late row used the same `pointpillar_early_fusion` checkpoint in singleton mode: `0.82/0.76/0.37`, with raw LiDAR payload `0` but nonzero detection-box overhead.
- The actual `pointpillar_late_fusion` checkpoint is stronger as a local detector, but it changes the detector family and is therefore sensitivity evidence, not a fair raw-LiDAR baseline.
- The attentive intermediate checkpoint has now been promoted from sensitivity evidence to the current forward-writing candidate because Table 1, Table 2, Table 3, Figure 1, Figure 2, Figure 3 and Figure 4 were all rerun under the same attentive detector. Under this detector, SGCP-PAPG reaches `0.87/0.81/0.36`, Pure late controlled reaches `0.82/0.65/0.28`, and Full20Early reaches `0.88/0.85/0.45`.
- COSDH is not directly usable in the current collapsed raw-point input path. Weight transplant is negative, and the real COSDH model currently produces zero final boxes in the one-frame smoke test.

## Paper-Writing Boundary

The attentive checkpoint result in `C:\Workspace\icdcs-paper\SGCP\main.tex` should be read as the current forward-writing candidate with a clear detector boundary:

- It supports the claim that SGCP's scheduled raw point-cloud sharing can outperform prediction-only late fusion under a common detector.
- It explains that the current AP@0.7 ceiling is partly detector/localization headroom.
- It replaces the legacy `pointpillar_early_fusion` Table 1 only because all comparable baselines and figures were rerun under the same attentive detector. Future detector changes must repeat this full rerun protocol.

## Next Trigger

When the remote early-fusion fine-tune watcher produces a checkpoint, create a new artifact directory instead of overwriting this one, then rerun at minimum:

- SGCP-PAPG 41 frames;
- Pure late controlled 41 frames;
- Full20Early upper reference if the SGCP/Pure late relationship changes materially.
