# LGCP Local-to-Global Ablation Alignment

This directory aligns the 11-frame Top-30 box-level hierarchy late-fusion result with existing flat selective-sharing baselines.

## Key Result

| Method | Structure | Budget | AP@0.5 | AP@0.7 | Bytes / frame | Byte proxy |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Full sharing | Flat | 20 agents | 0.839868 | 0.526521 | 190000.000000 | fixed 10KB per non-ego agent |
| Random | Flat | 10 agents | 0.598993 | 0.380556 | 90000.000000 | fixed 10KB per non-ego agent |
| Confidence top-k | Flat | 10 agents | 0.624088 | 0.436000 | 90000.000000 | fixed 10KB per non-ego agent |
| Comm-aware top-k | Flat | 10 agents | 0.686146 | 0.545736 | 90000.000000 | fixed 10KB per non-ego agent |
| Area-aware union | Flat | 10 agents | 0.676678 | 0.538273 | 90000.000000 | fixed 10KB per non-ego agent |
| LGCP Top-30 box late fusion | Local-to-global hierarchy | 30 areas | 0.602748 | 0.506345 | 119415.272727 | scheduled raw-slice plan |

## Interpretation

- LGCP Top-30 box-level hierarchy is now evaluated with real OpenCOOD calls over the same 11 local frames.
- Its AP@0.5 is close to random 10-agent flat selection and below the stronger flat top-k baselines, but its AP@0.7 is much stronger than random and closer to area-aware / communication-aware baselines.
- The byte columns are not a perfectly uniform accounting model. Existing flat baselines use a fixed selected-agent packet proxy, while LGCP uses raw-slice-aware scheduled upload bytes. Any manuscript table must explicitly label this.
- The result supports a careful claim: the current box-level hierarchy adapter validates the local-to-global mechanism path, but it does not yet prove that LGCP outperforms strong flat selective-sharing baselines in perception AP.
- A stronger claim requires either neural feature slicing / intermediate fusion or a recalibrated common byte budget for all methods.

## Next Step

Run a common-byte-budget comparison:

- flat selective sharing under the same `119.42 KB/frame` LGCP scheduled raw-slice budget, or
- LGCP under the same fixed `90 KB/frame` flat 10-agent budget.
