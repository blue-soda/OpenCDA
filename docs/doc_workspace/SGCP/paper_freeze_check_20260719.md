# SGCP Paper Freeze Check - 2026-07-19

This document records the current pre-freeze state of the SGCP paper draft. The paper source lives outside the OpenCDA git repository, so this file anchors the static checks and remaining risks inside the SGCP documentation workspace.

## Paper Files

- Main paper: `C:\Workspace\icdcs-paper\SGCP\main.tex`
- Bibliography: `C:\Workspace\icdcs-paper\SGCP\Reference.bib`
- Figure directory: `C:\Workspace\icdcs-paper\SGCP\fig`
- Review file: `C:\Workspace\icdcs-paper\SGCP\SGCP-review.txt`
- OpenCDA snapshot: `docs\doc_workspace\SGCP\artifacts\paper_freeze_snapshot_20260719`

## Current Draft State

- The experimental section is synchronized to the attentive forward-writing candidate.
- Table 1 uses protocol-native results with Pure late as a prediction-sharing reference, Full 20-CAV early fusion as an upper reference, FullPerception-PCS as a paper-faithful scheduler with raw-LiDAR adaptation, EdgeCooper-HD as an edge-assisted reference, and SGCP-PAPG as the RSU-free proposed protocol.
- Table 3 is explicitly scoped as an SGCP-compatible scheduler comparison, not a protocol-native system ranking.
- Table 4 uses the attentive parameter sensitivity results.
- Figure 1/2/3 PDFs in the paper figure directory have been replaced by attentive versions.
- SMARTFORM is cited and discussed to address the generic coalition-formation novelty concern.
- Strong claims around strict real-time guarantees, universal SOTA dominance, and unconditional exact-potential convergence have been softened.

## Static Check Results

Static check command was run from `C:\Workspace\OpenCDA` using Python over `main.tex` and `Reference.bib`.

Results:

- Citations: 43 citation occurrences, 29 unique citation keys, 0 missing bibliography keys.
- Labels: 32 labels, 0 duplicates.
- References: 22 `ref` / `eqref` targets, 0 missing labels.
- Figures: 7 `includegraphics` entries, 0 missing files.
- Environment balance:
  - `table`: 3 begin / 3 end
  - `figure`: 2 begin / 2 end
  - `figure*`: 4 begin / 4 end
  - `tabular`: 4 begin / 4 end
  - `equation`: 28 begin / 28 end
  - `algorithm`: 3 begin / 3 end
  - `itemize`: 5 begin / 5 end

High-risk phrase scan no longer finds:

- `eliminating blind spots`
- `Extensive simulations`
- `operates entirely`
- `Nearly all`
- `strict real-time`
- `outperforms all`
- `guaranteed`

## Remaining Risks

- The current machine has no `pdflatex`, `latexmk`, `bibtex`, `xelatex`, `lualatex`, `tectonic`, or WSL LaTeX toolchain. The paper still needs a real PDF compile and visual check.
- `main.tex` and `Reference.bib` are outside OpenCDA git. A snapshot has been copied into `artifacts/paper_freeze_snapshot_20260719`, but the active paper directory still needs separate archival or version control.
- Remote early-fusion fine-tuning on `mindspore-187` has not produced a new checkpoint because GPUs remain occupied. The attentive candidate is the current forward-writing version; any future checkpoint replacement must trigger a new full table/figure artifact version.
- Table 3 should continue to be described as a scheduler comparison inside the SGCP scaffold, because PACP-LiDAR and EdgeCooper-HD are proxy/reference methods with different information assumptions.
- Runtime wording should remain control-plane near-real-time, not detector-inclusive 100 ms guarantee.

## Next Actions

1. Compile the paper in an environment with a LaTeX toolchain.
2. Visually inspect table widths, figure placement, caption length, citation numbering, and any overfull boxes.
3. If compilation succeeds, archive the generated PDF path and update `paper_artifact_index.md`.
4. If the remote fine-tune watcher produces a checkpoint, create a new artifact directory and rerun Table 1/2/3/4 plus Figure 1/2/3 before changing paper numbers.
