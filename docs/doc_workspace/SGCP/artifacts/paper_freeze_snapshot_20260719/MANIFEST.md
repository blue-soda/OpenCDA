# SGCP Paper Freeze Snapshot - 2026-07-19

This artifact snapshots the current external SGCP paper source into the OpenCDA documentation tree because `C:\Workspace\icdcs-paper\SGCP` is not an OpenCDA git repository.

## Snapshot Files

- `main.tex`: copied from `C:\Workspace\icdcs-paper\SGCP\main.tex`
- `Reference.bib`: copied from `C:\Workspace\icdcs-paper\SGCP\Reference.bib`

The figure PDFs are not duplicated here because their source artifacts already live under the SGCP artifact directories. Their paper-side hashes are recorded below for integrity checking.

## SHA256 Hashes

| File | SHA256 |
| --- | --- |
| `C:\Workspace\icdcs-paper\SGCP\main.tex` | `CCC3651594FE99D99F5FFF317E1AE1068ACF1DFCB6CC98DBB15CE5CCA481D90B` |
| `C:\Workspace\icdcs-paper\SGCP\Reference.bib` | `8C9CF97D84BCA62DBBF09A8500BCD2BCC28E0650C65F05C8F6C2A1F0AB31D635` |
| `C:\Workspace\icdcs-paper\SGCP\fig\sgcp_protocol_breakdown.pdf` | `6FC7418EEA216133FD748232F8EA35863B814250FC658C6B90A2C204556814D5` |
| `C:\Workspace\icdcs-paper\SGCP\fig\sgcp_fusion_contribution.pdf` | `E1E44A060F0C17FCF7A9204ED3D18C556A4B1AAF9A3398E7BAEB92DE394C298E` |
| `C:\Workspace\icdcs-paper\SGCP\fig\sgcp_pareto_ap03.pdf` | `08EF66AE28517EBC20FB6125A3D4E7AD0C141F6145488C227F779AEED3860A67` |
| `C:\Workspace\icdcs-paper\SGCP\fig\sgcp_pareto_ap07.pdf` | `857203775EEDC4EB3B9FABF20DE6F24CE4AC19CD62D6E40DE6F77FA4DB4A39AC` |

## Validation State

- Static LaTeX checks passed: citations, labels, references, figures and major environments are internally consistent.
- Real PDF compilation is still pending because no LaTeX toolchain is available on the current Windows or WSL environment.
- If `main.tex`, `Reference.bib` or any paper-side figure is changed after this snapshot, create a new snapshot directory instead of overwriting this one.
