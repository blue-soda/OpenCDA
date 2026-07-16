# OpenCOOD Evaluation Entry and Log Format

本文档记录 OpenCOOD 中 OPV2V / V2XSet 模型评估入口、配置路径、输出格式，以及 LGCP 后续复用 / 扩展时需要注意的位置。

## Evaluation Entry

主入口：

```text
opencood/opencood/tools/inference.py
```

标准命令：

```powershell
cd C:\Workspace\OpenCDA\opencood
python opencood/tools/inference.py --model_dir <CHECKPOINT_DIR> --fusion_method <late|early|intermediate>
```

可选输出：

```powershell
python opencood/tools/inference.py --model_dir <CHECKPOINT_DIR> --fusion_method <late|early|intermediate> --save_npy
python opencood/tools/inference.py --model_dir <CHECKPOINT_DIR> --fusion_method <late|early|intermediate> --save_vis
```

`--model_dir` 指向 checkpoint 目录。`opencood.hypes_yaml.yaml_utils.load_yaml(None, opt)` 会自动读取：

```text
<CHECKPOINT_DIR>/config.yaml
```

因此 OPV2V / V2XSet 的测试路径不是通过命令行传入，而是由 checkpoint 下的 `config.yaml` 决定。关键字段：

```yaml
root_dir: <dataset train or validate path>
validate_dir: <dataset validate/test path>
fusion:
  core_method: LateFusionDataset | EarlyFusionDataset | IntermediateFusionDataset | IntermediateFusionDatasetV2
```

README 中要求测试前确认 checkpoint `config.yaml` 的 `validate_dir` 指向 testing dataset，例如 `opv2v_data_dumping/test`。

## Supported Dataset Classes

`opencood/opencood/data_utils/datasets/__init__.py` 通过 `fusion.core_method` 构建数据集：

| `fusion.core_method` | Class | Typical use |
| --- | --- | --- |
| `LateFusionDataset` | `late_fusion_dataset.py` | late fusion / no-fusion style baseline |
| `EarlyFusionDataset` | `early_fusion_dataset.py` | early fusion / raw point aggregation |
| `IntermediateFusionDataset` | `intermediate_fusion_dataset.py` | feature-level collaboration |
| `IntermediateFusionDatasetV2` | `intermediate_fusion_dataset_v2.py` | V2X-ViT / newer intermediate fusion variants |

OPV2V and V2XSet both use this same entry pattern; the difference is mainly checkpoint config, dataset root, and model-specific yaml.

## Inference Flow

`inference.py` does the following:

1. Load `<CHECKPOINT_DIR>/config.yaml`.
2. Build dataset with `build_dataset(hypes, visualize=True, train=False)`.
3. Create dataloader with `collate_batch_test`.
4. Create model from config and load checkpoint with `train_utils.load_saved_model`.
5. Run one of:
   - `inference_utils.inference_late_fusion`
   - `inference_utils.inference_early_fusion`
   - `inference_utils.inference_intermediate_fusion`
6. Accumulate TP / FP / GT for IoU thresholds `0.3`, `0.5`, `0.7`.
7. Save final metrics to `<CHECKPOINT_DIR>/eval.yaml`.

## Evaluation Output

`opencood/opencood/utils/eval_utils.py::eval_final_results` writes:

```text
<CHECKPOINT_DIR>/eval.yaml
```

Current keys:

| Key | Meaning |
| --- | --- |
| `ap30` | AP at IoU 0.30 |
| `ap_50` | AP at IoU 0.50 |
| `ap_70` | AP at IoU 0.70 |
| `mpre_50` | precision curve at IoU 0.50 |
| `mrec_50` | recall curve at IoU 0.50 |
| `mpre_70` | precision curve at IoU 0.70 |
| `mrec_70` | recall curve at IoU 0.70 |

Console output prints:

```text
The Average Precision at IOU 0.3 is ..., The Average Precision at IOU 0.5 is ..., The Average Precision at IOU 0.7 is ...
```

With `--save_npy`, `opencood/tools/inference_utils.py::save_prediction_gt` writes to:

```text
<CHECKPOINT_DIR>/npy/
```

Current file pattern:

| File | Content |
| --- | --- |
| `%04d_pcd.npy` | ego-frame point cloud |
| `%04d_pred.npy` | predicted boxes after postprocess / NMS |
| `%04d_gt.npy_test` | ground-truth boxes; note the unusual suffix |

The unusual `%04d_gt.npy_test` suffix is likely a typo in the upstream code. LGCP scripts should not assume a standard `.npy` suffix unless this is fixed or wrapped.

## LGCP Implications

For LGCP area-level AP / recall and multi-seed evaluation:

1. Reuse `inference.py` only for whole-frame AP baselines.
2. For area-level metrics, call the same inference functions with `return_output=True` or consume the postprocessed `pred_box_tensor`, `pred_score`, and `gt_box_tensor` before global `eval_final_results`.
3. Slice predictions and GT by LGCP area after postprocess to compute area-level recall / AP. This is already the direction used by `opencda/tools/lgcp_area_confidence_eval.py`.
4. Record checkpoint config snapshot because dataset path and wild-setting noise are controlled by `<CHECKPOINT_DIR>/config.yaml`, not command-line flags.
5. For OPV2V / V2XSet comparison, keep the same checkpoint family and fusion method while changing only dataset split / seed / LGCP selection logic; otherwise AP differences may mix model and system effects.

## Recommended LGCP Logging Convention

For future LGCP runs, archive:

```text
docs/doc_workspace/LGCP/experiments/opencood_eval/<run_id>/
  config.yaml
  command.txt
  eval.yaml
  npy_manifest.csv
  area_eval_summary.csv
  notes.md
```

Minimum metadata:

| Field | Description |
| --- | --- |
| `checkpoint_dir` | OpenCOOD checkpoint directory |
| `dataset` | OPV2V / V2XSet / OpenCDA dump |
| `validate_dir` | actual path from config |
| `fusion_method` | CLI fusion method |
| `fusion.core_method` | dataset class from config |
| `wild_setting` | async / loc_err / xyz_std / rpy_std when present |
| `ap30/ap50/ap70` | copied from `eval.yaml` |
| `area_metric_source` | whole-frame / sliced prediction / calibrated proxy |

