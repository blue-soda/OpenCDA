$ErrorActionPreference = "Continue"
$Repo = "C:\Workspace\OpenCDA"
$OutDir = Join-Path $Repo "docs\doc_workspace\SGCP\artifacts\hybrid_followup_20260801\rho_pareto"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Rhos = @(1, 2, 3, 4, 5)
$Budgets = @(1, 5, 10, 20, 40)
foreach ($rho in $Rhos) {
  foreach ($budget in $Budgets) {
    $tag = ("rho{0}_budget{1}" -f $rho, $budget).Replace(".", "p")
    $log = Join-Path $OutDir ($tag + ".out")
    $trace = Join-Path $OutDir ($tag + "_trace.csv")
    $eval = Join-Path $OutDir ($tag + "_eval_stats.csv")
    if ((Test-Path $log) -and (Test-Path $trace) -and (Test-Path $eval)) {
      Write-Host "skip $tag"
      continue
    }
    Write-Host "running $tag"
    Set-Location $Repo
    conda run --no-capture-output -n opencda python -m opencda.tools.offline_inference `
      --dataset-root D:\Data\Carla `
      --scenario-id 2026_07_29_02_32_08 `
      --ego-cav-id 1 `
      --max-frames 41 `
      --fusion-method early `
      --coperception-yaml docs\doc_workspace\SGCP\artifacts\early_from_late_checkpoint_20260719\enable_coperception_early_from_attentive.yaml `
      --bandwidth-mhz 40 `
      --num-channels 10 `
      --channel-estimator ns3 `
      --ns3-tb-size-bytes 899 `
      --ns3-slot-duration-ms 0.5 `
      --ns3-subchannel-prbs 10 `
      --ns3-symbols-per-slot 12 `
      --ns3-mcs 28 `
      --communication-deadline-ms 60 `
      --gt-scope full-frame `
      --sgcp-constrained `
      --clustering potential_verified_cov_coalition_game `
      --resource-allocation hybrid_round_robin_dynamic_marginal `
      --sgcp-receiver-policy all-cluster-heads `
      --sgcp-upload-mode grid `
      --sgcp-inter-cluster-late-fusion `
      --n-max 5 `
      --rho-th $rho `
      --head-rb-budget 2 `
      --sgcp-frame-mbps-budget $budget `
      --upload-density-cap-rho $rho `
      --sgcp-trace-output $trace `
      --eval-stats-output $eval *> $log
    if ($LASTEXITCODE -ne 0) {
      throw "failed $tag"
    }
  }
}
