#!/usr/bin/env pwsh
# reorganize.ps1
param()

$ROOT = "C:\Users\nakas\OneDrive\Desktop\reserch_final"

Write-Host "=== Reorganizing project ===" -ForegroundColor Cyan

function Move-Safe {
    param([string]$Src, [string]$Dst)
    if (Test-Path $Src) {
        $dstDir = Split-Path $Dst -Parent
        if (-not (Test-Path $dstDir)) {
            New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
        }
        Move-Item -Path $Src -Destination $Dst -Force
        Write-Host "  [OK] $(Split-Path $Src -Leaf) -> $Dst" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] Not found: $Src" -ForegroundColor Yellow
    }
}

# --- [1] scripts/
Write-Host "`n--- [1/6] scripts/ ---" -ForegroundColor Cyan
Move-Safe "$ROOT\traingraph_FusionPV.py" "$ROOT\scripts\train_mmk.py"

# --- [2] tools/eval/
Write-Host "`n--- [2/6] tools/eval/ ---" -ForegroundColor Cyan
Move-Safe "$ROOT\tools\eval_mmk_from_ckpt.py"   "$ROOT\tools\eval\eval_mmk_from_ckpt.py"
Move-Safe "$ROOT\tools\calculate_metrics.py"     "$ROOT\tools\eval\calculate_metrics.py"
Move-Safe "$ROOT\tools\compare_models_kw.py"     "$ROOT\tools\eval\compare_models_kw.py"

# --- [3] tools/plot/
Write-Host "`n--- [3/6] tools/plot/ ---" -ForegroundColor Cyan
Move-Safe "$ROOT\tools\plot_pred_vs_true_itr.py"          "$ROOT\tools\plot\plot_pred_vs_true_itr.py"
Move-Safe "$ROOT\tools\make_error_plots_24.py"            "$ROOT\tools\plot\make_error_plots_24.py"
Move-Safe "$ROOT\tools\compare_models_meanplots.py"       "$ROOT\tools\plot\compare_models_meanplots.py"
Move-Safe "$ROOT\tools\compare_mean96h_mmk_vs_itr.py"    "$ROOT\tools\plot\compare_mean96h_mmk_vs_itr.py"
Move-Safe "$ROOT\tools\plot_mmk_optimized.py"             "$ROOT\tools\plot\plot_mmk_optimized.py"
Move-Safe "$ROOT\tools\final_visual_showdown.py"          "$ROOT\tools\plot\final_visual_showdown.py"
Move-Safe "$ROOT\tools\plot_final_battle.py"              "$ROOT\tools\plot\plot_final_battle.py"

# --- [4] tools/interpret/
Write-Host "`n--- [4/6] tools/interpret/ ---" -ForegroundColor Cyan
Move-Safe "$ROOT\tools\make_gate_heatmap_featuretoken.py"                "$ROOT\tools\interpret\make_gate_heatmap_featuretoken.py"
Move-Safe "$ROOT\tools\extract_gate_heatmap_data.py"                     "$ROOT\tools\interpret\extract_gate_heatmap_data.py"
Move-Safe "$ROOT\tools\extract_last_window_96xexpert_heatmap.py"        "$ROOT\tools\interpret\extract_last_window_96xexpert_heatmap.py"
Move-Safe "$ROOT\tools\fit_taylor_by_timeband_ghi1h.py"                 "$ROOT\tools\interpret\fit_taylor_by_timeband_ghi1h.py"
Move-Safe "$ROOT\tools\plot_input_response_curve.py"                     "$ROOT\tools\interpret\plot_input_response_curve.py"
Move-Safe "$ROOT\extract_basis_response_same_location.py"               "$ROOT\tools\interpret\extract_basis_response_same_location.py"
Move-Safe "$ROOT\extract_taylorkan_formulas.py"                          "$ROOT\tools\interpret\extract_taylorkan_formulas.py"
Move-Safe "$ROOT\rank_taylor_layer2_contrib.py"                          "$ROOT\tools\interpret\rank_taylor_layer2_contrib.py"
Move-Safe "$ROOT\gates_new.py"                                           "$ROOT\tools\interpret\gates_new.py"

# --- [5] archive/
Write-Host "`n--- [5/6] archive/ ---" -ForegroundColor Cyan

# old_training_scripts
Move-Safe "$ROOT\train.py"                  "$ROOT\archive\old_training_scripts\train.py"
Move-Safe "$ROOT\train_mmk_fusionpv_v3.py" "$ROOT\archive\old_training_scripts\train_mmk_fusionpv_v3.py"
Move-Safe "$ROOT\traingraph.py"             "$ROOT\archive\old_training_scripts\traingraph.py"
Move-Safe "$ROOT\traingraph_V2.py"          "$ROOT\archive\old_training_scripts\traingraph_V2.py"
Move-Safe "$ROOT\traingraph_revin_v2.py"    "$ROOT\archive\old_training_scripts\traingraph_revin_v2.py"

# old_analysis (root)
Move-Safe "$ROOT\analyze_gates_latest.py"       "$ROOT\archive\old_analysis\analyze_gates_latest.py"
Move-Safe "$ROOT\extract_taylor_formula.py"     "$ROOT\archive\old_analysis\extract_taylor_formula.py"
Move-Safe "$ROOT\extract_taylor_formula_A11.py" "$ROOT\archive\old_analysis\extract_taylor_formula_A11.py"
Move-Safe "$ROOT\extract_kan_formulas.py"       "$ROOT\archive\old_analysis\extract_kan_formulas.py"
Move-Safe "$ROOT\loss_peak_weighted_v2.py"      "$ROOT\archive\old_analysis\loss_peak_weighted_v2.py"
Move-Safe "$ROOT\predict_save.py"               "$ROOT\archive\old_analysis\predict_save.py"
Move-Safe "$ROOT\plot_formula_curves.py"        "$ROOT\archive\old_analysis\plot_formula_curves.py"
Move-Safe "$ROOT\plot_var_expert_stats.py"      "$ROOT\archive\old_analysis\plot_var_expert_stats.py"
Move-Safe "$ROOT\ray_tune.py"                   "$ROOT\archive\old_analysis\ray_tune.py"
Move-Safe "$ROOT\simulate_nonlinear_effect.py"  "$ROOT\archive\old_analysis\simulate_nonlinear_effect.py"

# old_analysis (tools)
Move-Safe "$ROOT\tools\make_gating_heatmap_test_mmk.py"             "$ROOT\archive\old_analysis\make_gating_heatmap_test_mmk.py"
Move-Safe "$ROOT\tools\compare_pred_vs_true_mmk_fusionpv.py"        "$ROOT\archive\old_analysis\compare_pred_vs_true_mmk_fusionpv.py"
Move-Safe "$ROOT\tools\make_error_plots_24_time.py"                  "$ROOT\archive\old_analysis\make_error_plots_24_time.py"
Move-Safe "$ROOT\tools\plot_pred_vs_true_revin.py"                   "$ROOT\archive\old_analysis\plot_pred_vs_true_revin.py"
Move-Safe "$ROOT\tools\plot_pred_vs_true_peakrunner.py"              "$ROOT\archive\old_analysis\plot_pred_vs_true_peakrunner.py"
Move-Safe "$ROOT\tools\plot_pred_vs_true_itr_peak_lastwindow.py"    "$ROOT\archive\old_analysis\plot_pred_vs_true_itr_peak_lastwindow.py"
Move-Safe "$ROOT\tools\plot_prediction.py"                           "$ROOT\archive\old_analysis\plot_prediction.py"
Move-Safe "$ROOT\tools\plot_prediction_kw.py"                        "$ROOT\archive\old_analysis\plot_prediction_kw.py"
Move-Safe "$ROOT\tools\plot_results.py"                              "$ROOT\archive\old_analysis\plot_results.py"
Move-Safe "$ROOT\tools\plot_mmk_24h_last_window_kw.py"              "$ROOT\archive\old_analysis\plot_mmk_24h_last_window_kw.py"
Move-Safe "$ROOT\tools\plot_last4days_concat.py"                     "$ROOT\archive\old_analysis\plot_last4days_concat.py"
Move-Safe "$ROOT\tools\plot_hardzero_compare.py"                     "$ROOT\archive\old_analysis\plot_hardzero_compare.py"
Move-Safe "$ROOT\tools\plot_aligned96h_stitched.py"                  "$ROOT\archive\old_analysis\plot_aligned96h_stitched.py"
Move-Safe "$ROOT\tools\normalize_24_fromseries_npy.py"               "$ROOT\archive\old_analysis\normalize_24_fromseries_npy.py"
Move-Safe "$ROOT\tools\evaluate_mmk_kw_24h.py"                       "$ROOT\archive\old_analysis\evaluate_mmk_kw_24h.py"
Move-Safe "$ROOT\tools\build_dataset_24in24out_hardzero.py"         "$ROOT\archive\old_analysis\build_dataset_24in24out_hardzero.py"
Move-Safe "$ROOT\tools\build_dataset_from_series_24in24out_hardzero.py" "$ROOT\archive\old_analysis\build_dataset_from_series_24in24out_hardzero.py"
Move-Safe "$ROOT\tools\build_dataset_pv_windows.py"                  "$ROOT\scripts\prepare_dataset.py"
Move-Safe "$ROOT\tools\build_dataset_pv_windows_noscale.py"         "$ROOT\archive\old_analysis\build_dataset_pv_windows_noscale.py"
Move-Safe "$ROOT\tools\build_dataset_pv_windows_noscale_itr.py"     "$ROOT\archive\old_analysis\build_dataset_pv_windows_noscale_itr.py"
Move-Safe "$ROOT\tools\check_correlation.py"                         "$ROOT\archive\old_analysis\check_correlation.py"
Move-Safe "$ROOT\tools\check_data.py"                                "$ROOT\archive\old_analysis\check_data.py"
Move-Safe "$ROOT\tools\check_scale.py"                               "$ROOT\archive\old_analysis\check_scale.py"

# tools/unused folder
$unusedFolder = "$ROOT\tools\" + [char]0x4F7F + [char]0x3063 + [char]0x3066 + [char]0x306A + [char]0x3044 + [char]0x304B + [char]0x3082"
if (Test-Path $unusedFolder) {
    Move-Item -Path $unusedFolder -Destination "$ROOT\archive\old_analysis\tools_unused" -Force
    Write-Host "  [OK] tools/unused -> archive/old_analysis/tools_unused" -ForegroundColor Green
}

# old_models
Move-Safe "$ROOT\easytsf\model\SparseRMoK.py"       "$ROOT\archive\old_models\SparseRMoK.py"
Move-Safe "$ROOT\easytsf\model\TimeLLM.py"           "$ROOT\archive\old_models\TimeLLM.py"
Move-Safe "$ROOT\easytsf\model\iKransformer.py"      "$ROOT\archive\old_models\iKransformer.py"
Move-Safe "$ROOT\easytsf\model\MMK.py"               "$ROOT\archive\old_models\MMK.py"
Move-Safe "$ROOT\easytsf\model\DenseRMoK.py"         "$ROOT\archive\old_models\DenseRMoK.py"
Move-Safe "$ROOT\easytsf\model\MMK_Mix_RevIN.py"     "$ROOT\archive\old_models\MMK_Mix_RevIN.py"
Move-Safe "$ROOT\easytsf\model\MMK_Mix_RevIN_bn.py"  "$ROOT\archive\old_models\MMK_Mix_RevIN_bn.py"
Move-Safe "$ROOT\easytsf\model\MMK_Mix_Time.py"      "$ROOT\archive\old_models\MMK_Mix_Time.py"
Move-Safe "$ROOT\easytsf\model\MMK_Mix_24.py"        "$ROOT\archive\old_models\MMK_Mix_24.py"
Move-Safe "$ROOT\easytsf\model\iTransformer.py"      "$ROOT\archive\old_models\iTransformer.py"

# misc (loose images/csv in root)
Move-Safe "$ROOT\final_showdown_plot.png"               "$ROOT\archive\misc\final_showdown_plot.png"
Move-Safe "$ROOT\kan_learned_curves.png"                "$ROOT\archive\misc\kan_learned_curves.png"
Move-Safe "$ROOT\research_style_heatmap.png"            "$ROOT\archive\misc\research_style_heatmap.png"
Move-Safe "$ROOT\simulation_result_kwh.png"             "$ROOT\archive\misc\simulation_result_kwh.png"
Move-Safe "$ROOT\kan_symbolic_formulas.csv"             "$ROOT\archive\misc\kan_symbolic_formulas.csv"
Move-Safe "$ROOT\presentation_v12_layer0_red.png"       "$ROOT\archive\misc\presentation_v12_layer0_red.png"
Move-Safe "$ROOT\presentation_v12_layer1_red.png"       "$ROOT\archive\misc\presentation_v12_layer1_red.png"
Move-Safe "$ROOT\presentation_v12_layer2_red.png"       "$ROOT\archive\misc\presentation_v12_layer2_red.png"
Move-Safe "$ROOT\presentation_results_v12"              "$ROOT\archive\misc\presentation_results_v12"
Move-Safe "$ROOT\research_formula_analysis"             "$ROOT\archive\misc\research_formula_analysis"
Move-Safe "$ROOT\research_formula_analysis_v1"          "$ROOT\archive\misc\research_formula_analysis_v1"
Move-Safe "$ROOT\research_gate_results_latest"          "$ROOT\archive\misc\research_gate_results_latest"
Move-Safe "$ROOT\research_gate_results_pro"             "$ROOT\archive\misc\research_gate_results_pro"
Move-Safe "$ROOT\research_gate_results_v1_A11"          "$ROOT\archive\misc\research_gate_results_v1_A11"

# Japanese-named folders using encoded path
$jFolders = @(
    @{ src = "$ROOT\" + [char]0x3082 + [char]0x3046 + [char]0x4F7F + [char]0x308F + [char]0x306A + [char]0x3044; dst = "$ROOT\archive\misc\old_misc" },
    @{ src = "$ROOT\2002" + [char]0x30DF + [char]0x30B9 + [char]0x304B + [char]0x306A; dst = "$ROOT\archive\misc\2002_misc" },
    @{ src = "$ROOT\0111" + [char]0x7D50 + [char]0x679C; dst = "$ROOT\archive\misc\0111_results" },
    @{ src = "$ROOT\0112" + [char]0x7D50 + [char]0x679C; dst = "$ROOT\archive\misc\0112_results" },
    @{ src = "$ROOT\0113" + [char]0x7D50 + [char]0x679C; dst = "$ROOT\archive\misc\0113_results" },
    @{ src = "$ROOT\0115" + [char]0x7D50 + [char]0x679C; dst = "$ROOT\archive\misc\0115_results" }
)
foreach ($pair in $jFolders) {
    if (Test-Path $pair.src) {
        New-Item -ItemType Directory -Path (Split-Path $pair.dst -Parent) -Force | Out-Null
        Move-Item -Path $pair.src -Destination $pair.dst -Force
        Write-Host "  [OK] $($pair.src) -> $($pair.dst)" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] Not found: $($pair.src)" -ForegroundColor Yellow
    }
}

# --- [6] config/tsukuba_conf/ 整備
Write-Host "`n--- [6/6] config/ ---" -ForegroundColor Cyan
New-Item -ItemType Directory -Path "$ROOT\config\tsukuba_conf\baselines" -Force | Out-Null
Move-Safe "$ROOT\config\tsukuba_conf\NLinear_Tsukuba_96for96.py"           "$ROOT\config\tsukuba_conf\baselines\NLinear_Tsukuba_96for96.py"
Move-Safe "$ROOT\config\tsukuba_conf\PatchTST_Tsukuba_96for96.py"          "$ROOT\config\tsukuba_conf\baselines\PatchTST_Tsukuba_96for96.py"
Move-Safe "$ROOT\config\tsukuba_conf\RLinear_Tsukuba_96for96.py"           "$ROOT\config\tsukuba_conf\baselines\RLinear_Tsukuba_96for96.py"
Move-Safe "$ROOT\config\tsukuba_conf\RLinear_Tsukuba_4for1.py"             "$ROOT\config\tsukuba_conf\baselines\RLinear_Tsukuba_4for1.py"
Move-Safe "$ROOT\config\tsukuba_conf\MMK_FusionPV_Flatten_A9_96.py"        "$ROOT\archive\old_configs\MMK_FusionPV_Flatten_A9_96.py"
Move-Safe "$ROOT\config\tsukuba_conf\MMK_FusionPV_RevIN_96_8.py"           "$ROOT\archive\old_configs\MMK_FusionPV_RevIN_96_8.py"
Move-Safe "$ROOT\config\tsukuba_conf\MMK_Mix_FusionPV_RevIN_PV_96.py"      "$ROOT\archive\old_configs\MMK_Mix_FusionPV_RevIN_PV_96.py"
Move-Safe "$ROOT\config\tsukuba_conf\MMK_Mix_PV_96.py"                     "$ROOT\archive\old_configs\MMK_Mix_PV_96.py"
Move-Safe "$ROOT\config\tsukuba_conf\MMK_PVProcessed_96for96_hardzero.py"  "$ROOT\archive\old_configs\MMK_PVProcessed_96for96_hardzero.py"
Move-Safe "$ROOT\config\tsukuba_conf\iTransformer_peak_96.py"              "$ROOT\archive\old_configs\iTransformer_peak_96.py"
Move-Safe "$ROOT\config\tsukuba_conf\iTransformer_peak_96_noscale.py"      "$ROOT\archive\old_configs\iTransformer_peak_96_noscale.py"

$unusedCfg = "$ROOT\config\tsukuba_conf\" + [char]0x4F7F + [char]0x308F + [char]0x306A + [char]0x3044
if (Test-Path $unusedCfg) {
    Move-Item -Path $unusedCfg -Destination "$ROOT\archive\old_configs\tsukuba_unused" -Force
    Write-Host "  [OK] config/tsukuba_conf/unused -> archive/old_configs/tsukuba_unused" -ForegroundColor Green
}

Write-Host "`n=== Done! ===" -ForegroundColor Cyan
Write-Host "Next steps:"
Write-Host "  git init"
Write-Host "  git add ."
Write-Host "  git status"
Write-Host "  git commit -m 'Initial commit: minimal reproducible structure'"
