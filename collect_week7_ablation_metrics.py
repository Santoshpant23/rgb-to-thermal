#!/usr/bin/env python3
"""Collect Week 7 ablation metrics from ConvNeXt JSON and Swin CSV runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUNS = [
    # Three-seed ConvNeXt and Swin audit.
    ("convnext_no_reg", "ConvNeXt no-reg", "json", "week6_runs/week6_convnext_unet_no_registration_ann_arbor_robust_sigma03_seed42_e50"),
    ("convnext_no_reg", "ConvNeXt no-reg", "json", "week7_runs/week7_convnext_no_registration_ann_arbor_robust_sigma03_seed7_e50"),
    ("convnext_no_reg", "ConvNeXt no-reg", "json", "week7_runs/week7_convnext_no_registration_ann_arbor_robust_sigma03_seed123_e50"),
    ("convnext_affine_unc", "ConvNeXt affine + uncertainty", "json", "week5_runs/week5_matched_compute_aa_affine_lam0p5_robust_seed42_e50"),
    ("convnext_affine_unc", "ConvNeXt affine + uncertainty", "json", "week7_runs/week7_convnext_affine_lam0p5_ann_arbor_robust_sigma03_seed7_e50"),
    ("convnext_affine_unc", "ConvNeXt affine + uncertainty", "json", "week7_runs/week7_convnext_affine_lam0p5_ann_arbor_robust_sigma03_seed123_e50"),
    ("swin_no_reg", "Swin-T no-reg", "swin_csv", "week6_runs/ann_arbor/week6_swin_unet_ann_arbor_robust_sigma03_seed42_e50"),
    ("swin_no_reg", "Swin-T no-reg", "swin_csv", "week7_runs/ann_arbor/week7_swin_no_registration_ann_arbor_robust_sigma03_seed7_e50"),
    ("swin_no_reg", "Swin-T no-reg", "swin_csv", "week7_runs/ann_arbor/week7_swin_no_registration_ann_arbor_robust_sigma03_seed123_e50"),
    ("swin_affine", "Swin-T affine", "swin_csv", "week7_runs/ann_arbor/week7_swin_affine_lam0p5_ann_arbor_robust_sigma03_seed42_e50"),
    ("swin_affine", "Swin-T affine", "swin_csv", "week7_runs/ann_arbor/week7_swin_affine_lam0p5_ann_arbor_robust_sigma03_seed7_e50"),
    ("swin_affine", "Swin-T affine", "swin_csv", "week7_runs/ann_arbor/week7_swin_affine_lam0p5_ann_arbor_robust_sigma03_seed123_e50"),
    # Loss-confound and Week 7 loss-term ablations.
    ("convnext_no_reg_swin_loss", "ConvNeXt no-reg with Swin loss", "json", "week7_runs/week7_convnext_no_registration_swinloss_ann_arbor_robust_sigma03_seed42_e50"),
    ("convnext_no_reg_swin_loss_clean", "ConvNeXt no-reg with Swin loss, explicit no uncertainty terms", "json", "week7_runs/week7_convnext_no_registration_swinloss_no_unc_ann_arbor_robust_sigma03_seed42_e50"),
    ("convnext_affine_unc_decoupled", "ConvNeXt affine uncertainty-decoupled", "json", "week7_runs/week7_convnext_affine_deterministic_lam0p5_ann_arbor_robust_sigma03_seed42_e50"),
    ("convnext_affine_unc_decoupled", "ConvNeXt affine uncertainty-decoupled", "json", "week7_runs/week7_convnext_affine_deterministic_lam0p5_ann_arbor_robust_sigma03_seed7_e50"),
    ("convnext_affine_unc_decoupled", "ConvNeXt affine uncertainty-decoupled", "json", "week7_runs/week7_convnext_affine_deterministic_lam0p5_ann_arbor_robust_sigma03_seed123_e50"),
    ("convnext_affine_no_warp", "ConvNeXt affine no warp RGB loss", "json", "week7_runs/week7_convnext_affine_no_warprgb_ann_arbor_robust_sigma03_seed42_e50"),
    ("convnext_affine_no_edge", "ConvNeXt affine no edge loss", "json", "week7_runs/week7_convnext_affine_no_edge_lam0p5_ann_arbor_robust_sigma03_seed42_e50"),
    ("convnext_affine_no_ssim", "ConvNeXt affine no SSIM loss", "json", "week7_runs/week7_convnext_affine_no_ssim_lam0p5_ann_arbor_robust_sigma03_seed42_e50"),
    ("convnext_affine_no_affreg", "ConvNeXt affine no affine reg", "json", "week7_runs/week7_convnext_affine_no_affreg_lam0p5_ann_arbor_robust_sigma03_seed42_e50"),
    # Severity curve. Sigma 0.3 reuses the seed-42 audit rows above.
    ("severity_no_reg", "Severity no-reg sigma 0.0", "json", "week7_runs/week7_severity_no_registration_ann_arbor_robust_sigma0p0_seed42_e50"),
    ("severity_no_reg", "Severity no-reg sigma 0.1", "json", "week7_runs/week7_severity_no_registration_ann_arbor_robust_sigma0p1_seed42_e50"),
    ("severity_no_reg", "Severity no-reg sigma 0.2", "json", "week7_runs/week7_severity_no_registration_ann_arbor_robust_sigma0p2_seed42_e50"),
    ("severity_no_reg", "Severity no-reg sigma 0.5", "json", "week7_runs/week7_severity_no_registration_ann_arbor_robust_sigma0p5_seed42_e50"),
    ("severity_no_reg_unc_decoupled", "Severity no-reg uncertainty-decoupled sigma 0.0", "json", "week7_runs/week7_severity_no_registration_uncdec_ann_arbor_robust_sigma0p0_seed42_e50"),
    ("severity_no_reg_unc_decoupled", "Severity no-reg uncertainty-decoupled sigma 0.1", "json", "week7_runs/week7_severity_no_registration_uncdec_ann_arbor_robust_sigma0p1_seed42_e50"),
    ("severity_no_reg_unc_decoupled", "Severity no-reg uncertainty-decoupled sigma 0.2", "json", "week7_runs/week7_severity_no_registration_uncdec_ann_arbor_robust_sigma0p2_seed42_e50"),
    ("severity_no_reg_unc_decoupled", "Severity no-reg uncertainty-decoupled sigma 0.3", "json", "week7_runs/week7_severity_no_registration_uncdec_ann_arbor_robust_sigma0p3_seed42_e50"),
    ("severity_no_reg_unc_decoupled", "Severity no-reg uncertainty-decoupled sigma 0.5", "json", "week7_runs/week7_severity_no_registration_uncdec_ann_arbor_robust_sigma0p5_seed42_e50"),
    ("severity_affine_unc_decoupled", "Severity affine uncertainty-decoupled sigma 0.0", "json", "week7_runs/week7_severity_affine_deterministic_lam0p5_ann_arbor_robust_sigma0p0_seed42_e50"),
    ("severity_affine_unc_decoupled", "Severity affine uncertainty-decoupled sigma 0.1", "json", "week7_runs/week7_severity_affine_deterministic_lam0p5_ann_arbor_robust_sigma0p1_seed42_e50"),
    ("severity_affine_unc_decoupled", "Severity affine uncertainty-decoupled sigma 0.2", "json", "week7_runs/week7_severity_affine_deterministic_lam0p5_ann_arbor_robust_sigma0p2_seed42_e50"),
    ("severity_affine_unc_decoupled", "Severity affine uncertainty-decoupled sigma 0.3", "json", "week7_runs/week7_convnext_affine_deterministic_lam0p5_ann_arbor_robust_sigma03_seed42_e50"),
    ("severity_affine_unc_decoupled", "Severity affine uncertainty-decoupled sigma 0.5", "json", "week7_runs/week7_severity_affine_deterministic_lam0p5_ann_arbor_robust_sigma0p5_seed42_e50"),
]


FIELDNAMES = [
    "family",
    "label",
    "run",
    "kind",
    "dataset",
    "eval_dataset",
    "arch",
    "seed",
    "target_normalization",
    "loss_recipe",
    "disable_uncertainty_weight",
    "train_sigma",
    "eval_sigma",
    "lambda_warp_rgb",
    "lambda_ssim",
    "lambda_edge",
    "lambda_affine",
    "lambda_uncertainty",
    "lambda_uncertainty_tv",
    "epochs",
    "train_count",
    "val_count",
    "final_mae",
    "final_rmse",
    "final_psnr",
    "final_ssim",
    "final_corr",
    "final_theta_l2",
    "final_uncertainty",
    "final_warp_rgb_mae",
    "path",
]


def _last_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No metric rows in {path}")
    return rows[-1]


def _float(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def read_json_run(base: Path, family: str, label: str, rel: str) -> dict[str, object]:
    run_dir = base / rel
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    final = metrics["history"][-1]
    return {
        "family": family,
        "label": label,
        "run": run_dir.name,
        "kind": "json",
        "dataset": metadata.get("dataset", ""),
        "eval_dataset": metadata.get("eval_dataset") or metadata.get("dataset", ""),
        "arch": metadata.get("arch", ""),
        "seed": int(metadata.get("seed", 0)),
        "target_normalization": metadata.get("target_normalization", ""),
        "loss_recipe": metadata.get("loss_recipe", "registration"),
        "disable_uncertainty_weight": bool(metadata.get("disable_uncertainty_weight", False)),
        "train_sigma": _float(metadata.get("train_sigma")),
        "eval_sigma": _float(metadata.get("eval_sigma")),
        "lambda_warp_rgb": _float(metadata.get("lambda_warp_rgb")),
        "lambda_ssim": _float(metadata.get("lambda_ssim")),
        "lambda_edge": _float(metadata.get("lambda_edge")),
        "lambda_affine": _float(metadata.get("lambda_affine")),
        "lambda_uncertainty": _float(metadata.get("lambda_uncertainty")),
        "lambda_uncertainty_tv": _float(metadata.get("lambda_uncertainty_tv")),
        "epochs": int(metadata.get("epochs", 0)),
        "train_count": int(metadata.get("train_size", 0)),
        "val_count": int(metadata.get("val_size", 0)),
        "final_mae": _float(final.get("val_mae")),
        "final_rmse": _float(final.get("val_rmse")),
        "final_psnr": _float(final.get("val_psnr")),
        "final_ssim": _float(final.get("val_ssim")),
        "final_corr": _float(final.get("val_corr")),
        "final_theta_l2": _float(final.get("val_theta_l2")),
        "final_uncertainty": _float(final.get("val_uncertainty")),
        "final_warp_rgb_mae": _float(final.get("val_warp_rgb_mae")),
        "path": rel,
    }


def read_swin_run(base: Path, family: str, label: str, rel: str) -> dict[str, object]:
    run_dir = base / rel
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    final = _last_csv_row(run_dir / "metrics.csv")
    arch = metadata.get("arch") or "no_registration"
    return {
        "family": family,
        "label": label,
        "run": run_dir.name,
        "kind": "swin_csv",
        "dataset": metadata.get("dataset", ""),
        "eval_dataset": metadata.get("dataset", ""),
        "arch": arch,
        "seed": int(metadata.get("seed", 0)),
        "target_normalization": metadata.get("target_normalization", ""),
        "loss_recipe": "swin_combined",
        "disable_uncertainty_weight": "",
        "train_sigma": _float(metadata.get("train_sigma")),
        "eval_sigma": _float(metadata.get("eval_sigma")),
        "lambda_warp_rgb": _float(metadata.get("lambda_warp_rgb")),
        "lambda_ssim": "",
        "lambda_edge": "",
        "lambda_affine": _float(metadata.get("lambda_affine")),
        "lambda_uncertainty": "",
        "lambda_uncertainty_tv": "",
        "epochs": int(metadata.get("epochs", 0)),
        "train_count": int(metadata.get("train_size", 0)),
        "val_count": int(metadata.get("eval_size", 0)),
        "final_mae": _float(final.get("mae")),
        "final_rmse": _float(final.get("rmse")),
        "final_psnr": _float(final.get("psnr")),
        "final_ssim": _float(final.get("ssim")),
        "final_corr": _float(final.get("corr")),
        "final_theta_l2": _float(final.get("theta_l2")),
        "final_uncertainty": "",
        "final_warp_rgb_mae": _float(final.get("warp_rgb_mae")),
        "path": rel,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = Path(args.base)
    rows = []
    for family, label, kind, rel in RUNS:
        if kind == "json":
            rows.append(read_json_run(base, family, label, rel))
        elif kind == "swin_csv":
            rows.append(read_swin_run(base, family, label, rel))
        else:
            raise ValueError(kind)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
