"""
Results Validation Script
Benchmarking Lightweight Neural Networks Against Deep CNNs for Melanoma Detection
Esther Jokodola et al.

This script checks all numbers reported in the paper against the raw 
per-run outputs. Any mismatch is flagged as an error.

Usage:
    python validate_results.py

All raw per-run values are entered below exactly as they appear in your logs.
Paper-reported values (mean ± std) are entered separately for comparison.
"""

import math

# ─────────────────────────────────────────────
# RAW PER-RUN VALUES FROM LOGS
# Enter exactly as they appear in your output logs
# ─────────────────────────────────────────────

raw = {
    "HAM10000": {
        "MobileNetV2": {
            "sensitivity":    [0.8760, 0.8380, 0.8840],
            "AUC":            [0.9531, 0.9487, 0.9534],
            "accuracy":       [0.8813, 0.9344, 0.9031],
            "f1_malignant":   [0.8795, 0.8793, 0.8902],
            "inference_time": [0.0189, 0.0199, 0.0196],
        },
        "ResNet50": {
            "sensitivity":    [0.9020, 0.9100, 0.8840],
            "AUC":            [0.9755, 0.9727, 0.9743],
            "accuracy":       [0.9500, 0.9187, 0.9500],
            "f1_malignant":   [0.9232, 0.9082, 0.9113],
            "inference_time": [0.0757, 0.0760, 0.0772],
        },
    },
    "ISIC2020": {
        "MobileNetV2": {
            "sensitivity":    [0.6496, 0.6154, 0.6838],
            "AUC":            [0.8141, 0.8288, 0.8448],
            "accuracy":       [0.7708, 0.8125, 0.8125],
            "f1_malignant":   [0.6972, 0.6990, 0.7373],
            "inference_time": [0.0813, 0.0767, 0.0829],
        },
        "ResNet50": {
            "sensitivity":    [0.6496, 0.4957, 0.6410],
            "AUC":            [0.8689, 0.8748, 0.8847],
            "accuracy":       [0.8125, 0.7292, 0.7500],
            "f1_malignant":   [0.7308, 0.6270, 0.7317],
            "inference_time": [0.1205, 0.1224, 0.1227],
        },
    },
    "Derm7pt": {
        "MobileNetV2": {
            "sensitivity":    [0.5149, 0.4554, 0.4653],
            "AUC":            [0.7291, 0.6927, 0.6656],
            "accuracy":       [0.7138, 0.6974, 0.6974],
            "f1_malignant":   [0.5098, 0.4742, 0.4608],
            "inference_time": [0.0177, 0.0166, 0.0201],
        },
        "ResNet50": {
            "sensitivity":    [0.4950, 0.6535, 0.5545],
            "AUC":            [0.8023, 0.8114, 0.7995],
            "accuracy":       [0.7928, 0.7862, 0.7697],
            "f1_malignant":   [0.5650, 0.6346, 0.5773],
            "inference_time": [0.0673, 0.0700, 0.0695],
        },
    },
}

# ─────────────────────────────────────────────
# PAPER-REPORTED VALUES (mean ± std from paper)
# These are what appear in Table 3 of the paper
# ─────────────────────────────────────────────

paper = {
    "HAM10000": {
        "MobileNetV2": {
            "sensitivity":    (0.8660, 0.0201),
            "AUC":            (0.9517, 0.0021),
            "accuracy":       (0.9062, 0.0218),
            "f1_malignant":   (0.8830, 0.0051),
            "inference_time": (0.0195, 0.0004),
        },
        "ResNet50": {
            "sensitivity":    (0.8987, 0.0109),
            "AUC":            (0.9742, 0.0012),
            "accuracy":       (0.9396, 0.0147),
            "f1_malignant":   (0.9143, 0.0065),
            "inference_time": (0.0763, 0.0007),
        },
    },
    "ISIC2020": {
        "MobileNetV2": {
            "sensitivity":    (0.6496, 0.0279),
            "AUC":            (0.8293, 0.0126),
            "accuracy":       (0.7986, 0.0196),
            "f1_malignant":   (0.7112, 0.0185),
            "inference_time": (0.0803, 0.0026),
        },
        "ResNet50": {
            "sensitivity":    (0.5954, 0.0706),
            "AUC":            (0.8761, 0.0065),
            "accuracy":       (0.7639, 0.0354),
            "f1_malignant":   (0.6965, 0.0491),
            "inference_time": (0.1218, 0.0010),
        },
    },
    "Derm7pt": {
        "MobileNetV2": {
            "sensitivity":    (0.4785, 0.0260),
            "AUC":            (0.6958, 0.0260),
            "accuracy":       (0.7029, 0.0078),
            "f1_malignant":   (0.4816, 0.0207),
            "inference_time": (0.0182, 0.0015),
        },
        "ResNet50": {
            "sensitivity":    (0.5677, 0.0653),
            "AUC":            (0.8044, 0.0051),
            "accuracy":       (0.7829, 0.0097),
            "f1_malignant":   (0.5923, 0.0303),
            "inference_time": (0.0689, 0.0012),
        },
    },
}

# ─────────────────────────────────────────────
# VALIDATION FUNCTIONS
# ─────────────────────────────────────────────

def compute_mean(values):
    return sum(values) / len(values)

def compute_std(values):
    m = compute_mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values))
    return math.sqrt(variance)

def check(dataset, model, metric, runs, reported_mean, reported_std, tolerance=0.0005):
    computed_mean = compute_mean(runs)
    computed_std = compute_std(runs)
    mean_ok = abs(computed_mean - reported_mean) <= tolerance
    std_ok  = abs(computed_std  - reported_std)  <= tolerance
    status = "✅ PASS" if (mean_ok and std_ok) else "❌ FAIL"
    if not (mean_ok and std_ok):
        print(f"\n{status} | {dataset} | {model} | {metric}")
        if not mean_ok:
            print(f"         Mean:  computed={computed_mean:.4f}  paper={reported_mean:.4f}  diff={abs(computed_mean-reported_mean):.4f}")
        if not std_ok:
            print(f"         Std:   computed={computed_std:.4f}   paper={reported_std:.4f}   diff={abs(computed_std-reported_std):.4f}")
    return status

# ─────────────────────────────────────────────
# RUN VALIDATION
# ─────────────────────────────────────────────

print("=" * 60)
print("RESULTS VALIDATION REPORT")
print("Jokodola et al. — Melanoma CNN Benchmarking Study")
print("=" * 60)

total = 0
passed = 0
failed = 0
failures = []

for dataset in raw:
    for model in raw[dataset]:
        for metric in raw[dataset][model]:
            runs = raw[dataset][model][metric]
            rep_mean, rep_std = paper[dataset][model][metric]
            result = check(dataset, model, metric, runs, rep_mean, rep_std)
            total += 1
            if "PASS" in result:
                passed += 1
            else:
                failed += 1
                failures.append(f"{dataset} | {model} | {metric}")

print("\n" + "=" * 60)
print(f"SUMMARY: {passed}/{total} checks passed")
if failed == 0:
    print("✅ ALL VALUES VERIFIED — paper numbers match raw log outputs")
else:
    print(f"❌ {failed} MISMATCHES FOUND:")
    for f in failures:
        print(f"   → {f}")
print("=" * 60)

# ─────────────────────────────────────────────
# ADDITIONAL SPOT CHECKS
# Key derived values cited in paper prose
# ─────────────────────────────────────────────

print("\nSPOT CHECKS — Derived values cited in paper")
print("-" * 60)

# Model size ratio
mv2_size = 16.74
rn50_size = 102.70
size_ratio = rn50_size / mv2_size
print(f"Model size ratio: {rn50_size}/{mv2_size} = {size_ratio:.2f}x  (paper says 6.1x) {'✅' if abs(size_ratio-6.1)<0.1 else '❌'}")

# AUC gap HAM10000
auc_gap_ham = 0.9742 - 0.9517
print(f"AUC gap HAM10000: {auc_gap_ham:.4f}  (paper says 0.0225) {'✅' if abs(auc_gap_ham-0.0225)<0.001 else '❌'}")

# AUC gap Derm7pt
auc_gap_d7 = 0.8044 - 0.6958
print(f"AUC gap Derm7pt:  {auc_gap_d7:.4f}  (paper says 0.1086) {'✅' if abs(auc_gap_d7-0.1086)<0.001 else '❌'}")

# AUC gap ratio
gap_ratio = auc_gap_d7 / auc_gap_ham
print(f"AUC gap ratio D7/HAM: {gap_ratio:.2f}x  (paper says ~4.8x) {'✅' if abs(gap_ratio-4.8)<0.15 else '❌'}")

# Inference speed ratios
inf_ham  = 0.0763 / 0.0195
inf_isic = 0.1218 / 0.0803
inf_d7   = 0.0689 / 0.0182
print(f"Inference ratio HAM10000: {inf_ham:.2f}x  (paper says 3.9x) {'✅' if abs(inf_ham-3.9)<0.1 else '❌'}")
print(f"Inference ratio ISIC2020: {inf_isic:.2f}x (paper says 1.5x) {'✅' if abs(inf_isic-1.5)<0.1 else '❌'}")
print(f"Inference ratio Derm7pt:  {inf_d7:.2f}x  (paper says 3.8x) {'✅' if abs(inf_d7-3.8)<0.1 else '❌'}")

# Training time ratios
train_ham  = 110 / 29
train_isic = 125 / 50
train_d7   = 18  / 7
print(f"Training ratio HAM10000:  {train_ham:.2f}x (paper says 3.8x) {'✅' if abs(train_ham-3.8)<0.1 else '❌'}")
print(f"Training ratio ISIC2020:  {train_isic:.2f}x (paper says 2.5x) {'✅' if abs(train_isic-2.5)<0.1 else '❌'}")
print(f"Training ratio Derm7pt:   {train_d7:.2f}x  (paper says 2.6x) {'✅' if abs(train_d7-2.6)<0.1 else '❌'}")

# FNR on ISIC2020
fnr = 1 - 0.6496
print(f"FNR ISIC2020 MobileNetV2: {fnr:.4f} (~35%) {'✅' if abs(fnr-0.35)<0.01 else '❌'}")

print("\n" + "=" * 60)
print("Validation complete.")
print("=" * 60)
