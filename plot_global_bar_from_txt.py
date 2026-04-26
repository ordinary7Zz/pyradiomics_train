import re
import os
import argparse
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt


def parse_summary_txt(txt_path: str):
    """
    Parse 'SHAP Analysis Summary' txt like the one user posted.
    Returns:
      weights: dict[model_name] -> float
      model_top: dict[model_name] -> dict[feature] -> importance(float)
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    weights = {}
    model_top = {}

    # --- Parse ensemble weights block ---
    in_weights = False
    weight_pat = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*([0-9]*\.?[0-9]+)\s*$")
    for i, line in enumerate(lines):
        if line.strip().lower() == "ensemble weights:":
            in_weights = True
            continue
        if in_weights:
            if not line.strip():
                in_weights = False
                continue
            m = weight_pat.match(line)
            if m:
                weights[m.group(1)] = float(m.group(2))

    # --- Parse per-model top features ---
    i = 0
    model_header_pat = re.compile(r"^\s*Model:\s*(.+?)\s*$")
    feat_pat = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*([0-9]*\.?[0-9]+)\s*$")

    while i < len(lines):
        m = model_header_pat.match(lines[i])
        if not m:
            i += 1
            continue

        model_name = m.group(1).strip()
        model_top[model_name] = {}
        i += 1

        # move until "Top 10 features:" then parse subsequent feature lines
        while i < len(lines) and lines[i].strip().lower() != "top 10 features:":
            i += 1

        if i >= len(lines):
            break

        # now parse feature lines until blank or next "Model:"
        i += 1
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                break
            if model_header_pat.match(line):
                # next model starts
                i -= 1
                break

            fm = feat_pat.match(line)
            if fm:
                feat = fm.group(1)
                val = float(fm.group(2))
                model_top[model_name][feat] = val
            i += 1

        i += 1

    return weights, model_top


def compute_weighted_importance(weights: dict, model_top: dict):
    """
    Weighted sum over models: sum(w_m * importance_m(feature)).
    Only uses features present in txt (usually each model's top10).
    """
    weighted = defaultdict(float)
    used_models = 0

    for model_name, feats in model_top.items():
        if model_name not in weights:
            # If a model appears in txt but has no weight, skip it
            continue
        w = weights[model_name]
        used_models += 1
        for feat, val in feats.items():
            weighted[feat] += w * val

    if used_models == 0:
        raise RuntimeError(
            "No models matched between 'Ensemble weights' and 'Model:' sections. "
            "Check that the txt format/model names match."
        )

    # sort descending by importance
    weighted_sorted = OrderedDict(
        sorted(weighted.items(), key=lambda kv: kv[1], reverse=True)
    )
    return weighted_sorted


def plot_global_bar(weighted_sorted: OrderedDict, topk: int, out_png: str, out_pdf: str = None, title: str = None):
    feats = list(weighted_sorted.keys())[:topk]
    vals = [weighted_sorted[f] for f in feats]

    # barh: most important at top
    feats_rev = feats[::-1]
    vals_rev = vals[::-1]

    plt.figure(figsize=(10, max(4, 0.35 * len(feats_rev))))
    plt.barh(feats_rev, vals_rev)
    plt.xlabel("Weighted mean(|SHAP|)")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    if out_pdf:
        plt.savefig(out_pdf)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_txt", required=True, help="Path to SHAP Analysis Summary txt")
    ap.add_argument("--topk", type=int, default=20, help="Top-K features to show in Global Bar")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: same dir as txt)")
    ap.add_argument("--prefix", default="global_bar_weighted", help="Output filename prefix")
    args = ap.parse_args()

    txt_path = args.summary_txt
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(txt_path))
    os.makedirs(out_dir, exist_ok=True)

    weights, model_top = parse_summary_txt(txt_path)
    weighted_sorted = compute_weighted_importance(weights, model_top)

    out_png = os.path.join(out_dir, f"{args.prefix}_top{args.topk}.png")
    out_pdf = os.path.join(out_dir, f"{args.prefix}_top{args.topk}.pdf")

    title = f"Global Feature Importance (Weighted Ensemble) Top{args.topk}"
    plot_global_bar(weighted_sorted, args.topk, out_png, out_pdf, title=title)

    print("Saved:")
    print("  ", out_png)
    print("  ", out_pdf)

    # Also print topk to console for quick check
    print("\nTop features:")
    for i, (f, v) in enumerate(list(weighted_sorted.items())[:args.topk], start=1):
        print(f"{i:02d}. {f}: {v:.6f}")


if __name__ == "__main__":
    main()
