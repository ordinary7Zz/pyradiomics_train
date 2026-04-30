import argparse
import glob
import os

from plotting_utils import (
    load_shap_values,
    paper_friendly_name,
    parse_training_csv_from_summary,
    reconstruct_X_explain,
    save_beeswarm_plot,
)


def plot_beeswarm(shap_df, x_explain, out_png: str, max_display: int):
    missing = [col for col in shap_df.columns if col not in x_explain.columns]
    if missing:
        raise ValueError(f"X_explain missing columns: {missing[:10]} ...")

    x_use = x_explain.loc[:, shap_df.columns].copy()
    return save_beeswarm_plot(
        shap_df.values,
        x_use,
        out_png,
        max_display,
        feature_name_formatter=paper_friendly_name,
        save_feature_name_map=True,
        export_formats=("png", "svg", "pdf"),
        dpi=300,
        figsize=(12, 8),
        plot_type="dot",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_txt", required=True, help="shap_analysis_summary.txt path")
    ap.add_argument("--shap_dir", required=True, help="directory containing *_shap_values.csv")
    ap.add_argument("--label", default="label", help="label column in training csv")
    ap.add_argument("--background_samples", type=int, default=100)
    ap.add_argument("--explain_samples", type=int, default=500)
    ap.add_argument("--max_display", type=int, default=20)
    ap.add_argument("--out_dir", default=None, help="output dir (default: <shap_dir>/plots)")
    ap.add_argument("--pattern", default="*_shap_values.csv", help="glob pattern for shap values files")
    args = ap.parse_args()

    train_csv = parse_training_csv_from_summary(args.summary_txt)
    X_explain = reconstruct_X_explain(
        train_csv=train_csv,
        label_col=args.label,
        background_samples=args.background_samples,
        explain_samples=args.explain_samples,
    )

    out_dir = args.out_dir or os.path.join(args.shap_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.shap_dir, args.pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched: {os.path.join(args.shap_dir, args.pattern)}")

    for f in files:
        name = os.path.basename(f).replace("_shap_values.csv", "")
        out_png = os.path.join(out_dir, f"{name}_beeswarm_top{args.max_display}.png")

        shap_df = load_shap_values(f)
        # 空文件/列异常跳过
        if shap_df.shape[1] == 0 or shap_df.shape[0] == 0:
            print(f"[SKIP] empty shap file: {f}")
            continue

        try:
            saved_paths = plot_beeswarm(shap_df, X_explain, out_png, args.max_display)
            print(f"[OK] {name} -> {', '.join(saved_paths)}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")

if __name__ == "__main__":
    main()
