import argparse

from plotting_utils import (
    load_shap_values,
    parse_training_csv_from_summary,
    reconstruct_X_explain,
    save_beeswarm_plot,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_txt", required=True, help="shap_analysis_summary.txt path")
    ap.add_argument("--shap_values_csv", required=True, help="*_shap_values.csv path (sample-level SHAP)")
    ap.add_argument("--label", default="label", help="label column name in training csv")
    ap.add_argument("--background_samples", type=int, default=100, help="must match SHAP run (default 100)")
    ap.add_argument("--explain_samples", type=int, default=500, help="must match SHAP run (default 500)")
    ap.add_argument("--max_display", type=int, default=20, help="Top-K features to show")
    ap.add_argument("--out_png", default="beeswarm.png", help="output png path")
    args = ap.parse_args()

    train_csv = parse_training_csv_from_summary(args.summary_txt)
    X_explain = reconstruct_X_explain(
        train_csv=train_csv,
        label_col=args.label,
        background_samples=args.background_samples,
        explain_samples=args.explain_samples,
    )

    shap_df = load_shap_values(args.shap_values_csv)

    # 确保列对齐：SHAP 列顺序为准
    missing = [c for c in shap_df.columns if c not in X_explain.columns]
    if missing:
        raise ValueError(f"Some SHAP columns are missing in X_explain: {missing[:10]} ...")

    X_explain = X_explain.loc[:, shap_df.columns]

    saved_paths = save_beeswarm_plot(
        shap_df.values,
        X_explain,
        args.out_png,
        args.max_display,
        export_formats=("png",),
        dpi=300,
        plot_type="dot",
        figsize=(8, 6),
    )
    print(f"Saved beeswarm to: {saved_paths[0]}")


if __name__ == "__main__":
    main()
