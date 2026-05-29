#!/usr/bin/env python3
"""生成 Nature 风格的 SHAP 总览拼图。

这个脚本本身不负责计算 SHAP，也不负责挑选样本；它只负责读取外部提供的
图片资源，然后把它们按版式拼成一张适合论文展示的合成图。

输入说明
--------
通过 ``--config`` 传入一个 JSON 文件，里面需要显式给出每一张图片的路径。
脚本不会在代码里硬编码超声图、beeswarm 图或局部 SHAP 子图的位置。

JSON 结构示意
--------------
{
  "figure": {
    "title": "Nature-style draft: ...",
    "footer_text": "Top row: ...",
    "figsize": [20, 15.5],
    "output_png": "out/nature_shap_draft.png",
    "top_ratio": 1.28,
    "sample_ratio": 1.0,
    "sample_cols": 3
  },
  "layout": {
    "left": 0.03,
    "right": 0.993,
    "top": 0.962,
    "bottom": 0.055,
    "wspace": 0.10,
    "hspace": 0.14
  },
  "beeswarm_panels": [
    {
      "panel_label": "a",
      "title": "...",
      "image_path": "path/to/beeswarm.png"
    }
  ],
  "sample_panels": [
    {
      "panel_label": "d",
      "title": "...",
      "image_title": "Real ultrasound sample",
      "image_caption": "optional caption",
      "ultrasound_path": "path/to/ultrasound.png",
      "compact_shap_path": "path/to/compact_shap_bar.png"
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch

plt = None
np = None

DEFAULT_FIGURE_TITLE = "Nature-style draft: SHAP interpretation across three thyroid ultrasound binary tasks"
DEFAULT_FOOTER_TEXT = (
    "Top row: global SHAP beeswarm summaries. Middle and bottom rows: real ultrasound samples "
    "paired with compact local SHAP bars."
)
DEFAULT_OUTPUT_PNG = Path("out") / "nature_shap_draft.png"
DEFAULT_OUTPUT_PDF = Path("out") / "nature_shap_draft.pdf"
DEFAULT_FIGSIZE = (20, 15.5)
DEFAULT_TOP_RATIO = 1.28
DEFAULT_SAMPLE_RATIO = 1.0
DEFAULT_SAMPLE_COLS = 3


def _require_numpy():
    global np
    if np is None:
        import numpy as _np

        np = _np
    return np


def _require_matplotlib_pyplot():
    global plt
    if plt is None:
        import matplotlib.pyplot as _plt

        plt = _plt
    return plt


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Configuration JSON must contain an object at the top level.")
    return data


def _normalize_panel_list(
    base_dir: Path,
    panels: List[Dict[str, Any]],
    *,
    required_keys: Tuple[str, ...],
    path_keys: Tuple[str, ...],
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, panel in enumerate(panels, start=1):
        if not isinstance(panel, dict):
            raise ValueError(f"Panel #{idx} must be an object.")
        item = dict(panel)
        for key in required_keys:
            if key not in item or item[key] in (None, ""):
                raise ValueError(f"Panel #{idx} is missing required key: {key}")
        for key in path_keys:
            item[key] = str(_resolve_path(base_dir, item[key]))
        normalized.append(item)
    return normalized


def load_manifest(config_path: str | Path) -> Dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    base_dir = config_file.parent.parent if config_file.parent.name == "my_json" else config_file.parent
    raw = _load_json(config_file)

    figure_cfg = raw.get("figure", {})
    if figure_cfg is None:
        figure_cfg = {}
    if not isinstance(figure_cfg, dict):
        raise ValueError("'figure' must be an object when provided.")

    beeswarm_panels = raw.get("beeswarm_panels", [])
    sample_panels = raw.get("sample_panels", [])
    if not isinstance(beeswarm_panels, list):
        raise ValueError("'beeswarm_panels' must be a list.")
    if not isinstance(sample_panels, list):
        raise ValueError("'sample_panels' must be a list.")

    normalized = {
        "figure_title": figure_cfg.get("title", DEFAULT_FIGURE_TITLE),
        "footer_text": figure_cfg.get("footer_text", DEFAULT_FOOTER_TEXT),
        "figsize": tuple(figure_cfg.get("figsize", DEFAULT_FIGSIZE)),
        "output_png": str(_resolve_path(base_dir, figure_cfg.get("output_png", DEFAULT_OUTPUT_PNG))),
        "top_ratio": float(figure_cfg.get("top_ratio", DEFAULT_TOP_RATIO)),
        "sample_ratio": float(figure_cfg.get("sample_ratio", DEFAULT_SAMPLE_RATIO)),
        "sample_cols": int(raw.get("sample_cols", figure_cfg.get("sample_cols", DEFAULT_SAMPLE_COLS))),
        "layout": raw.get("layout", {}),
        "beeswarm_panels": _normalize_panel_list(
            base_dir,
            beeswarm_panels,
            required_keys=("image_path",),
            path_keys=("image_path",),
        ),
        "sample_panels": _normalize_panel_list(
            base_dir,
            sample_panels,
            required_keys=("ultrasound_path", "compact_shap_path"),
            path_keys=("ultrasound_path", "compact_shap_path"),
        ),
    }

    if normalized["sample_cols"] <= 0:
        raise ValueError("'sample_cols' must be a positive integer.")
    if len(normalized["beeswarm_panels"]) != normalized["sample_cols"]:
        raise ValueError(
            f"Expected exactly {normalized['sample_cols']} beeswarm panels, "
            f"but found {len(normalized['beeswarm_panels'])}."
        )
    if len(normalized["sample_panels"]) == 0:
        raise ValueError("'sample_panels' cannot be empty.")
    if len(normalized["sample_panels"]) % normalized["sample_cols"] != 0:
        raise ValueError(
            "The number of sample panels must be a multiple of 'sample_cols'."
        )

    return normalized


def _load_color_image(path: str | Path):
    plt_mod = _require_matplotlib_pyplot()
    return plt_mod.imread(path)


def _load_ultrasound_image(path: str | Path):
    np_mod = _require_numpy()
    plt_mod = _require_matplotlib_pyplot()

    image = plt_mod.imread(path)
    if image.ndim == 2:
        gray = image.astype(np_mod.float32)
    else:
        rgb = image[..., :3].astype(np_mod.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        gray = np_mod.dot(rgb, [0.299, 0.587, 0.114])
        gray = gray.astype(np_mod.float32)

    gray -= gray.min()
    max_value = gray.max()
    if max_value > 0:
        gray /= max_value
    return gray


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="black",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85),
        zorder=20,
    )


def _add_framed_axes(ax) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            boxstyle="round,pad=0.01",
            fill=False,
            linewidth=0.8,
            edgecolor="#d6d6d6",
            clip_on=False,
        )
    )


def _draw_beeswarm_panel(ax, panel: Dict[str, Any]) -> None:
    image = _load_color_image(panel["image_path"])
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)
    title = panel.get("title")
    if title:
        ax.set_title(title, pad=10, fontweight="semibold")
    label = panel.get("panel_label") or panel.get("label")
    if label:
        add_panel_label(ax, label)
    _add_framed_axes(ax)


def _draw_sample_panel(fig, spec, panel: Dict[str, Any]) -> None:
    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[1.08, 1.0], wspace=0.05)
    ax_img = fig.add_subplot(inner[0, 0])
    ax_shap = fig.add_subplot(inner[0, 1])

    ultrasound = _load_ultrasound_image(panel["ultrasound_path"])
    ax_img.imshow(ultrasound, cmap="gray", vmin=0, vmax=1)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for side in ["top", "right", "bottom", "left"]:
        ax_img.spines[side].set_visible(False)

    image_title = panel.get("image_title") or panel.get("title") or ""
    if image_title:
        ax_img.text(
            0.02,
            1.02,
            image_title,
            transform=ax_img.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="black",
            fontweight="semibold",
        )

    label = panel.get("panel_label") or panel.get("label")
    if label:
        add_panel_label(ax_img, label)

    image_caption = panel.get("image_caption")
    if image_caption:
        ax_img.text(
            0.03,
            0.05,
            image_caption,
            transform=ax_img.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="black", edgecolor="none", alpha=0.5),
        )

    shap_image = _load_color_image(panel["compact_shap_path"])
    ax_shap.imshow(shap_image)
    ax_shap.set_xticks([])
    ax_shap.set_yticks([])
    for side in ["top", "right", "bottom", "left"]:
        ax_shap.spines[side].set_visible(False)

    shap_title = panel.get("compact_shap_title")
    if shap_title:
        ax_shap.set_title(shap_title, pad=10, fontweight="semibold")

    _add_framed_axes(ax_img)
    _add_framed_axes(ax_shap)


def build_figure(config_path: str | Path) -> Tuple[Path, Path]:
    plt_mod = _require_matplotlib_pyplot()
    manifest = load_manifest(config_path)
    figure_cfg = manifest.get("figure") if isinstance(manifest.get("figure"), dict) else {}
    layout_cfg = manifest["layout"] if isinstance(manifest["layout"], dict) else {}

    beeswarm_panels = manifest["beeswarm_panels"]
    sample_panels = manifest["sample_panels"]
    sample_cols = manifest["sample_cols"]
    sample_rows = len(sample_panels) // sample_cols

    figsize = tuple(figure_cfg.get("figsize", manifest["figsize"]))
    height_ratios = [float(figure_cfg.get("top_ratio", manifest["top_ratio"]))] + [float(figure_cfg.get("sample_ratio", manifest["sample_ratio"]))] * sample_rows

    fig = plt_mod.figure(figsize=figsize, constrained_layout=False)
    fig.subplots_adjust(
        left=float(layout_cfg.get("left", 0.03)),
        right=float(layout_cfg.get("right", 0.993)),
        top=float(layout_cfg.get("top", 0.962)),
        bottom=float(layout_cfg.get("bottom", 0.055)),
    )

    gs = GridSpec(
        1 + sample_rows,
        sample_cols,
        figure=fig,
        height_ratios=height_ratios,
        wspace=float(layout_cfg.get("wspace", 0.10)),
        hspace=float(layout_cfg.get("hspace", 0.14)),
    )

    for col, panel in enumerate(beeswarm_panels):
        ax = fig.add_subplot(gs[0, col])
        _draw_beeswarm_panel(ax, panel)

    for idx, panel in enumerate(sample_panels):
        row = 1 + idx // sample_cols
        col = idx % sample_cols
        _draw_sample_panel(fig, gs[row, col], panel)

    footer_text = figure_cfg.get("footer_text", manifest["footer_text"])
    if footer_text:
        fig.text(
            0.5,
            0.018,
            footer_text,
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#333333",
        )

    figure_title = figure_cfg.get("title", manifest["figure_title"])
    if figure_title:
        fig.suptitle(
            figure_title,
            y=0.992,
            fontsize=15.5,
            fontweight="semibold",
        )

    output_png = Path(figure_cfg.get("output_png", manifest["output_png"]))
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    plt_mod.close(fig)
    return output_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Nature-style SHAP composite figure from a JSON manifest."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="JSON manifest that lists every beeswarm, ultrasound, and compact SHAP image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png = build_figure(args.config)
    print(f"Saved: {png}")


if __name__ == "__main__":
    main()
