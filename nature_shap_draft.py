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
    "output_png": "out/nature_shap_draft.png",
    "image_sizes": {
      "ultrasound": [768, 768],
      "beeswarm": [1200, 800],
      "compact_shap_bar": [540, 768]
    }
  },
  "layout": {
    "dpi": 300,
    "page_margin_x_px": 40,
    "page_margin_top_px": 22,
    "page_margin_bottom_px": 22,
    "figure_title_band_px": 72,
    "figure_title_gap_px": 18,
    "footer_band_px": 52,
    "footer_gap_px": 18,
    "top_row_title_band_px": 56,
    "sample_row_title_band_px": 56,
    "col_gap_px": 28,
    "row_gap_px": 36,
    "section_gap_px": 44,
    "inner_panel_gap_px": 18
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
from typing import Any, Dict, List, Tuple

from matplotlib.patches import FancyBboxPatch
from PIL import Image

plt = None
np = None

DEFAULT_FIGURE_TITLE = "Nature-style draft: SHAP interpretation across three thyroid ultrasound binary tasks"
DEFAULT_FOOTER_TEXT = (
    "Top row: global SHAP beeswarm summaries. Middle and bottom rows: real ultrasound samples "
    "paired with compact local SHAP bars."
)
DEFAULT_OUTPUT_PNG = Path("out") / "nature_shap_draft.png"
DEFAULT_SAMPLE_COLS = 3
DEFAULT_IMAGE_SIZES = {
    "ultrasound": (768, 768),
    "beeswarm": (1200, 800),
    "compact_shap_bar": (540, 768),
}


def _layout_int(layout_cfg: Dict[str, Any], key: str, default: int) -> int:
    value = layout_cfg.get(key, default)
    if value in (None, ""):
        return default
    return int(value)


def _load_pixel_layout(layout_cfg: Dict[str, Any]) -> Dict[str, int]:
    return {
        "dpi": _layout_int(layout_cfg, "dpi", 300),
        "page_margin_x_px": _layout_int(layout_cfg, "page_margin_x_px", 40),
        "page_margin_top_px": _layout_int(layout_cfg, "page_margin_top_px", 22),
        "page_margin_bottom_px": _layout_int(layout_cfg, "page_margin_bottom_px", 22),
        "figure_title_band_px": _layout_int(layout_cfg, "figure_title_band_px", 72),
        "figure_title_gap_px": _layout_int(layout_cfg, "figure_title_gap_px", 18),
        "footer_band_px": _layout_int(layout_cfg, "footer_band_px", 52),
        "footer_gap_px": _layout_int(layout_cfg, "footer_gap_px", 18),
        "top_row_title_band_px": _layout_int(layout_cfg, "top_row_title_band_px", 56),
        "sample_row_title_band_px": _layout_int(layout_cfg, "sample_row_title_band_px", 56),
        "col_gap_px": _layout_int(layout_cfg, "col_gap_px", 28),
        "row_gap_px": _layout_int(layout_cfg, "row_gap_px", 36),
        "section_gap_px": _layout_int(layout_cfg, "section_gap_px", 44),
        "inner_panel_gap_px": _layout_int(layout_cfg, "inner_panel_gap_px", 18),
    }


def _px_rect_to_fig_rect(canvas_w_px: int, canvas_h_px: int, rect_px: Tuple[float, float, float, float]) -> list[float]:
    x_px, y_px, w_px, h_px = rect_px
    return [
        x_px / canvas_w_px,
        1.0 - ((y_px + h_px) / canvas_h_px),
        w_px / canvas_w_px,
        h_px / canvas_h_px,
    ]


def _add_axes_from_px(fig, canvas_w_px: int, canvas_h_px: int, rect_px: Tuple[float, float, float, float]):
    return fig.add_axes(_px_rect_to_fig_rect(canvas_w_px, canvas_h_px, rect_px))


def _add_fig_text_from_px(
    fig,
    canvas_w_px: int,
    canvas_h_px: int,
    x_px: float,
    y_px: float,
    text: str,
    **kwargs,
):
    return fig.text(x_px / canvas_w_px, y_px / canvas_h_px, text, **kwargs)


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


def _normalize_image_size(value: Any, *, key: str, default: Tuple[int, int]) -> Tuple[int, int]:
    size = default if value in (None, "") else value
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"'{key}' must be a two-item list like [width, height].")
    width = int(size[0])
    height = int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"'{key}' must contain positive integers.")
    return (width, height)


def _resize_to_size(image: Any, target_size: Tuple[int, int], *, grayscale: bool) -> Any:
    np_mod = _require_numpy()
    resample = getattr(Image, "Resampling", Image).LANCZOS
    width, height = target_size

    if grayscale:
        arr = np_mod.asarray(image)
        if arr.ndim == 3:
            rgb = arr[..., :3].astype(np_mod.float32)
            if rgb.max() > 1.0:
                rgb /= 255.0
            arr = np_mod.dot(rgb, [0.299, 0.587, 0.114]).astype(np_mod.float32)
        else:
            arr = arr.astype(np_mod.float32)

        arr -= arr.min()
        max_value = arr.max()
        if max_value > 0:
            arr /= max_value

        pil_image = Image.fromarray((arr * 255).round().astype(np_mod.uint8), mode="L")
        resized = pil_image.resize((width, height), resample=resample)
        return np_mod.asarray(resized).astype(np_mod.float32) / 255.0

    arr = np_mod.asarray(image)
    if arr.ndim == 2:
        if np_mod.issubdtype(arr.dtype, np_mod.floating):
            arr = np_mod.clip(arr, 0.0, 1.0)
            arr = (arr * 255).round().astype(np_mod.uint8)
        else:
            arr = np_mod.clip(arr, 0, 255).astype(np_mod.uint8)
        arr = np_mod.stack([arr, arr, arr], axis=-1)
    else:
        arr = arr[..., :3]
        if np_mod.issubdtype(arr.dtype, np_mod.floating):
            arr = np_mod.clip(arr, 0.0, 1.0)
            arr = (arr * 255).round().astype(np_mod.uint8)
        else:
            arr = np_mod.clip(arr, 0, 255).astype(np_mod.uint8)

    pil_image = Image.fromarray(arr, mode="RGB")
    resized = pil_image.resize((width, height), resample=resample)
    return np_mod.asarray(resized)


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

    image_sizes_cfg = figure_cfg.get("image_sizes", {})
    if image_sizes_cfg is None:
        image_sizes_cfg = {}
    if not isinstance(image_sizes_cfg, dict):
        raise ValueError("'image_sizes' must be an object when provided.")

    image_sizes = {
        name: _normalize_image_size(
            image_sizes_cfg.get(name),
            key=f"figure.image_sizes.{name}",
            default=default,
        )
        for name, default in DEFAULT_IMAGE_SIZES.items()
    }

    normalized = {
        "figure_title": figure_cfg.get("title", DEFAULT_FIGURE_TITLE),
        "footer_text": figure_cfg.get("footer_text", DEFAULT_FOOTER_TEXT),
        "output_png": str(_resolve_path(base_dir, figure_cfg.get("output_png", DEFAULT_OUTPUT_PNG))),
        "sample_cols": int(raw.get("sample_cols", figure_cfg.get("sample_cols", DEFAULT_SAMPLE_COLS))),
        "layout": raw.get("layout", {}),
        "image_sizes": image_sizes,
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


def _load_color_image(path: str | Path, target_size: Tuple[int, int] | None = None):
    plt_mod = _require_matplotlib_pyplot()
    image = plt_mod.imread(path)
    if target_size is not None:
        image = _resize_to_size(image, target_size, grayscale=False)
    return image


def _load_ultrasound_image(path: str | Path, target_size: Tuple[int, int] | None = None):
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
    if target_size is not None:
        gray = _resize_to_size(gray, target_size, grayscale=True)
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


def _draw_beeswarm_panel(ax, panel: Dict[str, Any], target_size: Tuple[int, int]) -> None:
    image = _load_color_image(panel["image_path"], target_size)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(False)
    title = panel.get("title")
    if title:
        ax.set_title(title, pad=4, fontweight="semibold")
    label = panel.get("panel_label") or panel.get("label")
    if label:
        add_panel_label(ax, label)
    _add_framed_axes(ax)


def _draw_sample_panel(
    ax_img,
    ax_shap,
    panel: Dict[str, Any],
    ultrasound_size: Tuple[int, int],
    compact_shap_bar_size: Tuple[int, int],
) -> None:
    ultrasound = _load_ultrasound_image(panel["ultrasound_path"], ultrasound_size)
    ax_img.imshow(ultrasound, cmap="gray", vmin=0, vmax=1)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for side in ["top", "right", "bottom", "left"]:
        ax_img.spines[side].set_visible(False)

    image_title = panel.get("image_title") or panel.get("title") or ""
    if image_title:
        ax_img.set_title(image_title, pad=10, fontweight="semibold", fontsize=9)

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

    shap_image = _load_color_image(panel["compact_shap_path"], compact_shap_bar_size)
    ax_shap.imshow(shap_image)
    ax_shap.set_xticks([])
    ax_shap.set_yticks([])
    for side in ["top", "right", "bottom", "left"]:
        ax_shap.spines[side].set_visible(False)

    shap_title = panel.get("compact_shap_title")
    if shap_title:
        ax_shap.set_title(shap_title, pad=10, fontweight="semibold", fontsize=9)

    _add_framed_axes(ax_img)
    _add_framed_axes(ax_shap)


def build_figure(config_path: str | Path) -> Path:
    plt_mod = _require_matplotlib_pyplot()
    manifest = load_manifest(config_path)
    figure_cfg = manifest.get("figure") if isinstance(manifest.get("figure"), dict) else {}
    layout_cfg = manifest["layout"] if isinstance(manifest["layout"], dict) else {}
    pixel_layout = _load_pixel_layout(layout_cfg)

    beeswarm_panels = manifest["beeswarm_panels"]
    sample_panels = manifest["sample_panels"]
    sample_cols = manifest["sample_cols"]
    image_sizes = manifest["image_sizes"]
    sample_rows = len(sample_panels) // sample_cols

    beeswarm_w, beeswarm_h = image_sizes["beeswarm"]
    ultrasound_w, ultrasound_h = image_sizes["ultrasound"]
    shap_w, shap_h = image_sizes["compact_shap_bar"]

    dpi = pixel_layout["dpi"]
    page_margin_x_px = pixel_layout["page_margin_x_px"]
    page_margin_top_px = pixel_layout["page_margin_top_px"]
    page_margin_bottom_px = pixel_layout["page_margin_bottom_px"]
    figure_title_band_px = pixel_layout["figure_title_band_px"]
    figure_title_gap_px = pixel_layout["figure_title_gap_px"]
    footer_band_px = pixel_layout["footer_band_px"]
    footer_gap_px = pixel_layout["footer_gap_px"]
    top_row_title_band_px = pixel_layout["top_row_title_band_px"]
    sample_row_title_band_px = pixel_layout["sample_row_title_band_px"]
    col_gap_px = pixel_layout["col_gap_px"]
    row_gap_px = pixel_layout["row_gap_px"]
    section_gap_px = pixel_layout["section_gap_px"]
    inner_panel_gap_px = pixel_layout["inner_panel_gap_px"]

    panel_content_h_px = max(ultrasound_h, shap_h)
    sample_block_w_px = ultrasound_w + inner_panel_gap_px + shap_w
    top_row_min_w_px = sample_cols * beeswarm_w
    sample_row_w_px = sample_cols * sample_block_w_px + (sample_cols - 1) * col_gap_px
    inner_w_px = max(top_row_min_w_px, sample_row_w_px)
    top_row_gap_px = 0 if sample_cols <= 1 else (inner_w_px - top_row_min_w_px) / (sample_cols - 1)

    top_row_cell_h_px = top_row_title_band_px + beeswarm_h
    sample_row_cell_h_px = sample_row_title_band_px + panel_content_h_px

    canvas_w_px = int(page_margin_x_px * 2 + inner_w_px)
    canvas_h_px = int(
        page_margin_top_px
        + figure_title_band_px
        + figure_title_gap_px
        + top_row_cell_h_px
        + section_gap_px
        + sample_rows * sample_row_cell_h_px
        + max(sample_rows - 1, 0) * row_gap_px
        + footer_gap_px
        + footer_band_px
        + page_margin_bottom_px
    )

    fig = plt_mod.figure(
        figsize=(canvas_w_px / dpi, canvas_h_px / dpi),
        dpi=dpi,
        constrained_layout=False,
        facecolor="white",
    )

    figure_title = figure_cfg.get("title", manifest["figure_title"])
    if figure_title:
        _add_fig_text_from_px(
            fig,
            canvas_w_px,
            canvas_h_px,
            canvas_w_px / 2,
            canvas_h_px - page_margin_top_px - figure_title_band_px / 2,
            figure_title,
            ha="center",
            va="center",
            fontsize=15.5,
            fontweight="semibold",
            color="black",
        )

    top_row_x_px = page_margin_x_px
    top_row_y_px = page_margin_top_px + figure_title_band_px + figure_title_gap_px
    for col, panel in enumerate(beeswarm_panels):
        cell_x_px = top_row_x_px + col * (beeswarm_w + top_row_gap_px)
        ax = _add_axes_from_px(
            fig,
            canvas_w_px,
            canvas_h_px,
            (cell_x_px, top_row_y_px + top_row_title_band_px, beeswarm_w, beeswarm_h),
        )
        _draw_beeswarm_panel(ax, panel, image_sizes["beeswarm"])

    sample_row_x_px = page_margin_x_px + (inner_w_px - sample_row_w_px) / 2
    sample_row_top_px = top_row_y_px + top_row_cell_h_px + section_gap_px
    for row_idx in range(sample_rows):
        row_y_px = sample_row_top_px + row_idx * (sample_row_cell_h_px + row_gap_px)
        row_panels = sample_panels[row_idx * sample_cols : (row_idx + 1) * sample_cols]
        for col_idx, panel in enumerate(row_panels):
            cell_x_px = sample_row_x_px + col_idx * (sample_block_w_px + col_gap_px)
            content_y_px = row_y_px + sample_row_title_band_px
            ax_img = _add_axes_from_px(
                fig,
                canvas_w_px,
                canvas_h_px,
                (cell_x_px, content_y_px, ultrasound_w, panel_content_h_px),
            )
            ax_shap = _add_axes_from_px(
                fig,
                canvas_w_px,
                canvas_h_px,
                (cell_x_px + ultrasound_w + inner_panel_gap_px, content_y_px, shap_w, panel_content_h_px),
            )
            _draw_sample_panel(ax_img, ax_shap, panel, image_sizes["ultrasound"], image_sizes["compact_shap_bar"])

    footer_text = figure_cfg.get("footer_text", manifest["footer_text"])
    if footer_text:
        _add_fig_text_from_px(
            fig,
            canvas_w_px,
            canvas_h_px,
            canvas_w_px / 2,
            page_margin_bottom_px + footer_band_px / 2,
            footer_text,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#333333",
        )

    output_png = Path(figure_cfg.get("output_png", manifest["output_png"]))
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_png, dpi=dpi, bbox_inches=None)
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
