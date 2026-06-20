import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import pandas as pd  # noqa: F401
except ModuleNotFoundError:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.Series = object
    pandas_stub.DataFrame = object
    pandas_stub.isna = lambda value: False
    sys.modules["pandas"] = pandas_stub

from plots.plotting_utils import save_current_figure
from shap_analyze.shap_local_plots import save_compact_shap_bar_plot


class SaveCurrentFigurePaddingTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_save_current_figure_supports_pad_inches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            plt.figure(figsize=(2, 2))
            ax = plt.gca()
            ax.plot([0, 1], [0, 1])
            ax.text(-0.35, 0.5, "outside label", transform=ax.transAxes, ha="right", va="center", clip_on=False)

            no_pad_path = os.path.join(tmpdir, "tight_no_pad.png")
            padded_path = os.path.join(tmpdir, "tight_padded.png")

            save_current_figure(no_pad_path, export_formats=("png",), dpi=100, bbox_inches="tight", pad_inches=0.0)
            save_current_figure(padded_path, export_formats=("png",), dpi=100, bbox_inches="tight", pad_inches=0.4)

            with Image.open(no_pad_path) as no_pad_image:
                no_pad_size = no_pad_image.size
            with Image.open(padded_path) as padded_image:
                padded_size = padded_image.size

            self.assertGreater(padded_size[0], no_pad_size[0])
            self.assertGreater(padded_size[1], no_pad_size[1])


class CompactShapBarLegendTitleTest(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_compact_shap_bar_does_not_render_task_name_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "compact_plot.svg")
            task_name = "Benign vs Malignant"

            saved_paths = save_compact_shap_bar_plot(
                np.array([0.30, -0.28, 0.24, -0.22, -0.20]),
                [
                    "original_shape2D_Elongation",
                    "original_glrlm_RunLengthNonUniformity",
                    "original_shape2D_Sphericity",
                    "original_glrlm_GrayLevelNonUniformity",
                    "original_glrlm_ShortRunLowGrayLevelEmphasis",
                ],
                out_path,
                5,
                task_name=task_name,
                positive_class_name="malignant",
                negative_class_name="benign",
                output_space="raw score",
                export_formats=("svg",),
                figsize=(4.0, 4.8),
            )

            self.assertEqual([out_path], saved_paths)
            svg_text = Path(out_path).read_text(encoding="utf-8")
            self.assertNotIn(task_name, svg_text)


if __name__ == "__main__":
    unittest.main()
