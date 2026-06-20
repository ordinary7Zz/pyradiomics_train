import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path("/Users/wangbd/sysu/pyradiomics_train/shap_analyze/analyze_single_image/analyze_single_image.py")


try:
    import pandas as pd  # noqa: F401
except ModuleNotFoundError:
    pandas_stub = types.ModuleType("pandas")
    pandas_stub.Series = object
    pandas_stub.DataFrame = object
    pandas_stub.isna = lambda value: False
    sys.modules["pandas"] = pandas_stub


spec = importlib.util.spec_from_file_location("analyze_single_image_module", MODULE_PATH)
analyze_single_image = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(analyze_single_image)


class ParseArgsBatchModeTest(unittest.TestCase):
    def test_filename_can_be_omitted_when_filename_list_is_given(self) -> None:
        argv = [
            "analyze_single_image.py",
            "--model_dir",
            "dummy_model_dir",
            "--train_csv",
            "dummy_train.csv",
            "--filename_list",
            "targets.txt",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = analyze_single_image.parse_args()

        self.assertIsNone(args.filename)
        self.assertEqual("targets.txt", args.filename_list)

    def test_validate_args_rejects_missing_both_filename_and_filename_list(self) -> None:
        argv = [
            "analyze_single_image.py",
            "--model_dir",
            "dummy_model_dir",
            "--train_csv",
            "dummy_train.csv",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = analyze_single_image.parse_args()

        with self.assertRaises(ValueError):
            analyze_single_image.validate_args(args)

    def test_validate_args_rejects_using_filename_and_filename_list_together(self) -> None:
        argv = [
            "analyze_single_image.py",
            "--model_dir",
            "dummy_model_dir",
            "--train_csv",
            "dummy_train.csv",
            "--filename",
            "case_001.jpg",
            "--filename_list",
            "targets.txt",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = analyze_single_image.parse_args()

        with self.assertRaises(ValueError):
            analyze_single_image.validate_args(args)


class FilenameListParsingTest(unittest.TestCase):
    def test_load_target_filenames_from_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            list_path = Path(tmpdir) / "targets.txt"
            list_path.write_text("\ncase_001.jpg\n case_002.png \n\ncase_003.jpeg\n", encoding="utf-8")

            self.assertEqual(
                ["case_001.jpg", "case_002.png", "case_003.jpeg"],
                analyze_single_image.load_target_filenames(str(list_path)),
            )

    def test_load_target_filenames_from_csv_uses_filename_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            list_path = Path(tmpdir) / "targets.csv"
            with list_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["filename", "note"])
                writer.writeheader()
                writer.writerow({"filename": "case_001.jpg", "note": "keep"})
                writer.writerow({"filename": "case_002.jpg", "note": "keep"})

            self.assertEqual(
                ["case_001.jpg", "case_002.jpg"],
                analyze_single_image.load_target_filenames(str(list_path)),
            )

    def test_load_target_filenames_deduplicates_while_preserving_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            list_path = Path(tmpdir) / "targets.txt"
            list_path.write_text("case_001.jpg\ncase_002.jpg\ncase_001.jpg\n", encoding="utf-8")

            self.assertEqual(
                ["case_001.jpg", "case_002.jpg"],
                analyze_single_image.load_target_filenames(str(list_path)),
            )


if __name__ == "__main__":
    unittest.main()
