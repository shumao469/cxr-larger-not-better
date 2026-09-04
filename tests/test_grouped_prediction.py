"""Isolated CPU fixtures only. Never access formal images/checkpoints or CUDA."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
SPEC = importlib.util.spec_from_file_location("grouped_prediction", PROJECT / "src/19_predict_grouped_cxr_v2.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        self.head = nn.Linear(3, 4)

    def forward(self, images):
        return self.head(self.bn(images.mean(dim=(2, 3))))


class GroupedPredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cxr-grouped-unit-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = M.load_reference(PROJECT)
        pixels = np.arange(227 * 239 * 3, dtype=np.uint8).reshape(227, 239, 3)
        image = self.root / "image.png"
        Image.fromarray(pixels).save(image)
        rows = []
        for i, dataset in enumerate(("nih", "chexpert", "vindr")):
            row = dict(dataset=dataset, split="external_test" if dataset == "vindr" else "internal_test",
                       patient_id=f"p{i}", study_id=f"s{i}", image_id=f"i{i}", image_path_final=str(image))
            row.update({label: np.nan if i == 1 and n == 2 else float((i + n) % 2) for n, label in enumerate(M.LABELS)})
            rows.append(row)
        self.evaluation = pd.DataFrame(rows)
        self.meta = M.shared_metadata(self.evaluation)
        self.truth = self.evaluation[M.LABELS].to_numpy(dtype=np.float32)
        self.config = {"train_source": "nih", "subset_n": "1000", "seed": 1}
        self.spec = {"experiment": "nih_n1000_seed1", "checkpoint": self.root / "best.pt",
                     "checkpoint_sha256": "checkpoint-hash", "config": self.config}
        self.manifest_hash = "manifest-hash"
        self.digest = M.cohort_digest(self.meta, self.truth)
        self.probs = np.full((3, 4), 0.5, dtype=np.float32)
        self.audit = {"experiment": self.spec["experiment"], "checkpoint": str(self.spec["checkpoint"]),
                      "checkpoint_sha256": self.spec["checkpoint_sha256"], "prediction_rows": 3,
                      "counts": {"nih/internal_test": 1, "chexpert/internal_test": 1, "vindr/external_test": 1},
                      "labels": M.LABELS, "manifest_sha256": self.manifest_hash, "cohort_sha256": self.digest,
                      "precision": "float32", "batch_size": 128, "img_size": 224, "protocol": M.PROTOCOL,
                      "cpu_equivalence": {"status": "PASS"}, "runtime_equivalence": {"status": "PASS", "exact_probabilities": True},
                      "reference_script_sha256": "reference-hash"}

    def frame(self):
        return M.output_frame(self.meta, self.truth, self.probs, self.config)

    def stage(self):
        stage = self.root / "stage"
        M.prepare_prediction(stage, self.frame(), self.audit, self.spec, self.meta, self.truth)
        return stage

    def reusable(self, directory):
        return M.reusable_prediction(directory, self.spec, self.meta, self.truth,
                                     self.manifest_hash, self.digest, "reference-hash")

    def test_full_cpu_equivalence_self_test(self):
        with mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CPU self-test queried CUDA")):
            result = M.self_test(PROJECT)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["full_batch"], 128)
        self.assertEqual(result["tail_batch"], 1)

    def test_exact_filter_order_and_label_order(self):
        counts = {(a, b): 1 for a, b in zip(self.evaluation.dataset, self.evaluation.split)}
        result = M.build_eval_frame(self.evaluation.iloc[::-1], M.LABELS, expected_rows=3, expected_counts=counts)
        self.assertEqual(result.dataset.tolist(), ["nih", "chexpert", "vindr"])
        with self.assertRaisesRegex(RuntimeError, "labels/order"):
            M.build_eval_frame(self.evaluation, M.LABELS[::-1], expected_rows=3, expected_counts=counts)

    def test_duplicate_and_count_mismatch_rejected(self):
        frame = pd.concat([self.evaluation, self.evaluation.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            M.build_eval_frame(frame, M.LABELS, expected_rows=4)
        with self.assertRaisesRegex(RuntimeError, "counts"):
            M.build_eval_frame(self.evaluation, M.LABELS, expected_rows=3)

    def test_group_reuses_loader_once_and_matches_reference(self):
        models = {"one": Tiny().eval(), "two": Tiny().eval()}
        loader = list(M.make_loader(self.reference, self.evaluation, M.LABELS, batch_size=2))
        result = M.predict_group(models, iter(loader), self.meta, self.truth, device="cpu")
        for name, model in models.items():
            expected = M.reference_output(self.reference, model, loader, self.config)
            pd.testing.assert_frame_equal(M.output_frame(self.meta, self.truth, result[name], self.config), expected, check_exact=True)

    def test_training_model_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "eval mode"):
            M.predict_group({"model": Tiny()}, [], self.meta, self.truth, device="cpu")

    def test_metadata_order_mismatch_rejected(self):
        loader = M.make_loader(self.reference, self.evaluation.iloc[::-1], M.LABELS)
        with self.assertRaises(AssertionError):
            M.predict_group({"model": Tiny().eval()}, loader, self.meta, self.truth, device="cpu")

    def test_missing_tail_rejected(self):
        loader = M.make_loader(self.reference, self.evaluation.iloc[:2], M.LABELS)
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            M.predict_group({"model": Tiny().eval()}, loader, self.meta, self.truth, device="cpu")

    def test_nan_predictions_rejected(self):
        model = Tiny().eval()
        with torch.no_grad():
            model.head.weight.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "Invalid probabilities"):
            M.predict_group({"model": model}, M.make_loader(self.reference, self.evaluation, M.LABELS), self.meta, self.truth, device="cpu")

    def test_original_batch_boundary_selection(self):
        evaluation = pd.DataFrame({"dataset": ["nih"] * 21297 + ["chexpert"] * 28986 + ["vindr"] * 15000})
        indices = M.gate_sample_indices(evaluation)
        self.assertEqual(len(indices), 387)
        self.assertEqual(indices[:128], list(range(128)))
        for boundary in (21297, 50283):
            self.assertIn(boundary - 1, indices)
            self.assertIn(boundary, indices)
        self.assertEqual(indices[-3:], [65280, 65281, 65282])

    def test_runtime_cpu_gate_and_warm_cache_equal(self):
        from cxr_lossless_cache import LosslessResizeCache
        cache = LosslessResizeCache(self.root / "cache.sqlite3", max_bytes=30 * 1024**2,
                                   reserve_bytes=0, space_root=self.root)
        self.addCleanup(cache.close)
        result = M.runtime_equivalence(self.reference, {"tiny": Tiny().eval()}, self.evaluation,
                                       {"tiny": self.config}, device="cpu", cache=cache)
        self.assertEqual(result["status"], "PASS")
        self.assertGreater(cache.snapshot_stats()["hits"], 0)

    def test_cache_wrong_pixels_fail_runtime_gate(self):
        cache = mock.Mock()
        cache.load_rgb.return_value = Image.new("RGB", (224, 224))
        with self.assertRaisesRegex(RuntimeError, "Cached input tensor"):
            M.runtime_equivalence(self.reference, {"tiny": Tiny().eval()}, self.evaluation,
                                  {"tiny": self.config}, device="cpu", cache=cache)

    def test_create_only_roundtrip_and_completion_last(self):
        stage = self.stage()
        output = self.root / "published"
        linked = []
        original = M.os.link
        def observe(source, destination):
            linked.append(Path(destination).name)
            original(source, destination)
        with mock.patch.object(M.os, "link", side_effect=observe):
            M.publish_prediction(stage, output)
        self.assertEqual(linked, list(M.OUTPUT_NAMES))
        self.assertTrue(self.reusable(output))
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaisesRegex(RuntimeError, "Refusing"):
            M.publish_prediction(stage, output)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_bad_roundtrip_never_publishes(self):
        frame = self.frame()
        frame.loc[0, "image_id"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "metadata/order"):
            M.prepare_prediction(self.root / "stage", frame, self.audit, self.spec, self.meta, self.truth)
        self.assertFalse((self.root / "published").exists())

    def test_partial_existing_outputs_and_changed_hash_rejected(self):
        partial = self.root / "partial"
        partial.mkdir()
        (partial / "predictions.csv").write_text("preserve me", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "partial"):
            self.reusable(partial)
        stage = self.stage()
        with (stage / "predictions.csv").open("a") as handle:
            handle.write("tampering")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            self.reusable(stage)

    def test_failed_equivalence_cannot_be_reused(self):
        self.audit["runtime_equivalence"]["status"] = "FAIL"
        with self.assertRaisesRegex(RuntimeError, "equivalence"):
            self.stage()

    def test_no_readiness_no_gpu_lock(self):
        with mock.patch.object(M, "self_test", return_value={"status": "PASS"}), \
             mock.patch.object(M, "gpu_lock", side_effect=AssertionError("acquired GPU lock early")), \
             mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("queried GPU early")):
            with self.assertRaisesRegex(RuntimeError, "Not ready"):
                M.execute(self.root, 5, False)
        self.assertFalse((self.root / "results_formal_v2").exists())

    def test_cache_invalid_config_falls_back_without_cache_write(self):
        cache, info, hashes = M.open_image_cache(self.root, self.root / "cache.sqlite3")
        self.assertIsNone(cache)
        self.assertIn("fallback_to_original", info["reason"])
        self.assertEqual(hashes, {})
        self.assertFalse((self.root / "cache.sqlite3").exists())

    def test_final_fail_marker_rejected_before_gpu_lock(self):
        processed = self.root / "data/processed"
        processed.mkdir(parents=True)
        for name in ("model_manifest_formal_v2.csv", "model_manifest_formal_v2_audit.json",
                     "model_manifest_formal_v2_audit_overlap.csv"):
            (processed / name).touch()
        (processed / "model_manifest_formal_v2_finalization.json").write_text(json.dumps({"status": "FAIL"}))
        with self.assertRaisesRegex(RuntimeError, "before GPU lock"):
            M.readiness_counts(self.root)

    def test_wrong_training_identity_not_counted_complete(self):
        directory = self.root / "results_formal_v2/models/nih_n1000_seed1"
        directory.mkdir(parents=True)
        (directory / "completed.json").write_text(json.dumps({"status": "complete", "source": "chexpert",
                                                              "subset_n": "1000", "seed": 1, "epochs": 5}))
        with self.assertRaisesRegex(RuntimeError, "Invalid training completion"):
            M.readiness_counts(self.root)

    def test_changed_reference_cannot_reuse_existing_results(self):
        stage = self.stage()
        with self.assertRaisesRegex(RuntimeError, "reference version"):
            M.reusable_prediction(stage, self.spec, self.meta, self.truth, self.manifest_hash, self.digest, "new-reference")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
