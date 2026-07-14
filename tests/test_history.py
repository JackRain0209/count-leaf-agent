import tempfile
import unittest
from pathlib import Path

from app.history import HistoryStore, apply_manual_edits, recalculate_result


def _sample_result():
    return {
        "type": "result",
        "run_id": "sample-run",
        "plant_count": 3,
        "plants": [
            {"id": 1, "label": "主干", "pod_count": 1, "pixel_bbox": [1, 2, 3, 4]},
            {"id": 2, "label": "主枝", "pod_count": 10},
            {"id": 3, "label": "分枝", "pod_count": 5},
        ],
        "summary": {},
    }


class ResultRevisionTests(unittest.TestCase):
    def test_recalculate_result_rebuilds_all_totals(self):
        result = recalculate_result(_sample_result())

        self.assertEqual(result["branch_count"], 1)
        self.assertEqual(result["main_inflorescence_pod_count"], 10)
        self.assertEqual(result["main_stem_plus_main_pod_count"], 11)
        self.assertEqual(result["branch_pod_count"], 5)
        self.assertEqual(result["total_pods"], 16)

    def test_manual_edits_preserve_metadata_and_allow_new_branch(self):
        result = apply_manual_edits(
            _sample_result(),
            {
                "plant_count": 4,
                "plants": [
                    {"id": 1, "label": "主干", "pod_count": 2},
                    {"id": 2, "label": "主枝", "pod_count": 11},
                    {"id": 3, "label": "分枝", "pod_count": 6},
                    {"id": 4, "label": "分枝", "pod_count": 7},
                ],
            },
        )

        self.assertEqual(result["plant_count"], 4)
        self.assertEqual(result["branch_count"], 2)
        self.assertEqual(result["total_pods"], 26)
        self.assertEqual(result["plants"][0]["pixel_bbox"], [1, 2, 3, 4])
        self.assertTrue(result["plants"][3]["manual"])

    def test_invalid_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "非负整数"):
            apply_manual_edits(
                _sample_result(),
                {"plant_count": 3, "plants": [{"id": 1, "label": "主干", "pod_count": -1}]},
            )


class HistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = HistoryStore(Path(self.temp_dir.name) / "history.sqlite3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_complete_update_and_restore_history(self):
        self.store.start("sample-run", "sample.jpg", "uploads/sample.jpg")
        completed = self.store.complete(
            "sample-run",
            _sample_result(),
            [{"type": "step", "step": 1, "url": "/results/sample-run/step_01.png"}],
        )
        self.assertEqual(completed["total_pods"], 16)

        listing = self.store.list()
        self.assertEqual(listing["count"], 1)
        self.assertEqual(listing["items"][0]["status"], "completed")

        updated = self.store.update(
            "sample-run",
            {
                "plant_count": 3,
                "plants": [
                    {"id": 1, "label": "主干", "pod_count": 0},
                    {"id": 2, "label": "主枝", "pod_count": 12},
                    {"id": 3, "label": "分枝", "pod_count": 8},
                ],
            },
        )
        self.assertEqual(updated["total_pods"], 20)
        self.assertTrue(self.store.get("sample-run")["is_modified"])

        restored = self.store.restore("sample-run")
        self.assertEqual(restored["total_pods"], 16)
        self.assertFalse(self.store.get("sample-run")["is_modified"])


if __name__ == "__main__":
    unittest.main()
