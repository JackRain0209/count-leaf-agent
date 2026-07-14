import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main_module
from app.history import HistoryStore


class HistoryApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.original_store = main_module.history_store
        self.original_result_dir = main_module.RESULT_DIR
        self.original_upload_dir = main_module.UPLOAD_DIR
        self.original_analyzer = main_module.analyze_plant_image_stream
        main_module.history_store = HistoryStore(self.root / "history.sqlite3")
        main_module.RESULT_DIR = self.root / "results"
        main_module.RESULT_DIR.mkdir()
        main_module.UPLOAD_DIR = self.root / "uploads"
        main_module.UPLOAD_DIR.mkdir()
        self.client = TestClient(main_module.app)

        main_module.history_store.start("api-run", "sample.jpg", "uploads/sample.jpg")
        main_module.history_store.complete(
            "api-run",
            {
                "type": "result",
                "run_id": "api-run",
                "plant_count": 2,
                "plants": [
                    {"id": 1, "label": "主干", "pod_count": 1},
                    {"id": 2, "label": "分枝", "pod_count": 4},
                ],
                "summary": {},
            },
            [],
        )

    def tearDown(self):
        self.client.close()
        main_module.history_store = self.original_store
        main_module.RESULT_DIR = self.original_result_dir
        main_module.UPLOAD_DIR = self.original_upload_dir
        main_module.analyze_plant_image_stream = self.original_analyzer
        self.temp_dir.cleanup()

    def test_history_update_result_and_restore_endpoints(self):
        listing = self.client.get("/api/history")
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["count"], 1)

        update = self.client.put(
            "/api/history/api-run",
            json={
                "plant_count": 3,
                "plants": [
                    {"id": 1, "label": "主干", "pod_count": 1},
                    {"id": 2, "label": "分枝", "pod_count": 6},
                    {"id": 3, "label": "分枝", "pod_count": 7},
                ],
            },
        )
        self.assertEqual(update.status_code, 200)
        self.assertEqual(update.json()["result"]["total_pods"], 14)

        current = self.client.get("/api/results/api-run")
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["branch_count"], 2)
        self.assertTrue((self.root / "results" / "api-run" / "result.json").exists())

        restore = self.client.post("/api/history/api-run/restore")
        self.assertEqual(restore.status_code, 200)
        self.assertEqual(restore.json()["result"]["total_pods"], 5)

    def test_analyze_stream_persists_completed_history(self):
        def fake_analyzer(image_path, **kwargs):
            run_id = kwargs["run_id"]
            yield {
                "type": "step",
                "run_id": run_id,
                "step": 1,
                "url": f"/results/{run_id}/step_01.png",
                "description": "测试步骤",
            }
            yield {
                "type": "result",
                "run_id": run_id,
                "source_label": kwargs["source_label"],
                "plant_count": 2,
                "plants": [
                    {"id": 1, "label": "主干", "pod_count": 1},
                    {"id": 2, "label": "分枝", "pod_count": 9},
                ],
                "summary": {},
            }

        main_module.analyze_plant_image_stream = fake_analyzer
        response = self.client.post(
            "/api/analyze",
            files={"file": ("producer.jpg", b"not-a-real-image", "image/jpeg")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn('"type": "result"', response.text)
        listing = main_module.history_store.list(query="producer.jpg")
        self.assertEqual(listing["count"], 1)
        item = listing["items"][0]
        self.assertEqual(item["status"], "completed")
        self.assertEqual(item["total_pods"], 10)
        result_file = self.root / "results" / item["run_id"] / "result.json"
        self.assertTrue(result_file.exists())


if __name__ == "__main__":
    unittest.main()
