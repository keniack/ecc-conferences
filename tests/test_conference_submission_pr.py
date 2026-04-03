import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SPEC = importlib.util.spec_from_file_location(
    "conference_submission_pr_under_test",
    SCRIPTS_DIR / "conference_submission_pr.py",
)
assert SPEC is not None and SPEC.loader is not None
conference_submission_pr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conference_submission_pr)


class ConferenceSubmissionPrTests(unittest.TestCase):
    def test_enrich_submission_record_passes_seed_record_to_snapshot_fallback(self) -> None:
        fields = {
            "conference_input": "TestConf",
            "website": "https://conf.example",
        }
        prepared = SimpleNamespace(
            snapshots=["primary", "secondary"],
            heuristic={
                "reason": "Heuristic fallback",
                "confidence": 0.4,
                "record": {},
                "selected_url": None,
            },
        )

        with (
            mock.patch.object(conference_submission_pr, "prepare_conference", return_value=prepared),
            mock.patch.object(
                conference_submission_pr,
                "preferred_public_snapshot_url",
                return_value="https://conf.example/cfp",
            ) as preferred_public_snapshot_url_mock,
            mock.patch.object(conference_submission_pr, "llm_enabled", return_value=False),
        ):
            record, details = conference_submission_pr.enrich_submission_record(fields, timeout=25)

        seed_record = conference_submission_pr.build_seed_record(fields)
        preferred_public_snapshot_url_mock.assert_called_once_with(seed_record, prepared.snapshots)
        self.assertEqual(details["seed_record"], seed_record)
        self.assertEqual(record["website"], "https://conf.example/cfp")


if __name__ == "__main__":
    unittest.main()
