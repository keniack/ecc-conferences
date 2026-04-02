import unittest

from scripts import conference_agent


class ConferenceAgentExtractionTests(unittest.TestCase):
    def test_submission_and_notification_stay_tied_to_their_labels(self) -> None:
        record = {
            "name": "IEEE Smart World Congress",
            "acronym": "SWC",
            "submission_deadline": "01.04.2026",
            "notification": "01.07.2026",
            "conference_start": "07.09.2026",
            "conference_end": "11.09.2026",
            "location": "Rende, Italy",
            "website": "https://swc-ieee-2026.github.io/swc/",
        }
        lines = [
            "Tutorial/WS/SS Proposal",
            "Feb 28, 2026 Closed",
            "Tutorial/WS/SS Acceptance",
            "Mar 7, 2026 Closed",
            "Full Paper Submission Extended",
            "May 1, 2026",
            "Authors Notification July 1, 2026",
            "Camera-ready July 31, 2026",
        ]
        snapshot = conference_agent.PageSnapshot(
            url=record["website"],
            final_url=record["website"],
            text=" ".join(lines),
            ok=True,
            status_code=200,
            lines=lines,
        )

        extracted = conference_agent.extract_structured_updates_from_snapshot(record, snapshot)

        self.assertIsNotNone(extracted)
        updates, _, _ = extracted
        self.assertEqual(updates.get("submission_deadline"), "01.05.2026")
        self.assertEqual(updates.get("notification"), "01.07.2026")

    def test_extension_signal_ignores_camera_ready_dates(self) -> None:
        record = {
            "name": "IEEE Smart World Congress",
            "acronym": "SWC",
            "submission_deadline": "01.04.2026",
            "notification": "01.07.2026",
            "conference_start": "07.09.2026",
            "conference_end": "11.09.2026",
            "location": "Rende, Italy",
            "website": "https://swc-ieee-2026.github.io/swc/",
        }
        lines = [
            "Full Paper Submission Extended",
            "May 1, 2026",
            "Authors Notification July 1, 2026",
            "Camera-ready July 31, 2026",
        ]
        snapshot = conference_agent.PageSnapshot(
            url=record["website"],
            final_url=record["website"],
            text=" ".join(lines),
            ok=True,
            status_code=200,
            lines=lines,
        )

        deadline, reason = conference_agent.detect_deadline_extension_signal(record, snapshot)

        self.assertEqual(deadline, "01.05.2026")
        self.assertEqual(reason, "Submission-labeled schedule context shows an extended or new deadline.")


if __name__ == "__main__":
    unittest.main()
