import unittest
from unittest import mock

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

    def test_prepare_conference_fetches_linked_cfp_page(self) -> None:
        record = {
            "name": "International Conference on Service-Oriented System Engineering",
            "acronym": "SOSE",
            "submission_deadline": "01.04.2026",
            "notification": "20.05.2026",
            "conference_start": "27.07.2026",
            "conference_end": "30.07.2026",
            "location": "Fukuoka, Japan",
            "website": "https://cisose.fit.ac.jp/sose/",
        }
        cfp_url = "https://cisose.fit.ac.jp/sose/index.php/call-for-papers/"
        homepage_snapshot = conference_agent.PageSnapshot(
            url=record["website"],
            final_url=record["website"],
            text="IEEE SOSE 2026 Call for Papers",
            ok=True,
            status_code=200,
            links=[conference_agent.PageLink(url=cfp_url, text="Call for Papers")],
            lines=["IEEE SOSE 2026"],
        )
        cfp_snapshot = conference_agent.PageSnapshot(
            url=cfp_url,
            final_url=cfp_url,
            text=(
                "Call for Papers Important Dates "
                "Paper submission deadline: April 21, 2026 "
                "Author's notification: May 20, 2026 "
                "Conference dates: July 27-30, 2026"
            ),
            ok=True,
            status_code=200,
            lines=[
                "Call for Papers",
                "Important Dates",
                "Paper submission deadline: April 21, 2026",
                "Author's notification: May 20, 2026",
                "Conference dates: July 27-30, 2026",
            ],
        )

        with mock.patch.object(
            conference_agent,
            "fetch_page",
            side_effect=[homepage_snapshot, cfp_snapshot],
        ) as fetch_page:
            prepared = conference_agent.prepare_conference(
                record,
                timeout=25,
                search_fallback=False,
            )

        self.assertEqual(fetch_page.call_count, 2)
        self.assertEqual([snapshot.final_url for snapshot in prepared.snapshots], [record["website"], cfp_url])
        self.assertEqual(prepared.heuristic["status"], "review")
        self.assertEqual(prepared.heuristic["selected_url"], cfp_url)
        self.assertEqual(prepared.heuristic["record"]["submission_deadline"], "21.04.2026")


if __name__ == "__main__":
    unittest.main()
