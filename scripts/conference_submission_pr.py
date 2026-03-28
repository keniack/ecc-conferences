#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conference_agent import (
    build_pages_payload,
    collapse_whitespace,
    is_disallowed_conference_url,
    llm_enabled,
    normalize_date,
    parse_normalized_date,
    prepare_conference,
    preferred_public_snapshot_url,
    read_yaml,
    request_llm_completion,
    valid_url,
    write_yaml,
)

SUBMISSION_MARKER = "<!-- conference-submission -->"
FIELD_SPECS = (
    ("Conference", "conference_input"),
    ("Website", "website"),
)
ALLOWED_RANKINGS = {"A", "B", "C", "Unknown"}
DEFAULT_LOCATION = "TBD"
PLACEHOLDER_DAY_VALUES = {
    "submission_deadline": (1, 1),
    "notification": (1, 1),
    "conference_start": (2, 1),
    "conference_end": (3, 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a conference-submission PR from a GitHub issue event."
    )
    parser.add_argument(
        "--event-path",
        default=os.getenv("GITHUB_EVENT_PATH"),
        help="Path to the GitHub Actions event JSON payload.",
    )
    parser.add_argument(
        "--input",
        default="_data/conferences.yaml",
        help="Path to the conferences YAML file.",
    )
    parser.add_argument(
        "--report-file",
        help="Optional markdown file for the PR body.",
    )
    parser.add_argument(
        "--summary-file",
        help="Optional markdown file for the GitHub step summary.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=25,
        help="HTTP timeout in seconds for conference page checks.",
    )
    return parser.parse_args()


def read_event(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        raise ValueError("Missing --event-path and GITHUB_EVENT_PATH is not set.")
    path = Path(path_value)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_issue(event: dict[str, Any]) -> dict[str, Any]:
    issue = event.get("issue")
    if not isinstance(issue, dict):
        raise ValueError("GitHub event payload does not contain an issue object.")
    return issue


def extract_submission_fields(body: str) -> dict[str, str]:
    if SUBMISSION_MARKER not in body:
        raise ValueError("Issue body does not contain the conference submission marker.")

    normalized_body = body.replace("\r\n", "\n")
    fields: dict[str, str] = {}
    for label, key in FIELD_SPECS:
        match = re.search(rf"^- {re.escape(label)}:\s*(.+)$", normalized_body, re.MULTILINE)
        if not match:
            raise ValueError(f"Missing field in submission: {label}")
        fields[key] = collapse_whitespace(match.group(1))
    return fields


def validate_submission_fields(fields: dict[str, str], existing: list[dict[str, str]]) -> dict[str, str]:
    conference_input = collapse_whitespace(fields.get("conference_input", ""))
    website = collapse_whitespace(fields.get("website", ""))

    if not conference_input:
        raise ValueError("Conference name or acronym is required.")
    if not valid_url(website):
        raise ValueError("Website must be a valid public http(s) URL.")
    if is_disallowed_conference_url(website):
        raise ValueError("Website must not point to EasyChair, HotCRP, or EDAS.")

    normalized_website = website.lower()
    if any(item.get("website", "").strip().lower() == normalized_website for item in existing):
        raise ValueError("A conference with the same website already exists.")

    return {
        "conference_input": conference_input,
        "website": website,
    }


def placeholder_date(field: str) -> str:
    year = datetime.now(UTC).year
    day, month = PLACEHOLDER_DAY_VALUES[field]
    return f"{day:02d}.{month:02d}.{year}"


def looks_like_acronym(value: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]+", "", value)
    if not compact:
        return False
    if " " not in value and 2 <= len(compact) <= 24:
        return True
    letters = [character for character in compact if character.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
    return uppercase_ratio >= 0.75 and len(compact) <= 24


def derive_acronym(value: str) -> str:
    uppercase = "".join(character for character in value if character.isupper() or character.isdigit())
    if 2 <= len(uppercase) <= 16:
        return uppercase

    tokens = re.findall(r"[A-Za-z0-9]+", value)
    if tokens:
        initials = "".join(token[0].upper() for token in tokens[:10])
        if len(initials) >= 2:
            return initials

    compact = re.sub(r"[^A-Za-z0-9]+", "", value).upper()
    return compact[:16] or "UNKNOWN"


def infer_name_and_acronym(value: str) -> tuple[str, str]:
    normalized = collapse_whitespace(value)
    match = re.fullmatch(r"(.+?)\s+\(([A-Za-z0-9][A-Za-z0-9 .&/+_-]{1,30})\)", normalized)
    if match:
        name, acronym = match.groups()
        return collapse_whitespace(name), collapse_whitespace(acronym)

    if looks_like_acronym(normalized):
        acronym = re.sub(r"\s+", "", normalized).upper()
        return normalized, acronym

    return normalized, derive_acronym(normalized)


def build_seed_record(fields: dict[str, str]) -> dict[str, str]:
    name, acronym = infer_name_and_acronym(fields["conference_input"])
    return {
        "name": name,
        "acronym": acronym,
        "ranking": "Unknown",
        "submission_deadline": placeholder_date("submission_deadline"),
        "notification": placeholder_date("notification"),
        "conference_start": placeholder_date("conference_start"),
        "conference_end": placeholder_date("conference_end"),
        "location": DEFAULT_LOCATION,
        "website": fields["website"],
    }


def analyze_submission_with_llm(
    conference_input: str,
    seed_record: dict[str, str],
    prepared: Any,
    timeout: int,
) -> dict[str, Any]:
    response = request_llm_completion(
        {
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": textwrap.dedent(
                        """
                        You are enriching a conference tracker entry from a user submission.
                        Read the user-provided conference identifier and the supplied page excerpts.
                        Only use facts that are explicitly present in the excerpts.

                        Return JSON with this shape:
                        {
                          "confidence": 0.0,
                          "reason": "short explanation",
                          "selected_url": "https://...",
                          "record": {
                            "name": "Official conference name",
                            "acronym": "Common acronym",
                            "ranking": "A" | "B" | "C" | "Unknown",
                            "submission_deadline": "DD.MM.YYYY",
                            "notification": "DD.MM.YYYY",
                            "conference_start": "DD.MM.YYYY",
                            "conference_end": "DD.MM.YYYY",
                            "location": "City, Country",
                            "website": "https://..."
                          }
                        }

                        Rules:
                        - Prefer explicit facts from the page excerpts.
                        - Use empty strings for any field that is not explicit, except ranking which should default to Unknown.
                        - Do not invent rankings.
                        - If the page shows both an original and an extended submission deadline, use the currently active extended deadline.
                        - Do not confuse submission deadlines with abstract, workshop, or camera-ready deadlines.
                        - Dates must use DD.MM.YYYY.
                        - Do not use EasyChair, HotCRP, or EDAS URLs as selected_url or website.
                        - If the acronym is not explicit but the user submission clearly looks like an acronym, you may reuse it.
                        """
                    ).strip(),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "submitted_conference": conference_input,
                            "seed_record": seed_record,
                            "pages": build_pages_payload(prepared.snapshots),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        },
        timeout,
    )
    if not isinstance(response, dict):
        raise ValueError("LLM response was not a JSON object.")
    return response


def choose_public_url(candidates: list[str | None]) -> str | None:
    for candidate in candidates:
        if candidate and valid_url(candidate) and not is_disallowed_conference_url(candidate):
            return candidate
    return None


def sanitize_submission_record(
    seed_record: dict[str, str],
    proposed: dict[str, Any] | None,
    selected_url: str | None,
    fallback_url: str | None,
) -> dict[str, str]:
    proposed = proposed or {}
    updated = dict(seed_record)

    for field in ("name", "acronym", "location"):
        incoming = proposed.get(field)
        if incoming:
            updated[field] = collapse_whitespace(str(incoming))

    ranking = collapse_whitespace(str(proposed.get("ranking", "")))
    if ranking in ALLOWED_RANKINGS:
        updated["ranking"] = ranking

    for field in ("submission_deadline", "notification", "conference_start", "conference_end"):
        incoming = proposed.get(field)
        normalized = normalize_date(str(incoming).strip()) if incoming else None
        if normalized:
            updated[field] = normalized

    selected_public_url = choose_public_url(
        [
            selected_url,
            str(proposed.get("website")).strip() if proposed.get("website") else None,
            fallback_url,
            updated.get("website"),
        ]
    )
    if selected_public_url:
        updated["website"] = selected_public_url

    if updated.get("ranking") not in ALLOWED_RANKINGS:
        updated["ranking"] = "Unknown"

    if not updated.get("location"):
        updated["location"] = DEFAULT_LOCATION

    conference_start = parse_normalized_date(updated.get("conference_start"))
    conference_end = parse_normalized_date(updated.get("conference_end"))
    if conference_start and conference_end and conference_end < conference_start:
        updated["conference_end"] = updated["conference_start"]

    return updated


def enrich_submission_record(
    fields: dict[str, str],
    timeout: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    seed_record = build_seed_record(fields)
    prepared = prepare_conference(seed_record, timeout=timeout, search_fallback=True)
    fallback_url = preferred_public_snapshot_url(prepared.snapshots)
    details: dict[str, Any] = {
        "seed_record": seed_record,
        "reason": collapse_whitespace(str(prepared.heuristic.get("reason", ""))),
        "confidence": float(prepared.heuristic.get("confidence", 0.0) or 0.0),
        "used_llm": False,
    }

    if llm_enabled():
        try:
            response = analyze_submission_with_llm(
                fields["conference_input"],
                seed_record,
                prepared,
                timeout,
            )
            details["reason"] = collapse_whitespace(
                str(response.get("reason") or "Structured extraction completed from the submitted page.")
            )
            details["confidence"] = float(response.get("confidence", 0.0) or 0.0)
            details["used_llm"] = True
            record = sanitize_submission_record(
                seed_record,
                response.get("record"),
                response.get("selected_url"),
                fallback_url,
            )
            return record, details
        except Exception as exc:
            if isinstance(exc, urllib.error.HTTPError):
                details["reason"] = f"Structured enrichment unavailable ({exc}). Using placeholder defaults for missing fields."
            else:
                details["reason"] = f"Structured enrichment failed ({exc}). Using placeholder defaults for missing fields."

    record = sanitize_submission_record(
        seed_record,
        prepared.heuristic.get("record"),
        prepared.heuristic.get("selected_url"),
        fallback_url,
    )
    return record, details


def validate_final_record(record: dict[str, str], existing: list[dict[str, str]]) -> dict[str, str]:
    missing = [field for field in ("name", "acronym", "website") if not collapse_whitespace(record.get(field, ""))]
    if missing:
        raise ValueError(f"Submission is missing required fields after enrichment: {', '.join(missing)}")

    if not valid_url(record["website"]):
        raise ValueError("Final website must be a valid public http(s) URL.")
    if is_disallowed_conference_url(record["website"]):
        raise ValueError("Final website must not point to EasyChair, HotCRP, or EDAS.")

    acronym_key = collapse_whitespace(record["acronym"]).lower()
    if any(collapse_whitespace(item.get("acronym", "")).lower() == acronym_key for item in existing):
        raise ValueError(f'A conference with acronym "{record["acronym"]}" already exists.')

    website_key = record["website"].strip().lower()
    if any(item.get("website", "").strip().lower() == website_key for item in existing):
        raise ValueError("A conference with the same website already exists.")

    if record.get("ranking") not in ALLOWED_RANKINGS:
        record["ranking"] = "Unknown"

    conference_start = parse_normalized_date(record.get("conference_start"))
    conference_end = parse_normalized_date(record.get("conference_end"))
    if conference_start and conference_end and conference_end < conference_start:
        raise ValueError("Conference end date must be on or after the conference start date.")

    return record


def list_remaining_defaults(seed_record: dict[str, str], final_record: dict[str, str]) -> list[str]:
    fields = []
    for field in ("ranking", "submission_deadline", "notification", "conference_start", "conference_end", "location"):
        if final_record.get(field) == seed_record.get(field):
            fields.append(field)
    return fields


def build_report(
    issue: dict[str, Any],
    submitted_fields: dict[str, str],
    record: dict[str, str],
    enrichment_details: dict[str, Any],
) -> str:
    issue_number = issue.get("number")
    issue_url = issue.get("html_url")
    remaining_defaults = list_remaining_defaults(enrichment_details["seed_record"], record)

    lines = [
        f"Closes #{issue_number}",
        "",
        "Conference submission generated from the website form.",
        "",
        f'- Submitted conference: `{submitted_fields["conference_input"]}`',
        f'- Submitted website: `{submitted_fields["website"]}`',
        f'- Automation path: `{"llm-enriched" if enrichment_details.get("used_llm") else "fallback-defaults"}`',
        f'- Automation note: {enrichment_details.get("reason") or "No note provided."}',
        "",
        "Final record:",
        "",
        f'- `name`: `{record["name"]}`',
        f'- `acronym`: `{record["acronym"]}`',
        f'- `ranking`: `{record["ranking"]}`',
        f'- `submission_deadline`: `{record["submission_deadline"]}`',
        f'- `notification`: `{record["notification"]}`',
        f'- `conference_start`: `{record["conference_start"]}`',
        f'- `conference_end`: `{record["conference_end"]}`',
        f'- `location`: `{record["location"]}`',
        f'- `website`: `{record["website"]}`',
    ]

    if remaining_defaults:
        lines.extend(
            [
                "",
                "Fields still using defaults:",
                "",
                f'- `{", ".join(remaining_defaults)}`',
            ]
        )

    if issue_url:
        lines.extend(["", f"Source issue: {issue_url}"])

    return "\n".join(lines).strip() + "\n"


def write_report(path_value: str | None, content: str) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    event = read_event(args.event_path)
    issue = extract_issue(event)
    body = str(issue.get("body") or "")

    data = read_yaml(Path(args.input))
    conferences = data["conferences"]
    submitted_fields = validate_submission_fields(extract_submission_fields(body), conferences)
    record, enrichment_details = enrich_submission_record(submitted_fields, args.timeout)
    record = validate_final_record(record, conferences)

    conferences.append(record)
    write_yaml(Path(args.input), data)

    report = build_report(issue, submitted_fields, record, enrichment_details)
    write_report(args.report_file, report)
    write_report(args.summary_file, report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
