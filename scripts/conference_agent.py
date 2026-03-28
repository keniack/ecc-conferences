#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml

DATE_FIELDS = (
    "submission_deadline",
    "notification",
    "conference_start",
    "conference_end",
)
AUTO_UPDATE_FIELDS = DATE_FIELDS + ("location", "website")
KEYWORDS = (
    "call for papers",
    "call for paper",
    "important dates",
    "submission",
    "deadline",
    "extended deadline",
    "deadline extended",
    "new deadline",
    "final deadline",
    "hard deadline",
    "notification",
    "conference dates",
    "location",
    "venue",
)
EXTENSION_KEYWORDS = (
    "extended deadline",
    "deadline extended",
    "submission deadline extended",
    "deadline extension",
    "extended to",
    "new deadline",
    "final deadline",
    "hard deadline",
    "submission extended",
    "paper submission extended",
)
DEADLINE_CONTEXT_KEYWORDS = (
    "submission deadline",
    "important dates",
    "paper submission",
    "submission",
    "deadline",
)
MONTH_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DATE_CANDIDATE_PATTERNS = (
    re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{4}\b"),
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),
    re.compile(rf"\b{MONTH_PATTERN}\s+\d{{1,2}},\s+\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b\d{{1,2}}\s+{MONTH_PATTERN}\s+\d{{4}}\b", re.IGNORECASE),
)
DEFAULT_MODEL = "gpt-4.1-mini"
USER_AGENT = "ecc-conferences-agent/1.0 (+https://github.com/keniack/ecc-conferences)"
LLM_MAX_RETRIES = 2


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._chunks.append(data)

    def text(self) -> str:
        return " ".join(self._chunks)


@dataclass
class PageSnapshot:
    url: str
    final_url: str | None
    text: str
    ok: bool
    status_code: int | None = None
    error: str | None = None


@dataclass
class AnalysisResult:
    acronym: str
    status: str
    confidence: float
    reason: str
    selected_url: str | None
    changed_fields: list[str]
    updated_record: dict[str, str]
    review_note: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check conference pages and optionally update _data/conferences.yaml."
    )
    parser.add_argument(
        "--input",
        default="_data/conferences.yaml",
        help="Path to the conferences YAML file.",
    )
    parser.add_argument(
        "--report-file",
        help="Optional markdown file to write the run summary to.",
    )
    parser.add_argument(
        "--summary-file",
        help="Optional markdown file to write the same summary to.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N conferences.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=25,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.88,
        help="Minimum model confidence required for automatic updates.",
    )
    parser.add_argument(
        "--search-fallback",
        action="store_true",
        help="Search for replacement URLs when the current CFP page is missing or stale.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write approved updates back to the YAML file.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "conferences" not in data:
        raise ValueError(f"{path} does not contain a top-level 'conferences' key")
    if not isinstance(data["conferences"], list):
        raise ValueError(f"{path} has a non-list 'conferences' value")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            data,
            handle,
            sort_keys=False,
            allow_unicode=False,
            default_flow_style=False,
            width=1000,
        )


def collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def fetch_page(url: str, timeout: int) -> PageSnapshot:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
            html = raw.decode(charset, errors="replace")
            text = extract_text(html)
            status_code = getattr(response, "status", None)
            final_url = response.geturl()
            return PageSnapshot(
                url=url,
                final_url=final_url,
                text=text,
                ok=True,
                status_code=status_code,
            )
    except urllib.error.HTTPError as exc:
        return PageSnapshot(
            url=url,
            final_url=exc.geturl(),
            text="",
            ok=False,
            status_code=exc.code,
            error=f"HTTP {exc.code}",
        )
    except urllib.error.URLError as exc:
        return PageSnapshot(
            url=url,
            final_url=None,
            text="",
            ok=False,
            error=str(exc.reason),
        )
    except Exception as exc:
        return PageSnapshot(
            url=url,
            final_url=None,
            text="",
            ok=False,
            error=str(exc),
        )


def extract_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    return collapse_whitespace(unescape(parser.text()))


def build_excerpt(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    lowered = text.lower()
    windows: list[tuple[int, int]] = []
    for keyword in KEYWORDS:
        start = 0
        while True:
            position = lowered.find(keyword, start)
            if position == -1:
                break
            windows.append((max(0, position - 500), min(len(text), position + 2200)))
            if len(windows) >= 8:
                break
            start = position + len(keyword)
        if len(windows) >= 8:
            break

    if not windows:
        return text[:max_chars]

    merged: list[tuple[int, int]] = []
    for window_start, window_end in sorted(windows):
        if not merged or window_start > merged[-1][1]:
            merged.append((window_start, window_end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, window_end))

    parts: list[str] = []
    remaining = max_chars
    for window_start, window_end in merged:
        chunk = text[window_start:window_end]
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        parts.append(chunk)
        remaining -= len(chunk)
        if remaining <= 0:
            break
    excerpt = " ... ".join(parts)
    return excerpt[:max_chars]


def parse_search_results(html: str) -> list[str]:
    matches = re.findall(r'href="([^"]+)"', html)
    urls: list[str] = []
    for candidate in matches:
        parsed = urllib.parse.urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            urls.append(candidate)
            continue
        if parsed.path.startswith("/l/"):
            query = urllib.parse.parse_qs(parsed.query)
            uddg = query.get("uddg", [])
            if uddg:
                urls.append(urllib.parse.unquote(uddg[0]))
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


def search_candidate_urls(record: dict[str, str], timeout: int, limit: int = 3) -> list[str]:
    year_hint = (
        year_from_date(record.get("conference_start"))
        or year_from_date(record.get("submission_deadline"))
        or str(datetime.utcnow().year)
    )
    query = f'{record["acronym"]} {year_hint} "{record["name"]}" call for papers'
    search_url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(
        search_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            html = response.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    return parse_search_results(html)[:limit]


def year_from_date(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"(\d{4})$", value.strip())
    return match.group(1) if match else None


def valid_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urllib.parse.urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def normalize_date(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip()
    dot_match = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", normalized)
    if dot_match:
        day, month, year = dot_match.groups()
        return f"{int(day):02d}.{int(month):02d}.{year}"

    for fmt in (
        "%Y-%m-%d",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(normalized, fmt).strftime("%d.%m.%Y")
        except ValueError:
            continue
    return None


def llm_enabled() -> bool:
    return bool(get_env_value("OPENAI_API_KEY"))


def get_env_value(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped if stripped else default


def parse_normalized_date(value: str | None) -> datetime | None:
    normalized = normalize_date(value)
    if not normalized:
        return None
    try:
        return datetime.strptime(normalized, "%d.%m.%Y")
    except ValueError:
        return None


def date_text_variants(value: str | None) -> list[str]:
    date_value = parse_normalized_date(value)
    if not date_value:
        return []

    variants = [
        date_value.strftime("%d.%m.%Y"),
        f"{date_value.day}.{date_value.month}.{date_value.year}",
        date_value.strftime("%Y-%m-%d"),
        date_value.strftime("%d/%m/%Y"),
        f"{date_value.month}/{date_value.day}/{date_value.year}",
        f"{date_value.day}/{date_value.month}/{date_value.year}",
        f"{date_value.strftime('%B')} {date_value.day}, {date_value.year}",
        f"{date_value.strftime('%b')} {date_value.day}, {date_value.year}",
        f"{date_value.day} {date_value.strftime('%B')} {date_value.year}",
        f"{date_value.day} {date_value.strftime('%b')} {date_value.year}",
    ]
    return dedupe_preserve_order(variants)


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def collect_keyword_windows(
    text: str,
    keywords: tuple[str, ...],
    *,
    before: int = 260,
    after: int = 260,
    max_windows: int = 8,
) -> list[str]:
    lowered = text.lower()
    windows: list[tuple[int, int]] = []
    for keyword in keywords:
        start = 0
        while True:
            position = lowered.find(keyword, start)
            if position == -1:
                break
            windows.append(
                (max(0, position - before), min(len(text), position + len(keyword) + after))
            )
            if len(windows) >= max_windows:
                break
            start = position + len(keyword)
        if len(windows) >= max_windows:
            break

    if not windows:
        return []

    merged: list[tuple[int, int]] = []
    for window_start, window_end in sorted(windows):
        if not merged or window_start > merged[-1][1]:
            merged.append((window_start, window_end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, window_end))

    return [text[window_start:window_end] for window_start, window_end in merged]


def extract_date_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for pattern in DATE_CANDIDATE_PATTERNS:
        for match in pattern.findall(text):
            normalized = normalize_date(match)
            if normalized:
                candidates.append(normalized)
    return dedupe_preserve_order(candidates)


def detect_deadline_extension_signal(
    record: dict[str, str],
    snapshot: PageSnapshot,
) -> tuple[str | None, str | None]:
    if not snapshot.ok or not snapshot.text:
        return None, None

    current_deadline = normalize_date(record.get("submission_deadline"))
    current_deadline_dt = parse_normalized_date(current_deadline)
    if not current_deadline or not current_deadline_dt:
        return None, None

    windows = collect_keyword_windows(snapshot.text, EXTENSION_KEYWORDS, before=320, after=320)
    if not windows:
        return None, None

    later_candidates: list[str] = []
    for window in windows:
        for candidate in extract_date_candidates(window):
            candidate_dt = parse_normalized_date(candidate)
            if candidate_dt and candidate_dt > current_deadline_dt:
                later_candidates.append(candidate)

    later_candidates = dedupe_preserve_order(later_candidates)
    if later_candidates:
        latest_candidate = max(later_candidates, key=lambda value: parse_normalized_date(value))
        return (
            latest_candidate,
            f"Page mentions an extended or new deadline and shows a later date candidate ({latest_candidate}).",
        )

    return (
        None,
        "Page mentions an extended or new deadline, but no later replacement date was extracted confidently.",
    )


def page_mentions_current_deadline(record: dict[str, str], snapshot: PageSnapshot) -> bool:
    if not snapshot.ok or not snapshot.text:
        return False

    current_deadline = normalize_date(record.get("submission_deadline"))
    if not current_deadline:
        return False

    text_lower = snapshot.text.lower()
    for variant in date_text_variants(current_deadline):
        if variant.lower() in text_lower:
            return True

    windows = collect_keyword_windows(
        snapshot.text,
        DEADLINE_CONTEXT_KEYWORDS,
        before=260,
        after=260,
        max_windows=10,
    )
    for window in windows:
        if current_deadline in extract_date_candidates(window):
            return True

    return False


def rate_limit_backoff_seconds(exc: urllib.error.HTTPError, attempt: int) -> int:
    retry_after = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
    if retry_after:
        try:
            return max(1, min(int(retry_after), 30))
        except ValueError:
            pass
    return min(2**attempt, 30)


def parse_json_object(value: str) -> dict[str, Any]:
    stripped = value.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def analyze_with_llm(
    record: dict[str, str],
    snapshots: list[PageSnapshot],
    timeout: int,
) -> dict[str, Any]:
    pages = []
    for snapshot in snapshots:
        pages.append(
            {
                "url": snapshot.final_url or snapshot.url,
                "status_code": snapshot.status_code,
                "error": snapshot.error,
                "excerpt": build_excerpt(snapshot.text) if snapshot.text else "",
            }
        )

    payload = {
        "model": get_env_value("OPENAI_MODEL", DEFAULT_MODEL),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": textwrap.dedent(
                    """
                    You maintain a structured dataset of conference CFPs.
                    Read the current record and the provided page excerpts.
                    Only use facts that are explicitly present in the excerpts.

                    Return JSON with this shape:
                    {
                      "status": "unchanged" | "update" | "review",
                      "confidence": 0.0,
                      "reason": "short explanation",
                      "selected_url": "https://...",
                      "record": {
                        "submission_deadline": "DD.MM.YYYY",
                        "notification": "DD.MM.YYYY",
                        "conference_start": "DD.MM.YYYY",
                        "conference_end": "DD.MM.YYYY",
                        "location": "text",
                        "website": "https://..."
                      }
                    }

                    Rules:
                    - Prefer "review" instead of guessing.
                    - Use "update" only when the excerpt clearly describes the same conference edition.
                    - Keep fields unchanged unless the new value is explicit.
                    - If both an original deadline and an extended/new/final/hard deadline appear, use the currently active extended submission deadline.
                    - Do not confuse submission deadlines with abstract deadlines, workshop deadlines, camera-ready deadlines, or notification dates.
                    - Dates must use DD.MM.YYYY.
                    - selected_url should be the best CFP URL among the provided pages.
                    """
                ).strip(),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_record": record,
                        "pages": pages,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }

    api_key = get_env_value("OPENAI_API_KEY")
    base_url = get_env_value("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < LLM_MAX_RETRIES:
                time.sleep(rate_limit_backoff_seconds(exc, attempt + 1))
                continue
            raise
    body = json.loads(raw)
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return parse_json_object(content)


def heuristic_analysis(record: dict[str, str], snapshots: list[PageSnapshot]) -> dict[str, Any]:
    current = snapshots[0]
    if not current.ok:
        return {
            "status": "review",
            "confidence": 0.2,
            "reason": f'Current website is unreachable ({current.error or "request failed"}).',
            "selected_url": None,
            "record": {},
        }

    extended_deadline, extension_reason = detect_deadline_extension_signal(record, current)
    if extension_reason:
        return {
            "status": "review",
            "confidence": 0.74 if extended_deadline else 0.58,
            "reason": extension_reason,
            "selected_url": current.final_url or current.url,
            "record": {"submission_deadline": extended_deadline} if extended_deadline else {},
        }

    if page_mentions_current_deadline(record, current):
        return {
            "status": "unchanged",
            "confidence": 0.72,
            "reason": "Current submission deadline appears on the page.",
            "selected_url": current.final_url or current.url,
            "record": {},
        }

    text = current.text.lower()
    expected_year = year_from_date(record.get("conference_start")) or year_from_date(
        record.get("submission_deadline")
    )
    if expected_year and expected_year in text:
        return {
            "status": "review",
            "confidence": 0.5,
            "reason": "Page is reachable, but a structured update requires an API key.",
            "selected_url": current.final_url or current.url,
            "record": {},
        }

    return {
        "status": "review",
        "confidence": 0.35,
        "reason": "Page content looks stale or ambiguous.",
        "selected_url": current.final_url or current.url,
        "record": {},
    }


def sanitize_candidate_record(
    original: dict[str, str],
    proposed: dict[str, Any] | None,
    selected_url: str | None,
) -> tuple[dict[str, str], list[str]]:
    updated = dict(original)
    changed_fields: list[str] = []
    proposed = proposed or {}

    for field in AUTO_UPDATE_FIELDS:
        incoming = proposed.get(field)
        if field in DATE_FIELDS:
            normalized = normalize_date(str(incoming).strip()) if incoming else None
            if normalized and normalized != original.get(field):
                updated[field] = normalized
                changed_fields.append(field)
            continue

        if field == "website":
            candidate_url = selected_url or (str(incoming).strip() if incoming else None)
            if candidate_url and valid_url(candidate_url) and candidate_url != original.get(field):
                updated[field] = candidate_url
                changed_fields.append(field)
            continue

        if incoming:
            cleaned = collapse_whitespace(str(incoming))
            if cleaned and cleaned != original.get(field):
                updated[field] = cleaned
                changed_fields.append(field)

    return updated, changed_fields


def should_search_for_replacement(record: dict[str, str], snapshot: PageSnapshot) -> bool:
    if not snapshot.ok:
        return True
    if len(snapshot.text) < 500:
        return True
    expected_years = {
        year_from_date(record.get("conference_start")),
        year_from_date(record.get("submission_deadline")),
    }
    return not any(year and year in snapshot.text for year in expected_years)


def analyze_conference(
    record: dict[str, str],
    timeout: int,
    search_fallback: bool,
    min_confidence: float,
) -> AnalysisResult:
    primary_snapshot = fetch_page(record["website"], timeout)
    snapshots = [primary_snapshot]

    if search_fallback and should_search_for_replacement(record, primary_snapshot):
        for candidate_url in search_candidate_urls(record, timeout):
            if candidate_url == record["website"]:
                continue
            candidate_snapshot = fetch_page(candidate_url, timeout)
            snapshots.append(candidate_snapshot)

    heuristic = heuristic_analysis(record, snapshots)

    if llm_enabled() and heuristic.get("status") != "unchanged":
        try:
            analysis = analyze_with_llm(record, snapshots, timeout)
        except Exception as exc:
            analysis = dict(heuristic)
            heuristic_reason = collapse_whitespace(str(heuristic.get("reason", "")))
            prefix = (
                "LLM rate-limited; using heuristic result."
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 429
                else f"LLM analysis failed: {exc}. Using heuristic result."
            )
            analysis["reason"] = collapse_whitespace(f"{prefix} {heuristic_reason}")
            if not analysis.get("selected_url"):
                analysis["selected_url"] = primary_snapshot.final_url or primary_snapshot.url
    else:
        analysis = heuristic

    updated_record, changed_fields = sanitize_candidate_record(
        record,
        analysis.get("record"),
        analysis.get("selected_url"),
    )
    status = str(analysis.get("status", "review")).strip().lower()
    confidence = float(analysis.get("confidence", 0.0) or 0.0)
    reason = collapse_whitespace(str(analysis.get("reason", "No explanation provided.")))
    selected_url = analysis.get("selected_url")

    if status == "update" and confidence < min_confidence:
        review_note = (
            f"Model suggested an update at confidence {confidence:.2f}, below the configured threshold."
        )
        return AnalysisResult(
            acronym=record["acronym"],
            status="review",
            confidence=confidence,
            reason=reason,
            selected_url=selected_url,
            changed_fields=changed_fields,
            updated_record=record,
            review_note=review_note,
        )

    if status == "update" and not changed_fields:
        status = "unchanged"
        reason = "Model returned update, but no allowed fields changed."

    review_note = None
    if status == "review":
        review_note = reason

    return AnalysisResult(
        acronym=record["acronym"],
        status=status,
        confidence=confidence,
        reason=reason,
        selected_url=selected_url,
        changed_fields=changed_fields,
        updated_record=updated_record if status == "update" else record,
        review_note=review_note,
    )


def format_change_line(
    acronym: str,
    changed_fields: list[str],
    before: dict[str, str],
    after: dict[str, str],
) -> str:
    fragments = []
    for field in changed_fields:
        fragments.append(f"`{field}`: `{before[field]}` -> `{after[field]}`")
    return f"- `{acronym}`: " + ", ".join(fragments)


def build_report(
    processed: int,
    updated: list[tuple[dict[str, str], AnalysisResult]],
    reviews: list[AnalysisResult],
    unchanged: int,
    llm_mode: bool,
) -> str:
    lines = [
        "# Conference Agent",
        "",
        f"- Mode: {'auto-update' if llm_mode else 'check-only'}",
        f"- Processed conferences: {processed}",
        f"- Updated entries: {len(updated)}",
        f"- Needs review: {len(reviews)}",
        f"- Unchanged: {unchanged}",
        "",
    ]

    if updated:
        lines.append("## Applied Updates")
        lines.append("")
        for original, result in updated:
            lines.append(
                format_change_line(
                    result.acronym,
                    result.changed_fields,
                    original,
                    result.updated_record,
                )
            )
        lines.append("")

    if reviews:
        lines.append("## Needs Review")
        lines.append("")
        for result in reviews:
            detail = result.review_note or result.reason
            suffix = f" URL: {result.selected_url}" if result.selected_url else ""
            lines.append(
                f"- `{result.acronym}`: {detail} (confidence {result.confidence:.2f}).{suffix}"
            )
        lines.append("")

    if not llm_mode:
        lines.append("## Configuration")
        lines.append("")
        lines.append(
            "- Set the `OPENAI_API_KEY` secret to enable structured extraction and automatic updates."
        )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_report(path_value: str | None, content: str) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    data = read_yaml(input_path)
    conferences = data["conferences"]
    limit = args.limit or len(conferences)

    updated_results: list[tuple[dict[str, str], AnalysisResult]] = []
    review_results: list[AnalysisResult] = []
    unchanged_count = 0

    for index, conference in enumerate(conferences[:limit]):
        result = analyze_conference(
            conference,
            timeout=args.timeout,
            search_fallback=args.search_fallback,
            min_confidence=args.min_confidence,
        )
        if result.status == "update":
            original = dict(conference)
            conferences[index] = result.updated_record
            updated_results.append((original, result))
        elif result.status == "review":
            review_results.append(result)
        else:
            unchanged_count += 1

    if args.write and updated_results:
        write_yaml(input_path, data)

    report = build_report(
        processed=min(limit, len(conferences)),
        updated=updated_results,
        reviews=review_results,
        unchanged=unchanged_count,
        llm_mode=llm_enabled(),
    )
    write_report(args.report_file, report)
    write_report(args.summary_file, report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
