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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

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
SUBMISSION_LINE_LABELS = (
    "submission deadline",
    "paper submission deadline",
    "paper submission",
    "full paper submission",
    "submission due",
)
NOTIFICATION_LINE_LABELS = (
    "notification",
    "author notification",
    "authors notification",
    "acceptance notification",
)
CONFERENCE_DATE_LINE_LABELS = (
    "conference dates",
    "conference date",
    "event dates",
    "symposium dates",
)
LOCATION_LINE_LABELS = ("location", "venue")
DISALLOWED_LOCATION_PHRASES = (
    "accommodation",
    "call for papers",
    "committee",
    "committees",
    "contact",
    "getting there",
    "hotel reservation",
    "important dates",
    "past editions",
    "registration",
    "sponsor",
    "sponsorship",
    "travel grant",
    "travel grants",
    "venue & hotel reservation",
)
BLOCK_BREAK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}
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
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
USER_AGENT = "ecc-conferences-agent/1.0 (+https://github.com/keniack/ecc-conferences)"
LLM_MAX_RETRIES = 2
LLM_BATCH_SIZE = 5
LOCAL_TIMEZONE = ZoneInfo("Europe/Vienna")
RECENT_DEADLINE_WINDOW_DAYS = 10
LLM_MAX_REQUESTS_PER_MINUTE = 10
LLM_REQUEST_INTERVAL_SECONDS = 60 / LLM_MAX_REQUESTS_PER_MINUTE
DISALLOWED_WEBSITE_HOST_FRAGMENTS = ("easychair", "hotcrp", "edas")
_next_llm_request_at = 0.0


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


class LinkExtractor(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self._base_url = base_url
        self._ignored_depth = 0
        self._active_href: str | None = None
        self._active_chunks: list[str] = []
        self._links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth or tag != "a":
            return
        attributes = dict(attrs)
        href = attributes.get("href")
        if href:
            self._active_href = href
            self._active_chunks = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth or tag != "a" or not self._active_href:
            return
        resolved_url = urllib.parse.urljoin(self._base_url, self._active_href)
        resolved_url, _ = urllib.parse.urldefrag(resolved_url)
        self._links.append((resolved_url, collapse_whitespace(" ".join(self._active_chunks))))
        self._active_href = None
        self._active_chunks = []

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and self._active_href:
            self._active_chunks.append(data)

    def links(self) -> list[tuple[str, str]]:
        return self._links


class LineExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self._chunks: list[str] = []
        self._lines: list[str] = []

    def _flush_line(self) -> None:
        if not self._chunks:
            return
        line = collapse_whitespace(" ".join(self._chunks))
        if line:
            self._lines.append(line)
        self._chunks = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
            return
        if not self._ignored_depth and tag in BLOCK_BREAK_TAGS:
            self._flush_line()

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if not self._ignored_depth and tag in BLOCK_BREAK_TAGS:
            self._flush_line()

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._chunks.append(data)

    def lines(self) -> list[str]:
        self._flush_line()
        return self._lines


@dataclass
class PageLink:
    url: str
    text: str


@dataclass
class PageSnapshot:
    url: str
    final_url: str | None
    text: str
    ok: bool
    status_code: int | None = None
    error: str | None = None
    links: list[PageLink] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    acronym: str
    status: str
    confidence: float
    reason: str
    selected_url: str | None
    changed_fields: list[str]
    updated_record: dict[str, str]
    applied_fields: list[str] = field(default_factory=list)
    applied_record: dict[str, str] = field(default_factory=dict)
    review_note: str | None = None


@dataclass
class PreparedConference:
    index: int
    record: dict[str, str]
    snapshots: list[PageSnapshot]
    heuristic: dict[str, Any]


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
        help="Minimum LLM confidence required for automatic updates.",
    )
    parser.add_argument(
        "--heuristic-min-confidence",
        type=float,
        default=0.72,
        help="Minimum heuristic confidence required for automatic updates before using the LLM.",
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
            final_url = response.geturl()
            text = extract_text(html)
            links = extract_links(html, final_url)
            lines = extract_lines(html)
            status_code = getattr(response, "status", None)
            return PageSnapshot(
                url=url,
                final_url=final_url,
                text=text,
                ok=True,
                status_code=status_code,
                links=links,
                lines=lines,
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


def extract_lines(html: str) -> list[str]:
    parser = LineExtractor()
    parser.feed(html)
    return parser.lines()


def extract_links(html: str, base_url: str) -> list[PageLink]:
    parser = LinkExtractor(base_url)
    parser.feed(html)
    links: list[PageLink] = []
    seen: set[str] = set()
    for url, text in parser.links():
        if not valid_url(url) or url in seen:
            continue
        links.append(PageLink(url=url, text=text))
        seen.add(url)
    return links


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
        or str(datetime.now(UTC).year)
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
    return [
        url
        for url in parse_search_results(html)
        if not is_disallowed_conference_url(url)
        and not is_pdf_url(url)
        and not is_incompatible_edition_url(record, url)
    ][:limit]


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


def is_pdf_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urllib.parse.urlparse(value.strip())
    return parsed.path.lower().endswith(".pdf")


def is_disallowed_conference_url(value: str | None) -> bool:
    if not value:
        return False
    hostname = urllib.parse.urlparse(value.strip()).netloc.lower()
    return any(fragment in hostname for fragment in DISALLOWED_WEBSITE_HOST_FRAGMENTS)


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


def extract_ordered_date_candidates(text: str) -> list[str]:
    positioned_candidates: list[tuple[int, str]] = []
    for pattern in DATE_CANDIDATE_PATTERNS:
        for match in pattern.finditer(text):
            normalized = normalize_date(match.group(0))
            if normalized:
                positioned_candidates.append((match.start(), normalized))
    positioned_candidates.sort(key=lambda item: item[0])
    return dedupe_preserve_order([candidate for _, candidate in positioned_candidates])


def is_allowed_candidate_date(record: dict[str, str], candidate: str) -> bool:
    candidate_date = parse_normalized_date(candidate)
    reference_year = record_reference_year(record)
    if not candidate_date or reference_year is None:
        return bool(candidate_date)
    return candidate_date.year in {reference_year, reference_year + 1}


def contextual_line_windows(
    lines: list[str],
    labels: tuple[str, ...],
    *,
    lookahead: int = 2,
) -> list[str]:
    windows: list[str] = []
    for index, line in enumerate(lines):
        lowered = line.lower()
        if any(label in lowered for label in labels):
            windows.append(" ".join(lines[index : index + lookahead + 1]))
    return windows


def select_candidates(
    record: dict[str, str],
    text: str,
    *,
    prefer_latest: bool = False,
) -> str | None:
    candidates = [
        candidate
        for candidate in extract_ordered_date_candidates(text)
        if is_allowed_candidate_date(record, candidate)
    ]
    if not candidates:
        return None
    if prefer_latest:
        return max(candidates, key=parse_normalized_date)
    return candidates[0]


def extract_labeled_date(
    record: dict[str, str],
    lines: list[str],
    labels: tuple[str, ...],
    *,
    prefer_latest: bool = False,
) -> str | None:
    for index, line in enumerate(lines):
        lowered = line.lower()
        if not any(label in lowered for label in labels):
            continue

        same_line_candidate = select_candidates(
            record,
            line,
            prefer_latest=prefer_latest,
        )
        if same_line_candidate:
            return same_line_candidate

        if index + 1 < len(lines):
            next_line_candidate = select_candidates(
                record,
                " ".join(lines[index : index + 2]),
                prefer_latest=prefer_latest,
            )
            if next_line_candidate:
                return next_line_candidate
    return None


def parse_conference_date_range(record: dict[str, str], context: str) -> tuple[str, str] | None:
    same_month_match = re.search(
        rf"({MONTH_PATTERN})\s+(\d{{1,2}})\s*(?:-|–|to)\s*(\d{{1,2}}),\s*(\d{{4}})",
        context,
        re.IGNORECASE,
    )
    if same_month_match:
        month, start_day, end_day, year = same_month_match.groups()
        start = normalize_date(f"{month} {start_day}, {year}")
        end = normalize_date(f"{month} {end_day}, {year}")
        if (
            start
            and end
            and is_allowed_candidate_date(record, start)
            and is_allowed_candidate_date(record, end)
        ):
            return start, end

    split_month_match = re.search(
        rf"({MONTH_PATTERN})\s+(\d{{1,2}})\s*(?:-|–|to)\s*({MONTH_PATTERN})\s+(\d{{1,2}}),\s*(\d{{4}})",
        context,
        re.IGNORECASE,
    )
    if split_month_match:
        start_month, start_day, end_month, end_day, year = split_month_match.groups()
        start = normalize_date(f"{start_month} {start_day}, {year}")
        end = normalize_date(f"{end_month} {end_day}, {year}")
        if (
            start
            and end
            and is_allowed_candidate_date(record, start)
            and is_allowed_candidate_date(record, end)
        ):
            return start, end

    day_range_match = re.search(
        rf"(\d{{1,2}})\s*(?:-|–|to)\s*(\d{{1,2}})\s+({MONTH_PATTERN})\s+(\d{{4}})",
        context,
        re.IGNORECASE,
    )
    if day_range_match:
        start_day, end_day, month, year = day_range_match.groups()
        start = normalize_date(f"{month} {start_day}, {year}")
        end = normalize_date(f"{month} {end_day}, {year}")
        if (
            start
            and end
            and is_allowed_candidate_date(record, start)
            and is_allowed_candidate_date(record, end)
        ):
            return start, end

    ordered_candidates = [
        candidate
        for candidate in extract_ordered_date_candidates(context)
        if is_allowed_candidate_date(record, candidate)
    ]
    if len(ordered_candidates) >= 2:
        return ordered_candidates[0], ordered_candidates[1]
    return None


def extract_location_value(lines: list[str]) -> str | None:
    def normalize_location_candidate(value: str) -> str | None:
        cleaned = collapse_whitespace(value).strip(" -:")
        lowered = cleaned.lower()
        if not cleaned:
            return None
        if any(phrase in lowered for phrase in DISALLOWED_LOCATION_PHRASES):
            return None
        if ":" in cleaned:
            return None
        if any(marker in cleaned for marker in (".", "!", "?", ";", "/", "http")):
            return None
        if any(char.isdigit() for char in cleaned):
            return None
        if len(cleaned) > 80:
            return None
        words = re.findall(r"[A-Za-z][A-Za-z'&.-]*", cleaned)
        if not words or len(words) > 8:
            return None
        if cleaned.count(",") > 2:
            return None
        if "," in cleaned:
            parts = [part.strip() for part in cleaned.split(",")]
            if any(len(part) < 2 for part in parts):
                return None
            if not re.search(r"[A-Za-z]", "".join(parts)):
                return None
            return cleaned
        if len(words) <= 2 and cleaned == cleaned.title():
            return cleaned
        return None

    for index, line in enumerate(lines):
        lowered = line.lower()
        if not any(label in lowered for label in LOCATION_LINE_LABELS):
            continue
        match = re.search(r"(?:location|venue)\s*[:\-]\s*(.+)", line, re.IGNORECASE)
        if match:
            candidate = normalize_location_candidate(match.group(1))
            if candidate:
                return candidate
        if index + 1 < len(lines):
            candidate = normalize_location_candidate(lines[index + 1])
            if candidate:
                return candidate
    return None


def extract_structured_updates_from_snapshot(
    record: dict[str, str],
    snapshot: PageSnapshot,
) -> tuple[dict[str, str], float, str] | None:
    if not snapshot.ok or not snapshot.lines:
        return None

    updates: dict[str, str] = {}
    reasons: list[str] = []

    extended_deadline, _ = detect_deadline_extension_signal(record, snapshot)
    if extended_deadline:
        updates["submission_deadline"] = extended_deadline
        reasons.append("extended submission deadline")

    submission_deadline = extract_labeled_date(
        record,
        snapshot.lines,
        SUBMISSION_LINE_LABELS,
        prefer_latest=True,
    )
    if submission_deadline and "submission_deadline" not in updates:
        updates["submission_deadline"] = submission_deadline
        reasons.append("submission deadline")

    notification = extract_labeled_date(
        record,
        snapshot.lines,
        NOTIFICATION_LINE_LABELS,
    )
    if notification:
        updates["notification"] = notification
        reasons.append("notification date")

    for context in contextual_line_windows(snapshot.lines, CONFERENCE_DATE_LINE_LABELS):
        conference_dates = parse_conference_date_range(record, context)
        if conference_dates:
            updates["conference_start"], updates["conference_end"] = conference_dates
            reasons.append("conference dates")
            break

    if not updates:
        return None

    explicit_field_count = len(updates)
    confidence = 0.74
    if explicit_field_count >= 2:
        confidence = 0.8
    if explicit_field_count >= 4:
        confidence = 0.86

    source_url = snapshot.final_url or snapshot.url
    reason = collapse_whitespace(
        f"Heuristically extracted {', '.join(reasons)} from {source_url}."
    )
    return updates, confidence, reason


def cfp_link_score(snapshot: PageSnapshot, link: PageLink) -> int:
    candidate_url = link.url.strip()
    current_url = (snapshot.final_url or snapshot.url).strip()
    if not candidate_url or candidate_url == current_url:
        return -1
    if is_disallowed_conference_url(candidate_url):
        return -1

    snapshot_host = urllib.parse.urlparse(current_url).netloc.lower()
    candidate = urllib.parse.urlparse(candidate_url)
    if candidate.scheme not in {"http", "https"} or not candidate.netloc:
        return -1
    if candidate.netloc.lower() != snapshot_host:
        return -1

    signal_text = f"{candidate.path} {link.text}".lower()
    score = 0
    if "/cfp" in candidate.path.lower() or "cfp" in signal_text:
        score += 8
    if "call-for-papers" in signal_text or "call for papers" in signal_text:
        score += 8
    if "important-dates" in signal_text or "important dates" in signal_text:
        score += 6
    if "submission" in signal_text:
        score += 4
    if "deadline" in signal_text:
        score += 4
    if "research" in signal_text:
        score += 1
    return score


def linked_cfp_candidates(snapshot: PageSnapshot, limit: int = 5) -> list[PageLink]:
    scored_links: list[tuple[int, PageLink]] = []
    for link in snapshot.links:
        score = cfp_link_score(snapshot, link)
        if score >= 8:
            scored_links.append((score, link))
    scored_links.sort(key=lambda item: (-item[0], item[1].url))
    return [link for _, link in scored_links[:limit]]


def preferred_linked_cfp_url(snapshot: PageSnapshot) -> str | None:
    candidates = linked_cfp_candidates(snapshot, limit=1)
    return candidates[0].url if candidates else None


def record_reference_year(record: dict[str, str]) -> int | None:
    year = year_from_date(record.get("conference_start")) or year_from_date(
        record.get("submission_deadline")
    )
    return int(year) if year and year.isdigit() else None


def extract_year_hints(value: str | None) -> list[int]:
    if not value:
        return []
    hints: set[int] = set()
    for match in re.findall(r"(?<!\d)(20\d{2})(?!\d)", value):
        hints.add(int(match))
    for match in re.findall(r"(?<!\d)(2[5-9]|3\d)(?!\d)", value):
        hints.add(2000 + int(match))
    return sorted(hints)


def url_edition_year(record: dict[str, str], url: str | None) -> int | None:
    reference_year = record_reference_year(record)
    year_hints = extract_year_hints(url)
    if reference_year is None:
        return max(year_hints) if year_hints else None

    eligible_hints = [year for year in year_hints if year >= reference_year]
    return max(eligible_hints) if eligible_hints else None


def is_older_edition_url(
    record: dict[str, str],
    candidate_url: str | None,
    baseline_url: str | None,
) -> bool:
    baseline_year = url_edition_year(record, baseline_url)
    candidate_year = url_edition_year(record, candidate_url)
    reference_year = record_reference_year(record)
    if reference_year is None or baseline_year is None or candidate_year is None:
        return False
    return baseline_year > reference_year and candidate_year < baseline_year


def is_incompatible_edition_url(record: dict[str, str], url: str | None) -> bool:
    if not url:
        return False
    reference_year = record_reference_year(record)
    year_hints = sorted(set(extract_year_hints(url)))
    if reference_year is None or not year_hints:
        return False
    if len(year_hints) > 1:
        return True
    return year_hints[0] < reference_year


def cfp_url_signal_score(record: dict[str, str], url: str | None) -> int:
    if (
        not url
        or not valid_url(url)
        or is_disallowed_conference_url(url)
        or is_pdf_url(url)
        or is_incompatible_edition_url(record, url)
    ):
        return -1
    parsed = urllib.parse.urlparse(url)
    signal_text = f"{parsed.path} {parsed.query}".lower()
    score = 0
    if "/cfp" in parsed.path.lower() or "cfp" in signal_text:
        score += 8
    if "call-for-papers" in signal_text or "call for papers" in signal_text:
        score += 8
    if "important-dates" in signal_text or "important dates" in signal_text:
        score += 6
    if "submission" in signal_text:
        score += 4
    if "deadline" in signal_text:
        score += 4
    if "research" in signal_text:
        score += 1
    return score


def merge_selected_url(
    record: dict[str, str],
    analysis: dict[str, Any],
    heuristic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(analysis)
    current_url = record.get("website")
    selected_url = merged.get("selected_url")
    heuristic_selected_url = heuristic.get("selected_url") if heuristic else None

    candidates = [
        candidate
        for candidate in (current_url, selected_url, heuristic_selected_url)
        if (
            candidate
            and not is_older_edition_url(record, candidate, current_url)
            and (
                candidate == current_url
                or not is_incompatible_edition_url(record, candidate)
            )
        )
    ]
    if not candidates:
        return merged

    best_url = max(
        candidates,
        key=lambda candidate: (
            cfp_url_signal_score(record, str(candidate)),
            str(candidate) != (current_url or ""),
        ),
    )
    if best_url:
        merged["selected_url"] = best_url
    return merged


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


def wait_for_llm_request_slot() -> None:
    global _next_llm_request_at

    now = time.monotonic()
    if _next_llm_request_at > now:
        time.sleep(_next_llm_request_at - now)
    _next_llm_request_at = time.monotonic() + LLM_REQUEST_INTERVAL_SECONDS


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


def build_pages_payload(snapshots: list[PageSnapshot]) -> list[dict[str, Any]]:
    pages = []
    for snapshot in snapshots:
        pages.append(
            {
                "url": snapshot.final_url or snapshot.url,
                "status_code": snapshot.status_code,
                "error": snapshot.error,
                "excerpt": build_excerpt(snapshot.text) if snapshot.text else "",
                "linked_pages": [
                    {"url": link.url, "text": link.text}
                    for link in linked_cfp_candidates(snapshot)
                ],
            }
        )
    return pages


def request_llm_completion(payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    payload = {
        "model": get_env_value("OPENAI_MODEL", DEFAULT_MODEL),
        "temperature": 0,
        **payload,
    }

    api_key = get_env_value("OPENAI_API_KEY")
    base_url = get_env_value("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
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
            wait_for_llm_request_slot()
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


def parse_batch_result_id(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def analyze_batch_with_llm(
    prepared_batch: list[PreparedConference],
    timeout: int,
) -> dict[int, dict[str, Any]]:
    entries = []
    for prepared in prepared_batch:
        entries.append(
            {
                "id": prepared.index,
                "current_record": prepared.record,
                "pages": build_pages_payload(prepared.snapshots),
            }
        )

    response = request_llm_completion(
        {
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": textwrap.dedent(
                        """
                        You maintain a structured dataset of conference CFPs.
                        Read each entry's current record and page excerpts.
                        Only use facts that are explicitly present in the excerpts.
                        You may use linked_pages only to choose a better selected_url, not to infer facts from pages that were not fetched.

                        Return JSON with this shape:
                        {
                          "results": [
                            {
                              "id": 123,
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
                          ]
                        }

                        Rules:
                        - Return exactly one result for every provided id.
                        - Prefer "review" instead of guessing.
                        - Use "update" only when the excerpt clearly describes the same conference edition.
                        - Keep fields unchanged unless the new value is explicit.
                        - If both an original deadline and an extended/new/final/hard deadline appear, use the currently active extended submission deadline.
                        - Do not confuse submission deadlines with abstract deadlines, workshop deadlines, camera-ready deadlines, or notification dates.
                        - Dates must use DD.MM.YYYY.
                        - Do not use EasyChair, HotCRP, or EDAS URLs as selected_url or website.
                        - Do not use PDF URLs as selected_url or website. Prefer HTML CFP pages because PDF CFPs are often stale after deadline extensions.
                        - selected_url should be the best CFP URL among the provided pages.
                        """
                    ).strip(),
                },
                {
                    "role": "user",
                    "content": json.dumps({"entries": entries}, ensure_ascii=False),
                },
            ],
        },
        timeout,
    )

    results = response.get("results")
    if not isinstance(results, list):
        raise ValueError("LLM batch response did not contain a 'results' list")

    parsed_results: dict[int, dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        result_id = parse_batch_result_id(result.get("id"))
        if result_id is not None:
            parsed_results[result_id] = result

    return parsed_results


def best_structured_heuristic_result(
    record: dict[str, str],
    snapshots: list[PageSnapshot],
) -> dict[str, Any] | None:
    best_result: dict[str, Any] | None = None
    best_key: tuple[int, float, int] | None = None

    for snapshot in snapshots:
        extracted = extract_structured_updates_from_snapshot(record, snapshot)
        if not extracted:
            continue
        updates, confidence, reason = extracted
        selected_url = merge_selected_url(
            record,
            {"selected_url": snapshot.final_url or snapshot.url},
            {"selected_url": preferred_linked_cfp_url(snapshot)},
        ).get("selected_url")
        ranking_key = (
            len(updates),
            confidence,
            cfp_url_signal_score(record, selected_url),
        )
        if best_key is None or ranking_key > best_key:
            best_key = ranking_key
            best_result = {
                "status": "review",
                "confidence": confidence,
                "reason": reason,
                "selected_url": selected_url,
                "record": updates,
            }

    return best_result


def heuristic_analysis(record: dict[str, str], snapshots: list[PageSnapshot]) -> dict[str, Any]:
    current = snapshots[0]
    linked_cfp_url = preferred_linked_cfp_url(current)
    selected_url = merge_selected_url(
        record,
        {"selected_url": linked_cfp_url},
        {"selected_url": current.final_url or current.url},
    ).get("selected_url")

    structured_result = best_structured_heuristic_result(record, snapshots)
    if structured_result:
        return structured_result

    if not current.ok:
        return {
            "status": "review",
            "confidence": 0.2,
            "reason": f'Current website is unreachable ({current.error or "request failed"}).',
            "selected_url": None,
            "record": {},
        }

    if is_disallowed_conference_url(current.final_url or current.url):
        return {
            "status": "review",
            "confidence": 0.68,
            "reason": "Current website points to EasyChair, HotCRP, or EDAS. Use the public conference or CFP page instead.",
            "selected_url": preferred_public_snapshot_url(record, snapshots[1:]),
            "record": {},
        }

    extended_deadline, extension_reason = detect_deadline_extension_signal(record, current)
    if extension_reason:
        reason = extension_reason
        if linked_cfp_url:
            reason = collapse_whitespace(
                f"{reason} Homepage links to a likely CFP page: {linked_cfp_url}."
            )
        return {
            "status": "review",
            "confidence": 0.74 if extended_deadline else 0.58,
            "reason": reason,
            "selected_url": selected_url,
            "record": {"submission_deadline": extended_deadline} if extended_deadline else {},
        }

    if page_mentions_current_deadline(record, current):
        return {
            "status": "unchanged",
            "confidence": 0.72,
            "reason": "Current submission deadline appears on the page.",
            "selected_url": selected_url,
            "record": {},
        }

    text = current.text.lower()
    expected_year = year_from_date(record.get("conference_start")) or year_from_date(
        record.get("submission_deadline")
    )
    if expected_year and expected_year in text:
        reason = "Page is reachable, but heuristic fallback cannot confidently extract structured updates."
        if linked_cfp_url:
            reason = collapse_whitespace(
                f"{reason} Homepage links to a likely CFP page: {linked_cfp_url}."
            )
        return {
            "status": "review",
            "confidence": 0.5,
            "reason": reason,
            "selected_url": selected_url,
            "record": {},
        }

    reason = "Page content looks stale or ambiguous."
    if linked_cfp_url:
        reason = collapse_whitespace(
            f"{reason} Homepage links to a likely CFP page: {linked_cfp_url}."
        )
    return {
        "status": "review",
        "confidence": 0.35,
        "reason": reason,
        "selected_url": selected_url,
        "record": {},
    }


def sanitize_candidate_record(
    original: dict[str, str],
    proposed: dict[str, Any] | None,
    selected_url: str | None,
    allowed_fields: tuple[str, ...] = AUTO_UPDATE_FIELDS,
    allow_selected_url_for_website: bool = True,
) -> tuple[dict[str, str], list[str]]:
    updated = dict(original)
    changed_fields: list[str] = []
    proposed = proposed or {}

    for field in allowed_fields:
        incoming = proposed.get(field)
        if field in DATE_FIELDS:
            normalized = normalize_date(str(incoming).strip()) if incoming else None
            if normalized and normalized != original.get(field):
                updated[field] = normalized
                changed_fields.append(field)
            continue

        if field == "website":
            candidate_urls = []
            if selected_url and allow_selected_url_for_website:
                candidate_urls.append(selected_url)
            if incoming:
                candidate_urls.append(str(incoming).strip())
            for candidate_url in candidate_urls:
                if (
                    candidate_url
                    and valid_url(candidate_url)
                    and not is_disallowed_conference_url(candidate_url)
                    and not is_pdf_url(candidate_url)
                    and not is_incompatible_edition_url(original, candidate_url)
                    and not is_older_edition_url(original, candidate_url, original.get(field))
                    and candidate_url != original.get(field)
                ):
                    updated[field] = candidate_url
                    changed_fields.append(field)
                    break
            continue

        if incoming:
            cleaned = collapse_whitespace(str(incoming))
            if cleaned and cleaned != original.get(field):
                updated[field] = cleaned
                changed_fields.append(field)

    return updated, changed_fields


def promote_heuristic_analysis(
    record: dict[str, str],
    analysis: dict[str, Any],
    min_confidence: float,
) -> dict[str, Any]:
    promoted = dict(analysis)
    confidence = float(promoted.get("confidence", 0.0) or 0.0)
    if confidence < min_confidence:
        return promoted

    _, changed_fields = sanitize_candidate_record(
        record,
        promoted.get("record"),
        promoted.get("selected_url"),
        allowed_fields=DATE_FIELDS,
    )
    if changed_fields:
        promoted["status"] = "update"
    elif any(field in DATE_FIELDS for field in (promoted.get("record") or {})):
        promoted["status"] = "unchanged"
    return promoted


def should_search_for_replacement(record: dict[str, str], snapshot: PageSnapshot) -> bool:
    if is_disallowed_conference_url(record.get("website")):
        return True
    if is_pdf_url(record.get("website")):
        return True
    if not snapshot.ok:
        return True
    if len(snapshot.text) < 500:
        return True
    expected_years = {
        year_from_date(record.get("conference_start")),
        year_from_date(record.get("submission_deadline")),
    }
    return not any(year and year in snapshot.text for year in expected_years)


def preferred_public_snapshot_url(record: dict[str, str], snapshots: list[PageSnapshot]) -> str | None:
    for snapshot in snapshots:
        candidate_url = snapshot.final_url or snapshot.url
        if (
            snapshot.ok
            and valid_url(candidate_url)
            and not is_disallowed_conference_url(candidate_url)
            and not is_pdf_url(candidate_url)
            and not is_incompatible_edition_url(record, candidate_url)
        ):
            return candidate_url
    return None


def prepare_conference(
    record: dict[str, str],
    timeout: int,
    search_fallback: bool,
) -> PreparedConference:
    primary_snapshot = fetch_page(record["website"], timeout)
    snapshots = [primary_snapshot]

    if search_fallback and should_search_for_replacement(record, primary_snapshot):
        for candidate_url in search_candidate_urls(record, timeout):
            if candidate_url == record["website"]:
                continue
            candidate_snapshot = fetch_page(candidate_url, timeout)
            snapshots.append(candidate_snapshot)

    heuristic = heuristic_analysis(record, snapshots)
    return PreparedConference(
        index=-1,
        record=record,
        snapshots=snapshots,
        heuristic=heuristic,
    )


def build_heuristic_fallback_analysis(
    heuristic: dict[str, Any],
    primary_snapshot: PageSnapshot,
    exc: Exception,
) -> dict[str, Any]:
    analysis = dict(heuristic)
    heuristic_reason = collapse_whitespace(str(heuristic.get("reason", "")))
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
        prefix = "LLM rate-limited; using heuristic result."
    elif isinstance(exc, urllib.error.HTTPError) and exc.code == 401:
        prefix = "LLM provider rejected the API key or endpoint configuration (401 Unauthorized); using heuristic result."
    else:
        prefix = f"LLM analysis failed: {exc}. Using heuristic result."
    analysis["reason"] = collapse_whitespace(f"{prefix} {heuristic_reason}")
    if not analysis.get("selected_url"):
        analysis["selected_url"] = primary_snapshot.final_url or primary_snapshot.url
    return analysis


def finalize_analysis(
    record: dict[str, str],
    analysis: dict[str, Any],
    min_confidence: float,
    allowed_fields: tuple[str, ...] = AUTO_UPDATE_FIELDS,
    allow_selected_url_for_website: bool = True,
) -> AnalysisResult:
    updated_record, changed_fields = sanitize_candidate_record(
        record,
        analysis.get("record"),
        analysis.get("selected_url"),
        allowed_fields=allowed_fields,
        allow_selected_url_for_website=allow_selected_url_for_website,
    )
    status = str(analysis.get("status", "review")).strip().lower()
    confidence = float(analysis.get("confidence", 0.0) or 0.0)
    reason = collapse_whitespace(str(analysis.get("reason", "No explanation provided.")))
    selected_url = analysis.get("selected_url")
    applied_fields: list[str] = []
    applied_record = dict(record)

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
            applied_fields=applied_fields,
            applied_record=applied_record,
            review_note=review_note,
        )

    if status == "update" and not changed_fields:
        status = "unchanged"
        reason = "Model returned update, but no allowed fields changed."

    review_note = None
    if status == "update":
        applied_fields = list(changed_fields)
        applied_record = updated_record

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
        applied_fields=applied_fields,
        applied_record=applied_record,
        review_note=review_note,
    )


def conference_label(record: dict[str, str]) -> str:
    acronym = collapse_whitespace(record.get("acronym", ""))
    name = collapse_whitespace(record.get("name", ""))
    if acronym and name and acronym != name:
        return f"{acronym} ({name})"
    return acronym or name or "unknown conference"


def format_log_date(value: datetime) -> str:
    return value.strftime("%d.%m.%Y")


def should_process_conference(record: dict[str, str]) -> tuple[bool, str]:
    deadline = parse_normalized_date(record.get("submission_deadline"))
    if not deadline:
        return False, "submission deadline is missing or invalid"

    deadline_date = deadline.date()
    deadline_text = format_log_date(deadline)
    today = datetime.now(LOCAL_TIMEZONE).date()
    if deadline_date >= today:
        if deadline_date == today:
            return False, f"submission deadline is today ({deadline_text})"
        return False, f"submission deadline is in the future ({deadline_text})"

    if deadline_date.year < today.year:
        return True, f"submission deadline is from a previous year ({deadline_text})"

    deadline_age_days = (today - deadline_date).days
    if deadline_age_days > RECENT_DEADLINE_WINDOW_DAYS:
        return (
            False,
            f"submission deadline is older than {RECENT_DEADLINE_WINDOW_DAYS} days in the current year ({deadline_text})",
        )

    return True, f"submission deadline is within the last {RECENT_DEADLINE_WINDOW_DAYS} days ({deadline_text})"


def conference_processing_priority(record: dict[str, str]) -> tuple[int, int]:
    deadline = parse_normalized_date(record.get("submission_deadline"))
    if not deadline:
        raise ValueError("conference_processing_priority requires a valid deadline")

    deadline_ordinal = deadline.date().toordinal()
    today_year = datetime.now(LOCAL_TIMEZONE).date().year
    if deadline.date().year < today_year:
        return (0, deadline_ordinal)
    return (1, -deadline_ordinal)


def chunk_prepared_entries(
    prepared_entries: list[PreparedConference],
    size: int,
) -> list[list[PreparedConference]]:
    return [
        prepared_entries[index : index + size]
        for index in range(0, len(prepared_entries), size)
    ]


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
    skipped: int,
    updated: list[tuple[dict[str, str], AnalysisResult]],
    reviews: list[AnalysisResult],
    unchanged: int,
    llm_mode: bool,
) -> str:
    lines = [
        "# Conference Agent",
        "",
        f"- Mode: {'auto-update' if llm_mode else 'check-only'}",
        (
            "- Checked conferences: previous-year deadlines first (oldest first), "
            f"then current-year deadlines from the last {RECENT_DEADLINE_WINDOW_DAYS} days "
            "(most recent first)."
        ),
        f"- Processed conferences: {processed}",
        f"- Skipped conferences: {skipped}",
        f"- Applied updates: {len(updated)}",
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
                    result.applied_fields,
                    original,
                    result.applied_record,
                )
            )
        lines.append("")

    if reviews:
        lines.append("## Needs Review")
        lines.append("")
        for result in reviews:
            detail = result.review_note or result.reason
            if result.applied_fields:
                detail = collapse_whitespace(
                    f"{detail} Auto-applied fields: {', '.join(result.applied_fields)}."
                )
            suffix = f" URL: {result.selected_url}" if result.selected_url else ""
            lines.append(
                f"- `{result.acronym}`: {detail} (confidence {result.confidence:.2f}).{suffix}"
            )
        lines.append("")

    if not llm_mode:
        lines.append("## Configuration")
        lines.append("")
        lines.append(
            f"- Set the `OPENAI_API_KEY` secret to enable structured extraction. Default provider settings use `{DEFAULT_MODEL}` at `{DEFAULT_BASE_URL}`."
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
    processed_count = 0
    skipped_count = 0
    pending_llm_entries: list[PreparedConference] = []
    selected_conferences: list[tuple[tuple[int, int], int, dict[str, str], str, str]] = []

    for index, conference in enumerate(conferences):
        should_process, deadline_reason = should_process_conference(conference)
        label = conference_label(conference)
        if not should_process:
            skipped_count += 1
            sys.stdout.write(f"[skip] {label}: {deadline_reason}\n")
            continue

        selected_conferences.append(
            (
                conference_processing_priority(conference),
                index,
                conference,
                label,
                deadline_reason,
            )
        )

    selected_conferences.sort(key=lambda entry: entry[0])

    if args.limit is not None and len(selected_conferences) > limit:
        for _, _, _, label, _ in selected_conferences[limit:]:
            skipped_count += 1
            sys.stdout.write(
                f"[skip] {label}: deferred by --limit {limit} after priority sorting\n"
            )
        selected_conferences = selected_conferences[:limit]

    for _, index, conference, label, deadline_reason in selected_conferences:
        processed_count += 1
        sys.stdout.write(f"[check] {label}: {deadline_reason}\n")
        prepared = prepare_conference(
            conference,
            timeout=args.timeout,
            search_fallback=args.search_fallback,
        )
        prepared.index = index
        heuristic_analysis = promote_heuristic_analysis(
            conference,
            prepared.heuristic,
            args.heuristic_min_confidence,
        )
        heuristic_result = finalize_analysis(
            conference,
            heuristic_analysis,
            args.heuristic_min_confidence,
            allowed_fields=DATE_FIELDS,
            allow_selected_url_for_website=False,
        )

        if llm_enabled() and heuristic_result.status == "review":
            pending_llm_entries.append(prepared)
            continue

        result = heuristic_result
        if result.applied_fields:
            original = dict(conference)
            conferences[index] = result.applied_record
            updated_results.append((original, result))
        if result.status == "review":
            review_results.append(result)
        elif not result.applied_fields:
            unchanged_count += 1

    for batch in chunk_prepared_entries(pending_llm_entries, LLM_BATCH_SIZE):
        batch_error: Exception | None = None
        batch_analysis: dict[int, dict[str, Any]] = {}
        try:
            batch_analysis = analyze_batch_with_llm(batch, args.timeout)
        except Exception as exc:
            batch_error = exc

        for prepared in batch:
            analysis = batch_analysis.get(prepared.index)
            llm_selected_url_confirmed = False
            if analysis is None:
                fallback_exc = batch_error or ValueError(
                    f"LLM batch response omitted conference id {prepared.index}"
                )
                analysis = build_heuristic_fallback_analysis(
                    prepared.heuristic,
                    prepared.snapshots[0],
                    fallback_exc,
                )
            else:
                llm_selected_url_confirmed = bool(analysis.get("selected_url"))
            analysis = merge_selected_url(
                prepared.record,
                analysis,
                prepared.heuristic,
            )

            result = finalize_analysis(
                prepared.record,
                analysis,
                args.min_confidence,
                allow_selected_url_for_website=llm_selected_url_confirmed,
            )
            if result.applied_fields:
                original = dict(prepared.record)
                conferences[prepared.index] = result.applied_record
                updated_results.append((original, result))
            if result.status == "review":
                review_results.append(result)
            elif not result.applied_fields:
                unchanged_count += 1

    if args.write and updated_results:
        write_yaml(input_path, data)

    report = build_report(
        processed=processed_count,
        skipped=skipped_count,
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
