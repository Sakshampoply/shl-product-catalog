#!/usr/bin/env python3
"""
Scrape the SHL Product Catalog.

Supports two modes:
1. Default (all tables / all catalog items ~384 rows)
2. Restricted to the "Individual Test Solutions" table only (legacy behaviour)

URL: https://www.shl.com/products/product-catalog/
Extracts per-row fields:
    - name: Text of the assessment
    - url: Absolute URL to the assessment page
    - remote_testing: True/False (based on green dot / "yes" icon)
    - adaptive_iri: True/False (based on green dot / "yes" icon)
    - test_type_keys: List of single-letter codes shown in the final column (e.g., A, B, C, D, E, K, P, S)
    - description, job_levels, languages, assessment_length_minutes (from detail page)

Writes JSON and CSV outputs under ./output/

CLI examples:
    python scrape_shl_catalog.py                # scrape ALL (~384) items
    python scrape_shl_catalog.py --individual    # scrape only Individual Test Solutions (~156)
    python scrape_shl_catalog.py --no-details    # skip per-item detail page requests (faster)

This script uses only requests + BeautifulSoup and attempts to be resilient to minor HTML changes.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse
import re

import requests
from bs4 import BeautifulSoup, Tag

BASE_URL = "https://www.shl.com/products/product-catalog/"
HEADERS = {
    # A reasonable browser-like header set to avoid basic bot blocking
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class CatalogRow:
    name: str
    url: str
    remote_testing: Optional[bool]
    adaptive_iri: Optional[bool]
    test_type_keys: List[str]
    # Details from the product page
    description: Optional[str] = None
    job_levels: List[str] = None  # populated after detail fetch
    languages: List[str] = None
    assessment_length_minutes: Optional[int] = None


def _is_yes_icon(cell: Tag) -> Optional[bool]:
    """Return True if the cell contains a green/yes icon, False if an explicit "no" icon,
    and None if not determinable.

    The site uses dot icons with classes like one of the following (examples seen in the DOM):
    - catalogue_circle yes
    - catalogue__circle yes
    - catalogue-circle yes
    We check for any tag having class "yes" and containing the word "circle" in class names.
    """
    if not cell:
        return None
    # Positive/negative explicit icons
    yes_like = cell.select('[class*="circle"].yes, .yes[class]')
    no_like = cell.select('[class*="circle"].no, .no[class]')
    if yes_like:
        return True
    if no_like:
        return False
    # Sometimes they may put the text "yes"/"no"
    text = cell.get_text(strip=True).lower()
    if text == "yes":
        return True
    if text == "no":
        return False
    return None


def _extract_keys(cell: Tag) -> List[str]:
    """Extract test type keys (single letters) from the last column.

    Looks for spans with classes like 'product-catalogue__key' (hyphenated or underscored),
    falls back to parsing single-letter tokens in the text.
    """
    if not cell:
        return []
    spans = cell.select(
        "span[class*='product-catalogue'][class*='key'], span[class*='product_catalogue'][class*='key']"
    )
    keys: List[str] = []
    for sp in spans:
        t = sp.get_text(strip=True)
        if t and len(t) == 1 and t.isalpha():
            keys.append(t)
        else:
            dt = sp.get("data-tooltip") or sp.get("title")
            if dt and len(dt.strip()) == 1 and dt.strip().isalpha():
                keys.append(dt.strip())
    if keys:
        return keys
    # Fallback: split the visible text by whitespace and keep single uppercase letters
    text = " ".join(cell.stripped_strings)
    for tok in text.replace("|", " ").split():
        if len(tok) == 1 and tok.isalpha():
            keys.append(tok)
    return keys


def _parse_rows(table: Tag, page_url: str) -> Iterable[CatalogRow]:
    for tr in table.find_all("tr"):
        # Skip header rows that contain <th>
        if tr.find("th") is not None:
            continue
        tds = tr.find_all("td", recursive=False)
        if not tds:
            # Some rows may be wrapper rows; try all descendants
            tds = tr.find_all("td")
        if not tds:
            continue
        # First cell: name + link
        name_cell = tds[0]
        a = name_cell.find("a", href=True)
        name = a.get_text(strip=True) if a else name_cell.get_text(strip=True)
        if not name:
            continue  # ignore empty rows
        href = a["href"] if a else ""
        full_url = urljoin(page_url, href)

        # Second and third cell: yes/no icons for Remote Testing and Adaptive/IRI
        remote = _is_yes_icon(tds[1]) if len(tds) > 1 else None
        adaptive = _is_yes_icon(tds[2]) if len(tds) > 2 else None

        # Last cell: keys
        keys_cell = tds[-1] if tds else None
        keys = _extract_keys(keys_cell)

        yield CatalogRow(
            name=name,
            url=full_url,
            remote_testing=remote,
            adaptive_iri=adaptive,
            test_type_keys=keys,
        )


def _find_next_url_near_table(table: Tag, current_url: str) -> Optional[str]:
    """Find the pagination link that belongs to THIS table, not other sections.

    Strategy:
    - Look within the nearest ancestor wrapper for an anchor that contains
      the word "Next" and an href with a "start=" query parameter.
    - Prefer hrefs that also include "type=1" (catalog section identifier seen on site).
    - Fallback to the first following "Next" link.
    """

    def pick_next(container: Tag) -> Optional[str]:
        if not container:
            return None
        # First prefer links with type=1
        for a in container.select("a[href*='start='][href*='type=1']"):
            if "next" in a.get_text(strip=True).lower():
                return urljoin(current_url, a.get("href", ""))
        # Then any link labeled Next with start=
        for a in container.select("a[href*='start=']"):
            if "next" in a.get_text(strip=True).lower():
                return urljoin(current_url, a.get("href", ""))
        return None

    # Try within a few ancestors
    parent = table
    for _ in range(4):
        parent = parent.parent if parent else None
        if isinstance(parent, Tag):
            out = pick_next(parent)
            if out:
                return out

    # Fallback: first following Next link in DOM
    cand = table.find_next(
        lambda t: isinstance(t, Tag)
        and t.name == "a"
        and "next" in t.get_text(strip=True).lower()
    )
    if cand and cand.has_attr("href"):
        return urljoin(current_url, cand["href"])
    return None


def _find_individual_test_solutions_table(soup: BeautifulSoup) -> Optional[Tag]:
    """Locate ONLY the 'Individual Test Solutions' table and return it if present."""
    for table in soup.find_all("table"):
        hdr = table.find(
            lambda t: t.name in {"th", "td"}
            and t.get_text(strip=True).startswith("Individual Test Solutions")
        )
        if hdr:
            return table
    return None


def _find_all_product_tables(soup: BeautifulSoup) -> List[Tag]:
    """Return all tables that appear to contain product rows, deduplicated."""
    tables: List[Tag] = []
    for table in soup.find_all("table"):
        if table.find("tr", attrs={"data-entity-id": True}):
            tables.append(table)
            continue
        for tr in table.find_all("tr"):
            if tr.find("a", href=True) and len(tr.find_all("td")) >= 2:
                tables.append(table)
                break
    seen = set()
    uniq: List[Tag] = []
    for t in tables:
        if id(t) not in seen:
            seen.add(id(t))
            uniq.append(t)
    return uniq


def _extract_section_text(soup: BeautifulSoup, header_regex: str) -> Optional[str]:
    """Return concatenated paragraph text for a section whose header matches regex.

    Looks for an h2/h3/h4 whose text matches, then collects <p> within the same parent row.
    """
    header = soup.find(
        lambda t: t.name in {"h2", "h3", "h4"}
        and re.search(header_regex, t.get_text(strip=True), re.I)
    )
    if not header:
        return None
    container = header.find_parent(
        lambda tag: isinstance(tag, Tag) and tag.name in {"div", "section"}
    )
    if not container:
        return None
    parts = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = "\n\n".join([p for p in parts if p])
    return text or None


def _parse_job_levels(text: Optional[str]) -> List[str]:
    if not text:
        return []
    levels = [s.strip() for s in re.split(r",|/|\u2022|\|", text) if s.strip()]
    # Deduplicate preserving order
    seen = set()
    out: List[str] = []
    for lv in levels:
        if lv not in seen:
            seen.add(lv)
            out.append(lv)
    return out


def fetch_detail(url: str, sess: requests.Session, timeout: int = 30) -> dict:
    resp = sess.get(url, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    # Name: prefer the first <h1>
    h1 = soup.find("h1")
    name = h1.get_text(strip=True) if h1 else None
    desc = _extract_section_text(soup, r"^Description$")
    job_levels_text = _extract_section_text(soup, r"^Job levels?$")
    job_levels = _parse_job_levels(job_levels_text)
    # Languages
    languages_text = _extract_section_text(soup, r"^Languages?$")
    languages = []
    if languages_text:
        # Split by commas, bullets, or pipes; keep entries like "English (USA)"
        languages = [
            s.strip() for s in re.split(r",|/|\u2022|\|", languages_text) if s.strip()
        ]

    # Assessment length (minutes)
    assess_text = _extract_section_text(soup, r"^Assessment length$")
    minutes: Optional[int] = None
    if assess_text:
        # Try to find an explicit minutes number
        m = re.search(r"(\d+)\s*(?:min(?:ute)?s?)", assess_text, flags=re.I)
        if not m:
            # handle formats like "= 16" or "~ 25"
            m = re.search(r"=\s*(\d+)|(~|≈)?\s*(\d+)$", assess_text)
        if m:
            # m.groups may have multiple captures; pick the last numeric
            for g in m.groups()[::-1]:
                if g and g.isdigit():
                    minutes = int(g)
                    break

    return {
        "name": name,
        "description": desc,
        "job_levels": job_levels,
        "languages": languages,
        "assessment_length_minutes": minutes,
    }


def scrape_all_pages(
    start_url: str = BASE_URL,
    delay_sec: float = 0.7,
    details_delay_sec: float = 0.4,
    restrict_individual: bool = False,
    fetch_details: bool = True,
    types: Optional[List[str]] = None,
) -> List[CatalogRow]:
    """Scrape all pages for the given catalog types.

    When restrict_individual=True, overrides types to ["1"]. If types is None and not restricted, defaults to ["1", "2"].
    """
    sess = requests.Session()
    sess.headers.update(HEADERS)

    if restrict_individual:
        types = ["1"]
    elif types is None:
        types = ["1", "2"]

    all_rows: dict[str, CatalogRow] = {}

    # Simple offset sweep: iterate start=0,12,24,... and collect any product-like tables on each page.
    # Stop after several consecutive pages that yield no new items.
    all_rows: dict[str, CatalogRow] = {}
    consecutive_no_new = 0
    max_no_new = 8
    step = 12
    max_offset = 600  # safety bound

    for offset in range(0, max_offset + 1, step):
        pr = urlparse(start_url)
        q = dict(parse_qsl(pr.query, keep_blank_values=True))
        if offset:
            q["start"] = str(offset)
        else:
            # ensure no stale start param
            q.pop("start", None)
        new_q = urlencode(q)
        page_url = urlunparse(
            (pr.scheme, pr.netloc, pr.path, pr.params, new_q, pr.fragment)
        )

        resp = sess.get(page_url, timeout=30)
        if resp.status_code != 200:
            print(f"[info] Non-200 ({resp.status_code}) at {page_url}; stopping sweep.")
            break
        soup = BeautifulSoup(resp.text, "lxml")

        tables = _find_all_product_tables(soup)
        if restrict_individual:
            # prefer individual table if restricting
            t_ind = _find_individual_test_solutions_table(soup)
            tables = [t_ind] if t_ind else []

        if not tables:
            print(f"[info] No product tables found at offset={offset}; continuing.")
            consecutive_no_new += 1
            if consecutive_no_new >= max_no_new:
                print("[info] Too many consecutive empty pages; stopping sweep.")
                break
            time.sleep(delay_sec)
            continue

        page_rows: List[CatalogRow] = []
        for tbl in tables:
            page_rows.extend(list(_parse_rows(tbl, page_url)))

        # dedupe by URL and identify new rows
        new_count = 0
        for r in page_rows:
            if r.url not in all_rows:
                all_rows[r.url] = r
                new_count += 1

        print(f"[info] Offset={offset} parsed {len(page_rows)} rows ({new_count} new)")

        if new_count == 0:
            consecutive_no_new += 1
        else:
            consecutive_no_new = 0

        if fetch_details and new_count > 0:
            sess2 = requests.Session()
            sess2.headers.update(HEADERS)
            for r in list(all_rows.values()):
                if r.description is None or not r.job_levels:
                    try:
                        det = fetch_detail(r.url, sess2)
                        r.name = det.get("name") or r.name
                        r.description = det.get("description")
                        r.job_levels = det.get("job_levels") or []
                        r.languages = det.get("languages") or []
                        r.assessment_length_minutes = det.get(
                            "assessment_length_minutes"
                        )
                    except Exception as e:
                        print(f"[warn] Failed detail for {r.url}: {e}")
                    time.sleep(details_delay_sec)

        # Heuristic stop: if we've gone a long way with no new items, break
        if consecutive_no_new >= max_no_new:
            print("[info] Stopping sweep after consecutive no-new pages")
            break
        time.sleep(delay_sec)

    return list(all_rows.values())


def write_outputs(
    rows: List[CatalogRow], base_path: str = "output/shl_catalog"
) -> None:
    # JSON
    json_path = f"{base_path}.json"
    csv_path = f"{base_path}.csv"

    # Ensure output directory exists
    import os

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "url",
                "remote_testing",
                "adaptive_iri",
                "test_type_keys",
                "description",
                "job_levels",
                "languages",
                "assessment_length_minutes",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.name,
                    r.url,
                    {True: "Yes", False: "No"}.get(r.remote_testing, ""),
                    {True: "Yes", False: "No"}.get(r.adaptive_iri, ""),
                    " ".join(r.test_type_keys),
                    (r.description or "").replace("\n", " ").strip(),
                    "; ".join(r.job_levels or []),
                    "; ".join(r.languages or []),
                    (
                        r.assessment_length_minutes
                        if r.assessment_length_minutes is not None
                        else ""
                    ),
                ]
            )

    print(f"[ok] Wrote {json_path} and {csv_path}")


def main(argv: List[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Scrape SHL Product Catalog")
    ap.add_argument("--url", default=BASE_URL, help="Starting catalog URL")
    ap.add_argument(
        "--individual",
        action="store_true",
        help="Restrict to Individual Test Solutions table only",
    )
    ap.add_argument(
        "--no-details",
        action="store_true",
        help="Skip fetching each product's detail page",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0.7,
        help="Delay between catalog page requests (s)",
    )
    ap.add_argument(
        "--detail-delay",
        type=float,
        default=0.4,
        help="Delay between detail page requests (s)",
    )
    ap.add_argument(
        "--types",
        help="Comma-separated list of catalog type ids to scrape (default: 1,2 unless --individual)",
    )
    args = ap.parse_args(argv[1:])

    types_list = None
    if args.types:
        types_list = [t.strip() for t in args.types.split(",") if t.strip()]

    rows = scrape_all_pages(
        start_url=args.url,
        delay_sec=args.delay,
        details_delay_sec=args.detail_delay,
        restrict_individual=args.individual,
        fetch_details=not args.no_details,
        types=types_list,
    )
    if not rows:
        print(
            "[error] No rows scraped. The page structure may have changed or requires JS."
        )
        return 2
    write_outputs(rows)
    print(f"[ok] Total rows scraped: {len(rows)} (individual_only={args.individual})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
