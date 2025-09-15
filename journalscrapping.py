import re
import time
import json
import urllib.parse as ul
from typing import List, Dict, Optional

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# ---------------------------
# Config
# ---------------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; JournalScraper/1.0; +https://streamlit.io)"
}
BASE = "https://www.scimagojr.com"
SEARCH_URL = BASE + "/journalsearch.php?q={query}"

st.set_page_config(
    page_title="Journal Metrics Finder",
    page_icon="📚",
    layout="wide"
)

# ---------------------------
# Helpers
# ---------------------------

@st.cache_data(ttl=60 * 60)
def http_get(url: str, params: Optional[dict] = None) -> str:
    """GET with basic caching and headers."""
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text

def _find_results_table(soup: BeautifulSoup):
    # Try to identify the results table that has a "Title" column
    for table in soup.find_all("table"):
        ths = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if any("title" in t for t in ths):
            return table
    return None

def _absolute_url(href: str) -> str:
    if href.startswith("http"):
        return href
    return BASE + "/" + href.lstrip("/")

@st.cache_data(ttl=60 * 60)
def sjr_search(query: str) -> List[Dict]:
    """
    Search SCImago for journals by name. Returns a list of candidates with title and detail URL.
    """
    html = http_get(SEARCH_URL.format(query=ul.quote_plus(query)))
    soup = BeautifulSoup(html, "lxml")

    table = _find_results_table(soup)
    if table is None:
        return []

    candidates = []
    for row in table.select("tr")[1:]:
        cols = row.find_all("td")
        if not cols:
            continue

        # First cell should hold the Title with a link
        a = cols[0].find("a")
        if not a or not a.get("href"):
            continue

        title = a.get_text(strip=True)
        url = _absolute_url(a.get("href"))
        # Optional hints we can pick up if present
        hint = " ".join(c.get_text(" ", strip=True) for c in cols[1:]) if len(cols) > 1 else ""

        candidates.append({
            "title": title,
            "url": url,
            "hint": hint
        })

    return candidates

def _grab_first_number(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.I)
    if m:
        # prefer last group with a number
        groups = [g for g in m.groups() if g is not None]
        if groups:
            return groups[-1]
    return None

def _parse_highest_percentile_from_text(txt: str) -> Optional[int]:
    """
    Heuristic: SCImago category badges show like 'Q1 (97)' where 97 is the percentile.
    We try to find numbers in parentheses following 'Q[1-4]'.
    Fallback: look for 'percentile' phrases.
    """
    # Q1 (97) style
    nums = [int(n) for n in re.findall(r"Q[1-4]\s*\(\s*(\d{1,3})\s*\)", txt, flags=re.I) if 0 <= int(n) <= 100]
    if nums:
        return max(nums)

    # "... 97 percentile" style
    nums = [int(n) for n in re.findall(r"(\d{1,3})\s*(?:st|nd|rd|th)?\s*percentile", txt, flags=re.I) if 0 <= int(n) <= 100]
    if nums:
        return max(nums)

    return None

def _parse_categories(soup: BeautifulSoup) -> List[str]:
    """
    Try to extract the list of subject areas / categories shown on SJR page.
    We'll gather any sensible table with category-like strings.
    """
    cats = set()
    for table in soup.find_all("table"):
        # Heuristic: capture rows containing Qx or "Category"
        if "category" in (table.get_text(" ", strip=True).lower()):
            for row in table.find_all("tr"):
                tds = [td.get_text(" ", strip=True) for td in row.find_all("td")]
                for td in tds:
                    # Simple filter to avoid numeric-only cells
                    if len(td) > 3 and not re.fullmatch(r"[\d\W]+", td):
                        cats.add(td)
    # Fallback: scrape badges/labels
    if not cats:
        for span in soup.find_all(["span", "a", "div"]):
            t = span.get_text(" ", strip=True)
            if "Q1" in t or "Q2" in t or "Q3" in t or "Q4" in t:
                # Often category name is nearby; include this text anyway
                cats.add(t)
    return sorted(cats)

@st.cache_data(ttl=60 * 60)
def sjr_fetch_details(url: str) -> Dict:
    """
    Fetch a journal's detail page on SCImago and parse:
    - official name
    - H-index
    - SJR (used here as public 'impact' proxy)
    - Best quartile / Highest percentile
    - Publisher, Country, ISSN
    - Categories
    - Links
    """
    html = http_get(url)
    soup = BeautifulSoup(html, "lxml")
    page_text = soup.get_text(" ", strip=True)

    # Journal name
    h1 = soup.find("h1")
    name = h1.get_text(strip=True) if h1 else None

    # H-index
    h_index = _grab_first_number(r"H\s*[- ]?\s*index\s*:?\s*([0-9]{1,4})", page_text)

    # SJR (public indicator; not Clarivate JIF)
    sjr_val = _grab_first_number(r"SJR\s*:?\s*([0-9]+(?:\.[0-9]+)?)", page_text)

    # Best Quartile
    best_quartile = None
    m_bq = re.search(r"Best\s*Quartile\s*:?\s*(Q[1-4])", page_text, flags=re.I)
    if m_bq:
        best_quartile = m_bq.group(1).upper()

    # Highest percentile (from categories)
    highest_pct = _parse_highest_percentile_from_text(page_text)
    if highest_pct is not None:
        highest_pct = int(highest_pct)

    # Publisher / Country / ISSN (heuristics)
    publisher = _grab_first_number(r"Publisher\s*:?\s*(.+?)\s{2,}", page_text)  # not great; fallback below
    country = _grab_first_number(r"Country\s*:?\s*([A-Za-z \-]+)", page_text)
    # Try to capture ISSNs (ISSN, ISSN (Print), ISSN (Online))
    issns = re.findall(r"ISSN[^0-9]*([0-9]{4}\-?[0-9Xx]{4})", page_text)
    issns = list(dict.fromkeys(issns)) if issns else []

    # If publisher regex failed, try pulling from labeled spans
    if not publisher:
        for b in soup.find_all(["b", "strong"]):
            if "Publisher" in b.get_text():
                # next sibling text
                t = b.parent.get_text(" ", strip=True)
                m = re.search(r"Publisher\s*:?\s*(.*)$", t, flags=re.I)
                if m:
                    publisher = m.group(1).strip()
                    break

    categories = _parse_categories(soup)

    # Links: SJR page itself + any homepage link they may show
    links = {"scimago_page": url}
    # Try to find a Homepage anchor
    for a in soup.find_all("a"):
        text = a.get_text(" ", strip=True).lower()
        if "homepage" in text or "journal homepage" in text:
            links["journal_homepage"] = a.get("href")
            break

    # Compose result
    detail = {
        "journal_name": name,
        "h_index": int(h_index) if h_index else None,
        # Treat SJR as an "impact (SJR)" proxy; we will show it in the IF column with a label
        "sjr": float(sjr_val) if sjr_val else None,
        "best_quartile": best_quartile,
        "highest_percentile": highest_pct,
        "publisher": publisher,
        "country": country,
        "issn_list": issns,
        "categories": categories,
        "links": links
    }
    return detail

def row_from_detail(d: Dict) -> Dict:
    impact_str = "N/A"
    if d.get("sjr") is not None:
        impact_str = f"SJR {d['sjr']:.3f}"  # clearly labeled proxy

    details = {
        "Best quartile": d.get("best_quartile"),
        "Publisher": d.get("publisher"),
        "Country": d.get("country"),
        "ISSN(s)": ", ".join(d.get("issn_list") or []),
        "Categories": "; ".join(d.get("categories") or [])
    }
    details_str = "; ".join([f"{k}: {v}" for k, v in details.items() if v])

    return {
        "Journal name": d.get("journal_name") or "",
        "Highest percentile": d.get("highest_percentile"),
        "Impact factor": impact_str,  # proxy label
        "H-index": d.get("h_index"),
        "Details": details_str,
        "Links": json.dumps(d.get("links") or {})  # keep raw dict for download; hidden in table
    }

# ---------------------------
# UI
# ---------------------------
st.title("📚 Journal Metrics Finder")
st.caption(
    "Search a publication journal by name and view **Highest percentile (SJR)**, **Impact (SJR proxy)**, **H-index**, and details. "
    "Clarivate’s JIF is not scraped; the 'Impact factor' column uses **SJR** when JIF isn’t publicly available."
)

with st.sidebar:
    st.header("About")
    st.write(
        "- Data source: **SCImago Journal & Country Rank (SJR)** public pages.\n"
        "- H-index and category percentiles are parsed from the journal’s SJR page.\n"
        "- 'Impact factor' shown here = **SJR proxy**, *not* Clarivate JIF."
    )
    st.markdown("---")
    st.write("Tip: If the journal has many similar names, pick the exact one from the results list.")

query = st.text_input("🔎 Enter journal name", placeholder="e.g., Nature, PLoS ONE, Malaria Journal")
max_candidates = st.slider("Max search results to list", 1, 30, 10)

search_col1, search_col2 = st.columns([1, 3])
with search_col1:
    do_search = st.button("Search", type="primary")
with search_col2:
    export_btn = st.empty()

results_df = pd.DataFrame()
selected = None

if do_search and query.strip():
    with st.spinner("Searching SCImago…"):
        try:
            candidates = sjr_search(query.strip())[:max_candidates]
        except Exception as e:
            st.error(f"Search failed: {e}")
            candidates = []

    if not candidates:
        st.warning("No results found. Try a different or shorter name.")
    else:
        # Show as a selectable table
        cdf = pd.DataFrame([{"Title": c["title"], "Hint": c["hint"], "URL": c["url"]} for c in candidates])
        st.subheader("Search results")
        st.dataframe(cdf.drop(columns=["URL"]), use_container_width=True)

        pick = st.selectbox(
            "Choose a journal to fetch metrics",
            options=list(range(len(candidates))),
            format_func=lambda i: candidates[i]["title"]
        )
        chosen = candidates[pick]

        fetch = st.button("Fetch metrics for selected journal")
        if fetch:
            with st.spinner(f"Fetching metrics for **{chosen['title']}**…"):
                try:
                    detail = sjr_fetch_details(chosen["url"])
                    row = row_from_detail(detail)
                    results_df = pd.DataFrame([row])
                except Exception as e:
                    st.error(f"Failed to fetch/parse details: {e}")

if not results_df.empty:
    st.subheader("Results")
    # Hide the raw Links column in the visible table
    show_df = results_df.drop(columns=["Links"], errors="ignore")
    st.dataframe(show_df, use_container_width=True)

    # Download as CSV/JSON
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download CSV",
            data=show_df.to_csv(index=False).encode("utf-8"),
            file_name="journal_metrics.csv",
            mime="text/csv"
        )
    with c2:
        st.download_button(
            "⬇️ Download JSON",
            data=results_df.to_json(orient="records", force_ascii=False, indent=2),
            file_name="journal_metrics.json",
            mime="application/json"
        )

    # Quick links
    if "Links" in results_df.columns:
        try:
            links = json.loads(results_df.iloc[0]["Links"])
            st.markdown("**Links**:")
            if "scimago_page" in links:
                st.markdown(f"- SJR page: {links['scimago_page']}")
            if "journal_homepage" in links:
                st.markdown(f"- Journal homepage: {links['journal_homepage']}")
        except Exception:
            pass

st.markdown("---")
st.caption(
    "Educational use only. Respect site Terms of Service. For official Journal Impact Factor (JIF), "
    "please consult **Journal Citation Reports (Clarivate)**."
)
