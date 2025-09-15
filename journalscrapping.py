import re
import json
import unicodedata
import urllib.parse as ul
from typing import List, Dict, Optional
from difflib import SequenceMatcher

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
# HTTP helpers
# ---------------------------

@st.cache_data(ttl=60 * 60)
def http_get(url: str, params: Optional[dict] = None) -> str:
    """GET with basic caching and headers."""
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


def _absolute_url(href: str) -> str:
    if href.startswith("http"):
        return href
    return BASE + "/" + href.lstrip("/")


# ---------------------------
# SJR results parsing (tolerant)
# ---------------------------

def _find_results_table(soup: BeautifulSoup):
    """Look for any table that plausibly contains results (title/journal/source/publication)."""
    candidates = []
    for table in soup.find_all("table"):
        ths = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        header_blob = " ".join(ths)
        if any(k in header_blob for k in ["title", "journal", "source", "publication"]):
            candidates.append(table)
    return candidates[0] if candidates else None


def _extract_candidates_from_any_markup(soup: BeautifulSoup) -> List[Dict]:
    """Fallback: scrape journal links even without a recognizable table."""
    items: List[Dict] = []

    # A) links that point to a specific journal SID page
    for a in soup.select('a[href*="journalsearch.php"]'):
        href = a.get("href") or ""
        text = a.get_text(" ", strip=True)
        if "tip=sid" in href and text:
            items.append({
                "title": text,
                "url": _absolute_url(href),
                "hint": ""
            })

    # B) any table rows with a link in first cell
    if not items:
        for table in soup.find_all("table"):
            for row in table.find_all("tr"):
                a = row.find("a")
                if not a or not a.get("href"):
                    continue
                title = a.get_text(" ", strip=True)
                href = a.get("href")
                if not title or not href:
                    continue
                tds = row.find_all("td")
                hint = " ".join(td.get_text(" ", strip=True) for td in tds[1:]) if len(tds) > 1 else ""
                items.append({
                    "title": title,
                    "url": _absolute_url(href),
                    "hint": hint
                })

    # de-dupe by URL
    seen = set()
    uniq = []
    for it in items:
        key = it["url"]
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    return uniq


@st.cache_data(ttl=60 * 60)
def sjr_search(query: str) -> List[Dict]:
    """Search SCImago for journals by name/ISSN and return candidate rows."""
    html = http_get(SEARCH_URL.format(query=ul.quote_plus(query)))
    soup = BeautifulSoup(html, "lxml")

    candidates: List[Dict] = []

    table = _find_results_table(soup)
    if table:
        for row in table.select("tr")[1:]:
            cols = row.find_all("td")
            if not cols:
                continue
            a = cols[0].find("a")
            if not a or not a.get("href"):
                continue
            title = a.get_text(strip=True)
            url = _absolute_url(a.get("href"))
            hint = " ".join(c.get_text(" ", strip=True) for c in cols[1:]) if len(cols) > 1 else ""
            candidates.append({"title": title, "url": url, "hint": hint})

    if not candidates:
        candidates = _extract_candidates_from_any_markup(soup)

    return candidates


# ---------------------------
# Word‑intelligence helpers
# ---------------------------

STOPWORDS = {
    "the", "of", "and", "for", "in", "on", "to", "a", "an", "at", "by",
    "journal", "j", "rev", "reviews", "annals", "letters", "bulletin",
    "transactions", "archives", "acta"
}

ABBREV_EXPAND = {
    # common scholarly abbreviations
    "int": "international",
    "intl": "international",
    "int'l": "international",
    "med": "medical",
    "biol": "biology",
    "chem": "chemistry",
    "phys": "physics",
    "comput": "computing",
    "comp": "computer",
    "sci": "science",
    "eng": "engineering",
    "env": "environmental",
    "environ": "environmental",
    "technol": "technology",
    "techn": "technology",
    # orgs
    "ieee": "institute of electrical and electronics engineers",
    "acm": "association for computing machinery",
    "plos": "public library of science",
}


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")  # strip diacritics
    s = s.lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[/\-_:]+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokens(text: str) -> List[str]:
    return [t for t in normalize_text(text).split() if t]


def expand_abbrev(tok: str) -> List[str]:
    out = [tok]
    if tok in ABBREV_EXPAND:
        out.append(ABBREV_EXPAND[tok])
    return out


def make_acronym(text: str) -> str:
    toks = [t for t in tokens(text) if t not in STOPWORDS]
    ac = "".join(t[0] for t in toks if t)
    return ac.upper()


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def token_overlap_score(qtoks: List[str], ttoks: List[str]) -> float:
    if not qtoks or not ttoks:
        return 0.0
    inter = sum(1 for t in qtoks if t in ttoks)
    return inter / len(qtoks)


def hybrid_score(query: str, title: str) -> float:
    qn, tn = normalize_text(query), normalize_text(title)
    qt, tt = tokens(qn), tokens(tn)

    base = fuzzy_ratio(qn, tn)                    # 0..1
    jac = jaccard(qt, tt)                         # 0..1
    cov = token_overlap_score(qt, tt)             # 0..1
    acr_q = make_acronym(query)
    acr_t = make_acronym(title)
    acr = 1.0 if acr_q and acr_q == acr_t else 0.0

    score = 0.55 * base + 0.25 * jac + 0.20 * cov + 0.10 * acr
    return max(0.0, min(1.0, score))


def generate_query_variants(q: str) -> List[str]:
    qn = normalize_text(q)
    toks = qn.split()

    variants = {q.strip(), qn}

    # remove wrappers
    trimmed = re.sub(r"^(the|journal|journal of)\s+", "", qn)
    if trimmed != qn:
        variants.add(trimmed)

    # stopword-trimmed core
    core = " ".join([t for t in toks if t not in STOPWORDS])
    if core and core != qn:
        variants.add(core)

    # abbreviation expansions
    expanded_tokens = []
    for t in toks:
        expanded_tokens.extend(expand_abbrev(t))
    expanded = " ".join(expanded_tokens)
    if expanded and expanded != qn:
        variants.add(expanded)

    # keep raw acronym if query is all-caps letters
    raw_acr = "".join(ch for ch in q if ch.isalpha())
    if raw_acr.isupper() and len(raw_acr) >= 3:
        variants.add(raw_acr)

    # skim generic qualifiers
    drop_set = {"international", "intl", "letters", "reports", "annals", "bulletin", "transactions", "reviews", "journal"}
    skim = " ".join([t for t in toks if t not in drop_set])
    if skim and skim != qn:
        variants.add(skim)

    out = [v for v in variants if len(v) >= 2]
    out.sort(key=lambda s: (-len(s), s))
    return out[:8]


# ---------------------------
# Crossref fallback → ISSN → SJR
# ---------------------------

@st.cache_data(ttl=60 * 60)
def crossref_journal_lookup(query: str, rows: int = 8) -> List[Dict]:
    """Query Crossref for journals; return title + ISSN list to retry on SJR."""
    url = f"https://api.crossref.org/journals?query.title={ul.quote_plus(query)}&rows={rows}"
    try:
        r = requests.get(url, headers={"User-Agent": HEADERS["User-Agent"]}, timeout=20)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
    except Exception:
        return []

    out = []
    for it in items:
        title = it.get("title") or it.get("publisher") or ""
        issns = it.get("ISSN", []) or []
        out.append({"title": title, "issns": issns})
    return out


def _try_sjr_by_issn_list(issns: List[str]) -> List[Dict]:
    hits: List[Dict] = []
    for issn in issns:
        try:
            res = sjr_search(issn)
        except Exception:
            res = []
        for r in res:
            if r not in hits:
                hits.append(r)
    return hits


@st.cache_data(ttl=60 * 60)
def sjr_search_intelligent(query: str) -> List[Dict]:
    """Run multiple query variants; if none, use Crossref→ISSN as a fallback. Score & sort results."""
    seen: Dict[str, Dict] = {}
    variants = generate_query_variants(query)

    # 1) SJR with variants
    for v in variants:
        try:
            res = sjr_search(v)
        except Exception:
            res = []
        for r in res:
            key = r.get("url") or r.get("title")
            if key not in seen:
                seen[key] = r

    # 2) Crossref fallback
    if not seen:
        xrefs = crossref_journal_lookup(query, rows=8)
        issn_pool: List[str] = []
        for x in xrefs:
            issn_pool.extend(x.get("issns", []))
        issn_pool = list(dict.fromkeys(issn_pool))
        if issn_pool:
            for r in _try_sjr_by_issn_list(issn_pool):
                key = r.get("url") or r.get("title")
                if key not in seen:
                    seen[key] = r

    # score & sort
    ranked = []
    for r in seen.values():
        s = hybrid_score(query, r["title"])
        ranked.append({**r, "score": round(float(s), 4), "acronym": make_acronym(r["title"])})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


# ---------------------------
# Detail page parsing
# ---------------------------

def _grab_first_number(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, flags=re.I)
    if m:
        groups = [g for g in m.groups() if g is not None]
        if groups:
            return groups[-1]
    return None


def _parse_highest_percentile_from_text(txt: str) -> Optional[int]:
    nums = [int(n) for n in re.findall(r"Q[1-4]\s*\(\s*(\d{1,3})\s*\)", txt, flags=re.I) if 0 <= int(n) <= 100]
    if nums:
        return max(nums)
    nums = [int(n) for n in re.findall(r"(\d{1,3})\s*(?:st|nd|rd|th)?\s*percentile", txt, flags=re.I) if 0 <= int(n) <= 100]
    if nums:
        return max(nums)
    return None


def _parse_categories(soup: BeautifulSoup) -> List[str]:
    cats = set()
    for table in soup.find_all("table"):
        if "category" in (table.get_text(" ", strip=True).lower()):
            for row in table.find_all("tr"):
                tds = [td.get_text(" ", strip=True) for td in row.find_all("td")]
                for td in tds:
                    if len(td) > 3 and not re.fullmatch(r"[\d\W]+", td):
                        cats.add(td)
    if not cats:
        for span in soup.find_all(["span", "a", "div"]):
            t = span.get_text(" ", strip=True)
            if any(q in t for q in ["Q1", "Q2", "Q3", "Q4"]):
                cats.add(t)
    return sorted(cats)


@st.cache_data(ttl=60 * 60)
def sjr_fetch_details(url: str) -> Dict:
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

    # Publisher / Country / ISSN
    publisher = _grab_first_number(r"Publisher\s*:?\s*(.+?)\s{2,}", page_text)
    country = _grab_first_number(r"Country\s*:?\s*([A-Za-z \-]+)", page_text)
    issns = re.findall(r"ISSN[^0-9]*([0-9]{4}\-?[0-9Xx]{4})", page_text)
    issns = list(dict.fromkeys(issns)) if issns else []

    if not publisher:
        for b in soup.find_all(["b", "strong"]):
            if "Publisher" in b.get_text():
                t = b.parent.get_text(" ", strip=True)
                m = re.search(r"Publisher\s*:?\s*(.*)$", t, flags=re.I)
                if m:
                    publisher = m.group(1).strip()
                    break

    categories = _parse_categories(soup)

    links = {"scimago_page": url}
    for a in soup.find_all("a"):
        text = a.get_text(" ", strip=True).lower()
        if "homepage" in text or "journal homepage" in text:
            links["journal_homepage"] = a.get("href")
            break

    detail = {
        "journal_name": name,
        "h_index": int(h_index) if h_index else None,
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
    st.header("Search settings")
    mode = st.radio("Search mode", ["Word-Intelligence (recommended)", "Default"], index=0)
    auto_fetch = st.toggle("Auto-fetch top match if confidence ≥ 0.85", value=True)
    show_variants = st.toggle("Show query variants & debug", value=False)

    st.markdown("---")
    st.header("About")
    st.write(
        "- Data source: **SCImago Journal & Country Rank (SJR)** public pages.\n"
        "- H-index and category percentiles are parsed from the journal’s SJR page.\n"
        "- 'Impact factor' shown here = **SJR** proxy, *not* Clarivate JIF."
    )
    st.markdown("---")
    st.write("Tip: If the journal has many similar names, pick the exact one from the list.")

query = st.text_input("🔎 Enter journal name (ISSN or title)", placeholder="e.g., Nature, PLOS ONE, Malaria Journal")
max_candidates = st.slider("Max results to list", 1, 50, 12)

search_col1, search_col2 = st.columns([1, 3])
with search_col1:
    do_search = st.button("Search", type="primary")
with search_col2:
    export_btn = st.empty()

results_df = pd.DataFrame()

if do_search and query.strip():
    try:
        if mode.startswith("Word-Intelligence"):
            variants = generate_query_variants(query)
            if show_variants:
                with st.expander("Query variants"):
                    st.write(variants)

            with st.spinner("Searching SCImago with intelligent variants…"):
                ranked = sjr_search_intelligent(query)

            if not ranked:
                st.warning("No results found. Try a different or shorter name.")
            else:
                top = ranked[0]
                if top["score"] >= 0.75:
                    st.info(f"**Did you mean:** {top['title']}  \nConfidence: {top['score']:.2%}")

                cdf = pd.DataFrame([
                    {
                        "Title": r["title"],
                        "Hint": r.get("hint", ""),
                        "Confidence": f"{r['score']:.2%}",
                        "Acronym": r.get("acronym", ""),
                        "URL": r["url"],
                    }
                    for r in ranked[:max_candidates]
                ])
                st.subheader("Search results")
                st.dataframe(cdf.drop(columns=["URL"]), use_container_width=True)

                pick = st.selectbox(
                    "Choose a journal to fetch metrics",
                    options=list(range(min(len(ranked), max_candidates))),
                    format_func=lambda i: ranked[i]["title"],
                )
                chosen = ranked[pick]

                should_fetch = st.button("Fetch metrics for selected journal")
                if auto_fetch and ranked and ranked[0]["score"] >= 0.85:
                    chosen = ranked[0]
                    should_fetch = True
                    st.caption("Auto-fetching top match (confidence ≥ 0.85).")

                if should_fetch:
                    with st.spinner(f"Fetching metrics for **{chosen['title']}**…"):
                        try:
                            detail = sjr_fetch_details(chosen["url"])
                            row = row_from_detail(detail)
                            results_df = pd.DataFrame([row])
                        except Exception as e:
                            st.error(f"Failed to fetch/parse details: {e}")
        else:
            with st.spinner("Searching SCImago…"):
                candidates = sjr_search(query.strip())[:max_candidates]

            if not candidates:
                st.warning("No results found. Try a different or shorter name.")
            else:
                cdf = pd.DataFrame([{ "Title": c["title"], "Hint": c["hint"], "URL": c["url"] } for c in candidates])
                st.subheader("Search results")
                st.dataframe(cdf.drop(columns=["URL"]), use_container_width=True)

                pick = st.selectbox(
                    "Choose a journal to fetch metrics",
                    options=list(range(len(candidates))),
                    format_func=lambda i: candidates[i]["title"],
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
    except requests.HTTPError as e:
        st.error(f"HTTP error: {e}")
    except requests.RequestException as e:
        st.error(f"Network error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")


if not results_df.empty:
    st.subheader("Results")
    show_df = results_df.drop(columns=["Links"], errors="ignore")
    st.dataframe(show_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download CSV",
            data=show_df.to_csv(index=False).encode("utf-8"),
            file_name="journal_metrics.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "⬇️ Download JSON",
            data=results_df.to_json(orient="records", force_ascii=False, indent=2),
            file_name="journal_metrics.json",
            mime="application/json",
        )

    if "Links" in results_df.columns:
        try:
            links = json.loads(results_df.iloc[0]["Links"]) or {}
            st.markdown("**Links**:")
            if links.get("scimago_page"):
                st.markdown(f"- SJR page: {links['scimago_page']}")
            if links.get("journal_homepage"):
                st.markdown(f"- Journal homepage: {links['journal_homepage']}")
        except Exception:
            pass

st.markdown("---")
st.caption(
    "Educational use only. Respect site Terms of Service. For official Journal Impact Factor (JIF), "
    "please consult **Journal Citation Reports (Clarivate)**."
)

