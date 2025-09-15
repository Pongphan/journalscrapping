# Journal Scrapping
What “journal scraping” is

Programmatically collecting metadata (and sometimes full-text) about journals and their articles from websites. Typical targets: journal homepages, indexers, and aggregator sites. Most projects actually aim for metadata (title, ISSN, publisher, subject areas, links, metrics) rather than the PDFs themselves.

Common uses

Build a lookup tool or dashboard (e.g., search a journal and show metrics/details).

Enrich lab or departmental lists with ISSN, publisher, scope, categories.

Track changes in policies/fees (OA status, APCs).

Aggregate article-level metadata for literature reviews (when APIs aren’t enough).

Legal & ethical essentials (read this!)

Prefer APIs first (Crossref, OpenAlex, DOAJ, PubMed). They’re free and robust.

Respect Terms of Service and robots.txt; avoid paywalled content.

Rate limit (e.g., ≤1 request/sec), add backoff & caching, identify your user agent.

Don’t bypass CAPTCHAs/anti-bot systems or share code to do so.

Be clear that “Impact Factor” (Clarivate JIF) is licensed data—don’t scrape it.

Reliable open data sources (API-first)

Crossref – article & journal metadata (DOIs, ISSNs, titles, publishers).

OpenAlex – journals/works/authors, citation counts, fields of study.

DOAJ – vetted open-access journals (metadata, APCs/policies).

PubMed (NCBI E-utilities) – biomedical article metadata/PMIDs.

SCImago (SJR) – public journal pages (H-index, SJR, quartiles/percentiles); no official API but pages can be parsed politely.

Licensed metrics (no scraping): Clarivate JCR/JIF, Scopus/SciVal.

What to collect (journal-level)

Identity: Title, ISSN/eISSN, ISSN-L

Publisher, Country/Region

Homepage URL

Subject areas / categories

Open-access policy / APC (when available)

Metrics: H-index, SJR (public), quartile/percentile
(JIF requires JCR access—don’t scrape it.)

Recommended workflow

Resolve by ISSN (or name → ISSN via Crossref/OpenAlex).

Pull journal entity from API(s).

If you still need extras (e.g., SJR percentile), politely parse the public page.

Normalize names and merge records using ISSN/ISSN-L as keys.

Cache results; re-use for repeated queries.

Store in a small DB (SQLite/Postgres) for your Streamlit app.
