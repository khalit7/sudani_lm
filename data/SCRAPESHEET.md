# Scrapesheet — web acquisition survey

> Paths in this file are relative to the repo root.

Every Sudanese-Arabic web source evaluated for this project, whether or not it was crawled.
`DATASHEET.md` (alongside this file) remains the self-contained record of the datasets themselves (size, tokens,
role, licence, examples); this file records the **survey and the decisions** — what exists, what
we took, what we refused, and why. A source only earns a DATASHEET row once it is on disk.

Survey dates: sudaneseonline 2026-08-23; full web survey 2026-08-28 (all reachability, robots,
scale and dialect figures below were probed live on that date). Plan of record: `plan.md` §1.7.

## Policy — how we crawl

- Public pages only, no authentication, no paywall circumvention.
- Single-threaded, fixed delay, honest UA (`sudani-lm-crawler; personal research use`),
  exponential backoff, hard stop after 15 consecutive failures.
- `robots.txt` is honoured, **including AI-specific directives**: any site whose robots
  disallows `ClaudeBot` / `anthropic-ai` / `GPTBot` / `CCBot`, or that sets Cloudflare
  `Content-Signal: ai-train=no`, is excluded even where the content is otherwise public and
  even where it is the single largest corpus available. `crawl-delay` is obeyed as written.
- Private, non-redistributed use only. Nothing scraped is republished; everything under `data/` except the sheets is gitignored.
- Sudan-topical MSA is kept but **down-weighted** — the char-ngram dialect classifier
  (`src/preprocessing/dialect_score.py`, 98.1% holdout accuracy) scores every document and the
  mixture takes a threshold, so bulk MSA cannot swamp the dialect it is mixed with.

**`is_scraped` values:** `yes` = crawl complete · `partial` = crawl running/incomplete ·
`queued` = approved, not started · `no` = evaluated and rejected (reason in Notes).

## The survey

| Site | Category | Measured scale | Dialect | Scrapeability | is_scraped | Notes / decision |
|---|---|---|---|---|---|---|
| **sudaneseonline.com** | forum | 72,761 of ~73,180 unique threads fetched (99.4%) → **69,633 threads / ~243M tokens**; 1.6 GB HTML | med (banded) | moderate | **yes** | Gate passed 2026-08-23. Needed per-line mojibake repair for the site's 2015 double-encoding. Classifier bands: ≥0.8 ≈ 5.8M tokens genuine dialect, 0.5–0.8 ≈ 13.8M dialect-seasoned; mixture takes **≥0.5 ≈ 19.6M tokens**. |
| **t.me/s/novelsforus2** | Telegram serial fiction | 598 pages, channel **complete**, 90 MB | **very high** | easy | **yes** | Static HTML preview, ~20 msgs/page, `?before=<id>` cursor, no robots restrictions (`t.me/robots.txt` = 404). Highest dialect density found anywhere. |
| **t.me/s/klam_sudany** | Telegram prose | 928 pages, **complete**, 15 MB | **very high** | easy | **yes** | Personal Sudanese prose/quotes. |
| **t.me/s/sudanesenovels** | Telegram serial fiction | 336 pages, **complete**, 47 MB | **very high** | easy | **yes** | Novel chapters, heavy dialogue. |
| **t.me/s/Sd_rewaya3t** | Telegram serial fiction | 60 pages, **complete**, 9.9 MB | **very high** | easy | **yes** | |
| **t.me/s/sudanes0** | Telegram misc | 31 pages, **complete**, 1.6 MB | high | easy | **yes** | |
| **t.me/s/Diwansha3r** | Telegram poetry / دوبيت | 0 pages | high | **unavailable** | **no (preview disabled)** | Verified 2026-08-28: `t.me/s/Diwansha3r` 302-redirects to the plain channel page with zero message widgets — the channel has its web preview disabled, so the `t.me/s/` method cannot reach it. Not a crawler bug. |
| **alnilin.com** (comments) | news comments | **561,752 comments**, complete (5,618 pages) | **very high** | easy | **yes** | Open WordPress REST (`/wp-json/wp/v2/comments`), clean JSON, `X-WP-Total` gives exact scale. Reader comments are where dialect lives. |
| **alnilin.com** (articles) | news | **550,832 articles**, complete (5,509 pages, 2.2 GB total with comments) | low (silver) | easy | **yes** | Via wp-json `posts` endpoint. Sudan-topical MSA, ranked down by the classifier. |
| **anasudani.net/forum** | forum (phpBB) | **1,161,442 posts / 52,973 topics / 26,496 members**; frozen ~Jan 2017 | high | easy | **partial** | Crawling. **No robots.txt at all** (404 → unrestricted). Discovery walks `viewforum.php` listings (paginates by 20 — stepping 40 in round 1 silently halved coverage; fixed, re-discovering), then per-topic fetch. |
| **sudanile.com** | news + opinion | **142,731 posts** (`X-WP-Total`) | low–med | easy | **yes** | Complete via `scrape_wpjson.py` — 1,428 pages ≈ 142,800 posts, 1.4 GB. **0 comments**. Columnists drift into dialect. |
| **wadmadani.com/vb** | forum (vBulletin) | thread IDs ≥ **41,443** | high | moderate | **no (now 403) → wayback** | Live site began returning **403 to our crawler UA** on 2026-08-28 — treated as refusal per policy, live crawl stopped. Rerouted to the Wayback miner (queued after the current pass). |
| **sudanyat.org/vb** (سودانيات) | dead forum | thread IDs ≥ **39,234**; **43** Wayback page-blocks | high | moderate | **partial** | Wayback mining running (`scrape_wayback.py`). Live site serves only "المنتدى تحت الصيانة". Contains a multi-page *مصطلحات عامّية سودانية* thread. |
| **mugrn.net/vb** (المقرن) | dead forum | now Sedo-parked; hosts "موسوعة الشعر الحلمنتيشي" | high | moderate | **partial** | Wayback mining running (`scrape_wayback.py`). |
| **algorer.net/vb** (القـريـر) | dead forum | thread IDs ≥ **20,071**; DNS dead | high | moderate | **partial** | Wayback mining running; indexed threads include أمثال/سيرة سودانية collections. |
| **sudanelite.com/vb** | dead forum | connection refused | high | moderate | **queued** | Wayback only. |
| **alhasahisa.org** | dead forum | DNS dead | high | moderate | **queued** | Wayback only. |
| **forum.kooora.com?f=132** | forum, football | Sudanese section of a large pan-Arab board, multi-page | high | moderate | **queued** | ⚠️ **Encoding quirk: `charset=windows-1256`** — must iconv; UTF-8 readers get mojibake. `f.aspx?f=132&pg=N`. |
| **`*.sudanforums.net`, `*.yoo7.com`, ahlamontada, 7olm** (shababkulkoal, watane, hausa, awladalfawo, eymoo, alhilalalsudan, alhelal, alhilal-sd, almoatn, abdo111, newsudan) | forum long tail | small: shababkulkoal **4,979 posts**, almoatn **900 posts** | high | easy | **queued** | Individually negligible, collectively worth a few million words. Uniform Forumotion HTML — one scraper covers the whole namespace. Low priority. |
| **Blogger cluster** — hageebatalfun (كلمات الحقيبة), sudaneseshortstorieswriters, sudanese-novels, katabsudsnese | lyrics / stories | small–mid, all reachable | **high** | easy | **yes** | Complete via Blogger Atom feeds — the four blogs are tiny (25–300 posts each); Haqiba lyrics captured. |
| **aghaniwamthal.com** | proverbs + lyrics | small (136 KB home) | **high, very clean** | easy | **yes** | Complete — BFS mirror, 2,039 pages / 54 MB. Token-for-token the densest dialect found. |
| **alsudaninews.com** (صحيفة السوداني) | news | **~5,302** listing pages ≈ 50K+ articles | low | easy | **queued** | WordPress, `?p=<id>`; no AI-crawler disallows found. Silver. |
| **dabangasudan.org** (ar) | news | **25,432** posts (all languages) | low | easy | **queued** | wp-json; human-rights reportage, MSA. Low priority. |
| **sudaress.com** | press aggregator | URL pattern `/{paper}/{sequential-id}`; **101** Wayback page-blocks | low | **currently down** | **no (revisit)** | Returned **HTTP 523** (Cloudflare: origin unreachable) on every probe including `/robots.txt` — we could not read its policy, so we cannot crawl it. Per-paper sequential IDs make it trivially enumerable *if* it returns. Re-check later; Wayback is the fallback. |
| **alsahafa.info · altayar.info · aljareeda.net · sudanhorizon.com · darfur24.com · sudanpost.info** | news | thin: sitemap indices carry only 4–27 children | low | easy | **no** | Alive but small MSA archives; not worth crawler time while sudanile and alnilin are unfinished. Revisit only if the silver tier runs dry. |
| **alrakoba.net** | news | large | low–med | — | **no (excluded)** | **Excluded on policy.** Explicit AI-crawler disallows / `ai-train=no`. |
| **vb.alrakoba.net** (منتديات الراكوبة) | forum (XenForo) | thread IDs ≥ **145,739** — the single largest live Sudanese forum found | **high** | blocked | **no (excluded)** | **Excluded on policy, despite being the #1 target by raw size.** Cloudflare robots sets `Content-Signal: search=yes, ai-train=no, use=reference` and explicitly disallows `ClaudeBot`, `GPTBot`, `CCBot`, `Google-Extended`, `Amazonbot`, `meta-externalagent`; returns 403 to non-browser clients. Note this is a *separate* forum from the alrakoba news domain. |
| **altaghyeer.info** | news + comments | **47,845 posts + 5,950 comments** | low / high (comments) | blocked | **no (excluded)** | **Excluded on policy.** robots disallows `ClaudeBot`, `GPTBot`, `anthropic-ai`; `ai-train=no`. Serves 403 to default UAs (a browser UA gets 200 — we do not spoof past an explicit refusal). |
| **sudanakhbar.com** | news aggregator | post IDs ~**1.83M** (sparse), 2017– | low | blocked | **no (excluded)** | **Excluded on policy.** robots disallows `ClaudeBot`, `Claude-Web`, `anthropic-ai`, `GPTBot`, `CCBot`, `PerplexityBot`. |
| **sudan-forall.org** (منبر الحوار الديمقراطي) | forum (phpBB) | active into Oct 2025; high-quality Sudanese intellectual prose | med | **JS-required** | **no** | Behind a Cloudflare **managed JS challenge** — even `/robots.txt` returns the interstitial, so its policy is unreadable and a crawl would mean defeating a bot check. Excluded. |
| **songs.alrakoba.net** (كلمات أغاني الحقيبة) | lyrics archive | node-per-song archive | **high** | blocked | **no** | 403 to bots; same operator as an `ai-train=no` domain. Substitute: hageebatalfun Blogger feed. |
| **sudanmirror.com · sudanarchive.net** | news / heritage archive | — | low | blocked | **no** | 403 to non-browser clients. sudanarchive is scanned PDFs anyway (OCR required). |
| **sudanmemory.org / Sudan Open Archive** | scanned heritage | large but image/PDF | low (historical MSA) | hard | **no** | Would need an OCR pipeline for historical-register MSA we do not need. Revisit only for a historical-register experiment. |
| **andariya.com/ar** | essay magazine | small (271 KB home) | low–med | easy | **no** | Good quality, negligible volume; bilingual. Not worth a bespoke scraper. |
| **alintibaha.net · hurriyatsudan.com · almashhadsudani.com · akhbaralwatan.net · sudantribune.net · alsudanalyoum.com · sudanafrit.com · baj.news · sudanjournal.com · merrikhabonline.net** | news / sports | dead at DNS or connection level | — | — | **no** | Off the live web. Only hurriyatsudan (44 Wayback page-blocks) has enough archive depth to be worth a future Wayback pass; the rest are not. |
| **r/Sudan** | social | — | **low** (mostly English) | hard | **no** | Anonymous `.json` endpoints now return the HTML shell on both `www` and `old.reddit.com`; would need OAuth. Poor dialect return for the effort. |
| **YouTube Sudanese channels (auto-captions)** | transcripts | — | high (speech) | moderate, noisy | **no** | Arabic ASR is MSA-biased and mangles Sudanese; would inject MSA-normalised pseudo-dialect. We already hold cleaner speech-derived text via oddadmix transcripts (2.49M tokens). |
| **Facebook / X public pages** | social | large | very high | hard | **no** | Auth walls and ToS; not attempted. |

## Discovery resources (for extending this table)

- **`sudan2.com/dir/site-NNNN.html`** — "بنك المواقع السودانية", sequential IDs verified to at
  least **1169**, ~15 KB per described site. Cheapest way to enumerate the long tail.
  Cert quirk: valid for apex `sudan2.com` only, **not** `www.sudan2.com`.
- `sudaneseonline.com/board/30/msg/-1361141364.html` — in-forum directory of 66+ Sudanese sites
  (mostly dead → Wayback candidates).
- `mtwersd.com/sudanese-blogs/` and `/bigger-sudanese-websites/` — curated Sudanese site lists.
- Telegram channel discovery: `telemetr.io/en/catalog/sudan`, `tgstat.com`,
  `dir-telegram.blogspot.com`; also `site:t.me` search with dialect markers
  (شنو، دايراك، كيفن، زهجت) works well. Each new channel is one line in `CHANNELS` —
  note that preview-disabled channels (302 from `t.me/s/`) are unreachable by this method.
- `waslat.com/Sudan` — directory, was returning 522 on survey day.
- Wayback CDX scale check used throughout:
  `http://web.archive.org/cdx/search/cdx?url=<domain>&matchType=domain&showNumPages=true`.

## Scrapers

| Script | Target | Method |
|---|---|---|
| `scripts/scrape_sudaneseonline.py` | sudaneseonline.com | sitemap → thread URLs → HTML, resumable |
| `scripts/scrape_telegram.py` | `t.me/s/<channel>` | static preview pages, `?before=<id>` cursor, one JSON per page-fetch |
| `scripts/scrape_alnilin.py` | alnilin.com | WordPress REST `posts` / `comments`, 100/page |
| `scripts/scrape_anasudani.py` | anasudani.net/forum | phpBB: listing-walk discovery (step 20) → per-topic fetch; 404 tombstones |
| `scripts/scrape_wpjson.py` | any open WP site (sudanile, …) | generic WordPress REST posts/comments |
| `scripts/scrape_blogger.py` | Blogger cluster | Atom feeds, full post bodies |
| `scripts/scrape_small_sites.py` | tiny sites (aghaniwamthal, …) | polite same-domain BFS, capped |
| `scripts/scrape_wayback.py` | 6 dead forum domains | CDX enumerate → snapshot fetch |
| `scripts/scrape_wadmadani.py` | wadmadani.com/vb | archive-page sweep at crawl-delay 60 |
| `scripts/crawl_watchdog.sh` | the fleet | 10-min health checks: death/stall detection |
| `scripts/download_sudanese_sources.py` | HF datasets | not a crawler — public dataset pulls |

Every crawler writes `data/raw/<source>/` plus a `crawl.log`, is resumable by page/cursor file,
and is single-threaded with a fixed delay.

## Priority order for the queue

1. Finish alnilin comments (the highest dialect yield still in flight), then alnilin posts.
2. Extend `CHANNELS` from the Telegram directories (skip preview-disabled channels).
3. `anasudani.net` full phpBB crawl — 1.16M posts, unrestricted, no competitor for value.
4. Wayback pass over the five dead vBulletin forums (uniform `archive/index.php/t-N.html`).
5. `wadmadani.com` at crawl-delay 60, Wayback backfill.
6. sudanile wp-json for bulk silver.
7. kooora Sudanese section (windows-1256), then the Blogger/Forumotion long tail.

---

Keep this file current when a site is surveyed, crawled, finished, or refused. Token counts,
preprocessing outcomes and mixture roles for anything landed go in `DATASHEET.md`.
