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
| **t.me/s/Diwansha3r** | Telegram poetry / دوبيت | 0 pages | high | **unavailable** | **no (preview disabled)** | Verified 2026-08-28: `t.me/s/Diwansha3r` 302-redirects to the plain channel page with zero message widgets — the channel has its web preview disabled, so the `t.me/s/` method cannot reach it. Not a crawler bug. Re-verified 2026-08-31 incl. the `?before=` bypass — still unreachable; `tatwer3` (real دوبيت) is the partial substitute. |
| **Telegram 2026-08-31 delta (11 channels)** — **nasra8ya** (عشاق الروايات, 33,973 msgs / **105.7M chars**), **rwayatSudan** (46,152 / 37.3M), w00057777 (9,802 / 7.7M, ad-diluted), sudan_4g (22,927 / 3.8M), comidyann (2.5M), sudanese_shair (1.9M), telegraaaammmmmmm (142 msgs / 0.6M — densest dialect measured, 6.3 markers/1k), sudanesevip, nikat7e3n, tatwer3 (دوبيت), SudaneseHD (أمثال) | Telegram fiction / poetry / proverbs / ونسة | **complete: 124,150 msgs / ~161M chars raw** (≈35–45M raw tokens pre-dedup), 470 MB, all channels paginated to id 1 same day | **very high** — verified samples per channel | easy | **yes** | The find of the final sweep — bigger high-dialect delta than everything else combined. Message counts < id-ceilings are channel deletions, not crawl gaps (every channel terminated at id ≤ 1). Found via **cross-promo lists inside channels** (the directory sites are all 403 now). Rejected after probing: storykaligi/ahgeel/oshaq_elrewayat/ROAYATE1 (**Egyptian**), amal14097y (promo spam), nabdalsudan (sports MSA), kuatrSD (MSA quotes), 9 preview-disabled channels. |
| **Common Crawl bolt-on** — historical WARCs of sudanyat/mugrn/algorer/wadmadani | dead-forum thread pages | **complete 2026-09-01: 94,512 pages / 1.2 GB** — sudanyat 33,666, **wadmadani 56,176** (the forum that 403'd our live crawler — CC's index held 57,173 unique thread urls, far beyond the pre-run 5–8k estimate), algorer 4,083, mugrn 587 | high | easy (Range GETs, unthrottled) | **yes** | `scripts/scrape_commoncrawl.py`: per-collection CDX, `filter==status:200` (many records 406 — forums refused CC's UA some years), byte-range fetch from data.commoncrawl.org, payloads raw windows-1256. wadmadani processed same day: 3,323 unique-content docs / 13.2M chars after cross-capture dedup. kooora NOT recoverable this way (2 CC blocks, generic pan-Arab threads only). |
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
| **forum.kooora.com?f=132** | forum, football | Sudanese section of a large pan-Arab board, multi-page | high | dead | **no (forum removed)** | 2026-08-31 recheck: kooora redesigned; `?f=132` now 301→`/404`. Wayback recovery impractical — CDX regex filters over the whole (huge) domain 504 server-side; would need full domain pagination for football-fan chatter. Skipped. |
| **`*.sudanforums.net`, `*.yoo7.com`, ahlamontada, 7olm** (shababkulkoal, watane, hausa, awladalfawo, eymoo, alhilalalsudan, alhelal, alhilal-sd, almoatn, abdo111, newsudan) | forum long tail | small: shababkulkoal **4,979 posts**, almoatn **900 posts** | high | easy | **EXCLUDED (robots)** | 2026-08-31 recheck: the Forumotion platform robots.txt (served identically on every `*.sudanforums.net`-style host) lists `anthropic-ai`, `ClaudeBot`, `Claude-Web`, `CCbot` (and dozens more bots) with `Disallow: /`. Whole namespace excluded per policy. |
| **Blogger cluster** — hageebatalfun (كلمات الحقيبة), sudaneseshortstorieswriters, sudanese-novels, katabsudsnese **+ 2026-08-31 delta:** unothati, olive2020, ajba77, sudanesemollified, salahamza2, ar-cher, 22montser2019, trendsudani | lyrics / stories / personal, women's & cooking blogs | 12 blogs, ~30 MB Atom XML total (unothati alone 577 posts / 9.7 MB) | **high** | easy | **yes** | Complete via Blogger Atom feeds. Delta found via mtwersd.com/sudanese-blogs/; personal/cooking registers are colloquial. |
| **aghaniwamthal.com** | proverbs + lyrics | small (136 KB home) | **high, very clean** | easy | **yes** | Complete — BFS mirror, 2,039 pages / 54 MB. Token-for-token the densest dialect found. |
| **koorasudan.net** (كورة سودانية) | sports news + **comments** | **130,484 posts + 75,599 comments**, complete (1,305 + 756 wp-json pages, 518 MB) | posts low, **comments high** — football fans write the way they talk | easy | **yes** | Found 2026-08-31 via directory sweep; crawl complete same day. robots.txt is a bare `User-agent: *` allow-all; open wp-json. Best find of the directory sweep — ~⅛ of alnilin's comment volume. |
| **cover-sd.com** (صحيفة كفر ووتر) | news | **1,847 posts** (`X-WP-Total`) | low | easy | **partial (crawling)** | Found 2026-08-31. Clean robots, open wp-json; opportunistic 20-minute crawl. |
| **sudanesesongs.net** (مكتبة الأغنية السودانية) | dead lyrics **forum** (IPB) | **58 Wayback page-blocks** — deeper than sudanyat (43) or hurriyatsudan (44) | **high** — lyrics + fan discussion | moderate | **partial (wayback)** | Found 2026-08-31; DNS-dead. Added to `scrape_wayback.py` DOMAINS with new IPB URL patterns (`showtopic`, `lofiversion`); dedicated miner instance running. |
| **sudancam.net** | dead news site | 35 Wayback page-blocks | low | moderate | **no (low priority)** | Wayback-minable but silver-tier news; only if that tier runs dry. |
| **alsoug.com** (سوق السودان) | classifieds | largest Sudanese classifieds site | med but short/templated ad text | **JS-required** | **no** | Robots clean, but the site is an SPA with no server-rendered listing links — needs headless rendering for low token density. Revisit only if we ever build JS rendering. |
| **sammaniya.org** | Sufi order / مديح | — | — | — | **no** | 2026-08-31: the domain just 301-redirects to a Facebook page. Nothing to crawl. |
| **alsudaninews.com** (صحيفة السوداني) | news | **~5,302** listing pages ≈ 50K+ articles | low | easy | **queued** | WordPress, `?p=<id>`; no AI-crawler disallows found. Silver. |
| **dabangasudan.org** (ar) | news | **25,432** posts (all languages) | low | easy | **queued** | wp-json; human-rights reportage, MSA. Low priority. |
| **sudaress.com** | press aggregator | URL pattern `/{paper}/{sequential-id}`; **101** Wayback page-blocks | low | **origin down** | **no (revisit)** | 2026-08-31 recheck: apex now serves robots.txt from the Cloudflare edge — it is the Content-Signal *preamble only*, no signal line, no Disallow (nothing restricts us) — but the site itself is still dead: apex 301→www, www origin 523. Aggregator of alnilin/sudanile-class news → mostly duplicate MSA; low value even if it returns. |
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
| **archive.org Arabic-Sudan texts** | OCR'd books | 3,765 Sudan matches but only ~16 Arabic+Sudan items with `_djvu.txt` layers | low (historical-political MSA monographs) | easy | **no (settled 2026-08-31)** | Same register we declined at sudanmemory.org, plus severe Arabic OCR noise. Dialect-term queries (أمثال سودانية etc.) return zero. No Archive Team WARCs of Sudanese forums exist. |
| **HF `muzammilsoft/Sudanese_dialect_dataset`** | **synthetic** instruction data | 13.6 MB (2 jsonl files), CC-BY-4.0, published 2026-08-27 | very high but **LLM-generated** | easy | **yes (downloaded)** | New since our HF survey. Downloaded to `data/raw/hf_muzammilsoft/` for evaluation against our own synth QC — belongs to the synthetic pool (SYNTHSHEET), never to the organic web corpus. |
| **HF `ArSyra/arsyra-sudanese`** | expert-written dialect | unknown | likely high | **gated (401)** | **no** | Needs a HF access request from the account owner — user action if wanted. |
| **HF `O96a/sudanese-mt-benchmark`** | eval set | <1K rows | high | easy | **no — hold out** | Benchmark data (arXiv:2507.20301). Keep OUT of training like Flores DEVTEST. |
| **fnanen.com · Wikisource · Wattpad · Bluesky/Mastodon · Sudanese podcasts RSS** | misc | — | — | — | **no (settled 2026-08-31)** | fnanen: unrestricted robots but no Sudanese section. Wikisource: 1930s مجلة الرسالة MSA, negligible. Wattpad: crawlable but Arabic writing there is Egyptian/Gulf — Sudanese fiction lives on Telegram. Bluesky/Mastodon: negligible presence. Podcasts: audio-only feeds, no transcripts; YouTube `@alsudanpodcast` captions all `kind=asr` (auto) — confirms the existing ASR exclusion. |

## Discovery resources (for extending this table)

- **`sudan2.com/dir`** — **EXHAUSTED 2026-08-31.** Real pagination is `dir/orderbylast-{1..19}.html`;
  the bank holds only **73 entries** total (IDs sparse to 8289 — unlisted IDs serve a generic shell),
  mostly 2010-era angelfire/geocities personal pages. Nothing further here.
- ~~`sudaneseonline.com/board/30/msg/-1361141364.html`~~ — **mislabelled, worthless**: not a site
  directory but one user's parked-domain sale portfolio (~90 domains, all dead, ≤1 Wayback block each).
- `mtwersd.com/sudanese-blogs/` — **harvested 2026-08-31** (8-blog Blogger delta, done);
  `/bigger-sudanese-websites/` — pure overlap with this table, nothing new.
- Telegram channel discovery — **updated 2026-08-31**: the directory sites are dead to us
  (telemetr.io and tgstat 403 even to browsers/WebFetch; telegram-store 404; dir-telegram
  has no Sudan links). The vector that actually works is **cross-promotion lists posted
  inside Sudanese channels themselves** (that is how nasra8ya, the largest find, surfaced)
  plus `site:t.me` searches with dialect markers (شنو، دايراك، كيفن، زهجت). Each new channel
  is one line in `CHANNELS`; preview-disabled channels (302 from `t.me/s/`) are unreachable.
  ⚠️ Verify dialect before crawling: several huge "روايات" channels are Egyptian
  (storykaligi, ahgeel, oshaq_elrewayat, ROAYATE1 — all rejected).
- `waslat.com/Sudan` — back up 2026-08-31 but **EXCLUDED**: `Content-Signal:
  search=yes, ai-train=no, use=reference` plus explicit `Disallow: /` for ClaudeBot/GPTBot/CCBot.
  Do not retry.
- Dead-forum long tail is **unarchived**, not just dead: shatyalnail.net, nbdalsudan.com,
  helatomar.net, chatsudan.com, albahala.com, alsagia.com, orbinanet.com, igdelgalad.net all
  carry 1–6 Wayback page-blocks (vs 43–58 for the forums we mine). No recovery possible.
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
| `scripts/scrape_commoncrawl.py` | CC WARCs of 4 dead forums | per-collection CDX index → byte-range GETs |
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
7. ~~kooora Sudanese section~~ (forum removed from live site, 2026-08-31) and ~~Forumotion long tail~~ (platform robots blocks AI crawlers — excluded, 2026-08-31).

---

Keep this file current when a site is surveyed, crawled, finished, or refused. Token counts,
preprocessing outcomes and mixture roles for anything landed go in `DATASHEET.md`.
