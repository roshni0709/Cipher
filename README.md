# Cipher

*Sahi daam, every time.*

**Find the same product across Indian e-commerce sites, then compare the price you actually pay.**

Titles differ ("boAt Airdopes 141 Bluetooth Truly Wireless..." vs "boAt Airdopes 141 with 42 Hours Playback Bluetooth Headset"), sellers reuse photos, and the listed price hides shipping, coupons and return risk. Cipher matches listings across sites with multimodal embeddings, ranks offers by true landed cost, detects fake discounts from price history, and advises buy-now-or-wait.

Built as a portfolio project: every phase below ships something visible.

## Quickstart (5 minutes, no downloads, no API keys)

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+
pip install -r requirements.txt pytest
# Optional, for the embedding + CLIP matchers (~2 GB):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   pip install -r requirements-ml.txt
python -m pricematch.pipeline seed                  # sample catalog + 60 days of prices
python -m pricematch.pipeline eval --compare        # TF-IDF vs embedding matcher
python -m pricematch.pipeline agent "boAt airdopes under 1500"
pytest -q
```

## The intelligence layer (all offline, all tested)

Ten capabilities sit on top of the matcher. Each is a small, explainable module you can speak about in an interview; none needs an API key.

| Feature | Module | How it works | Test file |
| --- | --- | --- | --- |
| RAG agent tab | `pricematch/rag/` | Retrieves offer + price-intel chunks (TF-IDF, MiniLM, or FAISS), writes an answer grounded ONLY in those chunks with [n] citations. Optional FLAN-T5 writer passes through a quality gate (length, digit presence, not-verbatim); weak output falls back to the template writer automatically. | `tests/test_rag.py` |
| Agentic planner | `pricematch/agent/planner.py` | Transparent rule planner: parses your budget, picks which tools to call (comparison, history, fake-discount, timing, alerts, reviews) and records a plan trace the UI shows. Tools register in a plain registry. | `tests/test_planner.py` |
| FAISS vector retrieval | `pricematch/rag/index.py` | FAISS flat index over chunk embeddings; the retriever auto-resolves faiss → minilm → tfidf so the demo runs with zero ML deps. | `tests/test_faiss_index.py` |
| Review summarizer | `pricematch/review_intel.py` | Extractive TF-IDF-centroid ranking picks the 3 most representative pro and con sentences per product. Deterministic, no model weights. | `tests/test_review_intel.py` |
| Fake-review detector | `pricematch/review_intel.py` | Trust score 0-100 from three signals: 5-star bursts in a 7-day window, copy-paste near-duplicate text across authors, one-sided ratings. Two sample products carry deliberate manipulation patterns so the demo has something to catch. | `tests/test_review_intel.py` |
| Price forecasting | `pricematch/forecast.py` | OLS on trend + weekly seasonality (sin/cos of the 7-day cycle), residual-based bands widened by sqrt(h) for h days ahead. The chart labels it a projection, not a promise. | `tests/test_forecast.py` |
| Verdict explainability | `pricematch/buy_or_wait.py` (`explain`) | The buy-or-wait model is logistic regression, so feature contributions are exactly coefficient x value (what SHAP reduces to for linear models, with no shap dependency). The UI shows the top contributions in plain English. | `tests/test_explainability.py` |
| Price-drop alerts | `pricematch/alerts.py` | Flags a drop when the latest price is >=3% below its trailing 30-day average AND a new tracked low, or when it crosses your target. Renders an email-ready digest. | `tests/test_alerts.py` |
| Photo-to-price | `pricematch/photo.py` | CLIP (clip-ViT-B-32) embeds your photo and the bundled catalog photos, cosine-ranks clusters, and refuses to guess below a confidence threshold. Matched cluster flows into the normal landed-cost + verdict pipeline. Without the ML extras it degrades gracefully to a notice. | `tests/test_photo.py` |
| Docker + CI | `Dockerfile`, `.github/workflows/ci.yml` | One-container demo (`docker build -t pricematch . && docker run -p 8501:8501 pricematch`); CI runs the suite on Python 3.10/3.11 plus a Docker build and in-container smoke test. | CI |

Catalog photos in `pricematch/data/photos/` are the actual products' manufacturer or retailer listing images (one to six views per covered SKU), with per-file sources in `pricematch/data/photos/SOURCES.md`. No stand-ins: a photo either shows the real SKU or the cluster has no photo. Matching is calibrated (softmax confidence + an absolute similarity floor over per-view max cosine); below the gates the app says "visually similar products" and shows no price or verdict, so it never puts a confident price on the wrong product.

## What is verified vs scaffolded

Verified (run by the author of this repo; the harness splits entity-disjoint where the data allows it, i.e. no product cluster appears in both train and test):

| Component | Data | Split | Result |
| --- | --- | --- | --- |
| TF-IDF baseline | WDC computers train-small, first 2000 pairs | 1152 train / 332 test, entity-disjoint | F1 0.808, precision 0.803, recall 0.814, top-1 0.981 |
| sentence-transformers (all-MiniLM-L6-v2) | WDC computers train-small, first 2000 pairs | same entity-disjoint split | F1 0.791, top-1 0.977. Yes, slightly BELOW TF-IDF here: the fancier model is not automatically better, which is exactly why the harness exists. |
| CLIP multimodal (ViT-B-32) | bundled image-eval smoke set (8 products, 2 photos each, from Wikimedia Commons) | stratified pair split, too small for entity-disjoint | runs end to end, F1 1.0 on the 10-pair test split, image coverage 1.0. Smoke test only: proves the pipeline, claims nothing. |
| Collector | local fixture server (robots.txt + JSON-LD page) | end-to-end test | robots honored, product extracted, offer + snapshot stored |
| Landed cost, fake-discount, buy-or-wait, DB, planner, RAG, reviews, forecasting, alerts, photo search plumbing | unit tests + seeded sample DB | `pytest -q`, 92 tests | pass |

Still pending (code complete and smoke-tested, real numbers not yet earned):

- **A publishable multimodal result**: the image-eval set is 8 products. A real CLIP number needs image-covered data at scale (Amazon-Berkeley Products, or your own hand-labeled pairs with downloaded photos), an entity-disjoint split, and image coverage reported. The harness refuses to emit a multimodal score on data with zero image coverage, on purpose.
- **Buy-or-wait**: fitted on synthetic histories so it runs out of the box; retrain on real collected history before trusting it.
- **Collector on live sites**: verified end-to-end against a local fixture server, plus live single-page probes (Nov 20, 2026): Croma and Vijay Sales product pages yielded real prices; Amazon.in serves bot-tailored HTML with no readable price markup (NO DATA); Flipkart answers scripted fetches with HTTP 429. Other sites need your own careful one-URL tests.

If you put any metric on your resume, it must come from a run you did, with the dataset and split named.

## The demo UI

```bash
pip install -r requirements.txt
python -m pricematch.pipeline demo
```

Search a product, see matched offers ranked by landed cost, its 60-day price chart, the fake-discount verdict, and the buy-or-wait call. The agent tab answers questions like "redmi 13c under 10000" offline; set `OPENAI_API_KEY` (and `pip install openai`) and it switches to a real LLM tool-calling agent.

First run seeds the sample catalog into `pricematch.db` (created next to the folder you launch from; about 10 seconds, one time). The header chips are computed after seeding finishes, so a fresh start always shows the full catalog: 100 products across 17 stores. If the chips ever show fewer, you are looking at an older app instance still running on the same port or a stale `pricematch.db`: stop every streamlit terminal, delete `pricematch.db`, hard-refresh the browser (Ctrl+Shift+R), and start again. The app also reseeds automatically when it finds a DB from an older catalog version (the old file is kept as `pricematch.stale-<timestamp>.db`).

**After applying any patch:** replacing files while the app is running is not enough. Streamlit re-runs the visible script but keeps the replaced Python modules cached in the old process, so the app can keep serving the old catalog and skip the reseed check. Fully stop EVERY streamlit terminal (Ctrl+C or close the windows), start the app once more, then hard-refresh the browser. If the app ever shows a "Catalog mismatch" warning, that is exactly this situation.

### Deploy it free on Streamlit Community Cloud

The repo is ready as-is: root `requirements.txt` carries the demo deps, and the app seeds its own database on first start, so a fresh deploy works with no setup. Main file path to enter on share.streamlit.io: `pricematch/demo/app.py`. Step-by-step is in the plan page / your notes; the deployed demo runs on the bundled sample data until your collector fills real history.

## Repo map

| Path | Phase | What it is |
| --- | --- | --- |
| `pricematch/eval/` | 0 | The harness: precision/recall/F1 with threshold tuning + top-1 accuracy. Everything is judged here. |
| `pricematch/matching/tfidf.py` | 0 | Baseline text matcher. The number to beat. |
| `pricematch/matching/embeddings.py` | 1 | sentence-transformers matcher. |
| `pricematch/matching/multimodal.py` | 1 | CLIP image + text embeddings, late fusion with a learned weight. The centerpiece. |
| `pricematch/index/` | 1 | FAISS index with an exact NumPy fallback. |
| `pricematch/db/` | 2 | SQLite: offers + daily price snapshots. |
| `pricematch/collector/` | 2 | robots.txt-respecting snapshot collector; JSON-LD Product extraction. |
| `pricematch/landed_cost.py` | 2 | price + shipping - coupon, with return-policy risk. |
| `pricematch/fake_discount.py` | 3 | Flags "price inflated before the sale" from history. |
| `pricematch/buy_or_wait.py` | 4 | P(price drop in 14 days): features + logistic regression. |
| `pricematch/agent/` | 5 | Tool-calling LLM agent + offline rule agent fallback. |
| `pricematch/demo/` | 5 | Streamlit UI. Deploy this, not the collector. |

## Data

**Training and evaluation** use existing benchmarks; do not scrape at scale to train.

- WDC Large-Scale Product Corpus v2, the standard product-matching benchmark: https://webdatacommons.org/largescaleproductcorpus/v2/
  The eval harness downloads it for you: `python -m pricematch.pipeline eval --dataset wdc --category computers --size small --compare`
- More categories: cameras, shoes, watches, all. Amazon-Berkeley Products (images + text) suits the multimodal phase: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
- Entity-resolution classics (Abt-Buy etc.): https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution/

**Live data** comes from `pricematch.collector.snapshot`: it fetches only URLs you list, respects robots.txt, waits between requests and identifies itself. Run it daily to accumulate real price history. Two affiliate APIs exist (Flipkart Affiliate API, Amazon Product Advertising API) but both require approved affiliate accounts with sales activity, so they are a later option, not a dependency. Verify their current terms at signup.

**Anti-bot reality:** Amazon and Flipkart aggressively block datacenter IPs and scripted browsing. The demo runs on collected snapshots, never live scraping, so it cannot break in front of a recruiter. Collect slowly, from a residential connection, a handful of pages at a time, or skip Amazon/Flipkart entirely and use sites with lighter protection. Public product data scraping sits in a grey zone under Indian law (site ToS, IT Act s.43); stay public, stay slow, respect robots.txt, and keep it non-commercial.

## How matching is judged

- **Precision / recall / F1** at a threshold tuned on a train split, reported on a held-out test split.
- **Top-1 accuracy**: for each product, the highest-scoring candidate must be a true match. This is the number the demo lives or dies by.
- **Entity-disjoint splits only** (the harness does this for you): a random pair split puts the same product on both sides and inflates every score.
- **The multimodal claim**, once measured, is F1(multimodal) - F1(text-only baseline) on the same entity-disjoint split, with image coverage reported. Hard negatives matter: sellers reuse photos, so include same-image/different-product pairs in eval or the model wins on images alone.

## Roadmap (10-14 weeks at ~8-10 hrs/week)

1. **Week 1** - baseline matcher + eval numbers on WDC (done for you; reproduce them).
2. **Weeks 2-5** - embeddings + CLIP multimodal matcher; beat the baseline; hand-label 200-500 Indian e-commerce pairs as your domain test set.
3. **Weeks 5-6** - DB + daily collector; landed-cost ranking on real offers.
4. **Weeks 7-8** - fake-discount detection on your accumulated history.
5. **Weeks 9-10** - buy-or-wait retrained on real history.
6. **Weeks 11-12** - LLM agent polish + deployed demo + README GIF.

Already done for you in this repo: the CLIP matcher and the collector are built, wired into the eval harness (`--matcher multimodal --dataset image-eval`), and covered by tests. Your real work is the honest kind: collecting data and earning publishable numbers.

Start daily price collection in week 1 even before the models exist: fake-discount detection and buy-or-wait need weeks of history, and you cannot backfill time.

## Portfolio checklist

- Pin this repo; README leads with a 60-second demo GIF and the live demo URL.
- Deploy the demo (AWS free tier, or Vercel + Render). It runs on a frozen snapshot so it cannot break live.
- Build over weeks with real commits; recruiters check.
- Resume bullets (fill your numbers): "Built a multimodal product-matching engine (CLIP + text embeddings, FAISS) reaching 0.XX F1 on the WDC benchmark, +X over a TF-IDF baseline" / "Designed landed-cost ranking and fake-discount detection over N weeks of collected price history."
