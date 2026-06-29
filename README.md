# E-commerce RFM Segmentation & Uplift Modeling

**Olist Brazilian E-commerce · Hillstrom MineThatData · DuckDB · XGBoost · Streamlit**

> "Which customers will respond to promotions — and who will buy anyway?"

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](STREAMLIT_URL_HERE)

---

## Core Findings

| Finding | Detail |
|---------|--------|
| **97% one-time buyers** | Olist is acquisition-heavy — F dimension nearly useless for loyalty scoring |
| **Promising = 37% revenue** | Largest segment by volume ($5.7M) — highest conversion opportunity |
| **At Risk avg $324/head** | Highest per-customer value, lowest recency — retention campaign priority #1 |
| **Naive ATE = +4.5pp** | Email appears to lift visit rate by 4.5pp across all customers |
| **Persuadables lift = +15.3pp** | T-Learner finds 3.4× higher lift in targeted segment vs naive ATE |
| **Lost Causes lift = −7.5pp** | 17.1% of base — email *reduces* their visit rate (Sleeping Dogs effect) |
| **Targeting efficiency** | Top 25% targeting = 4.8× more incremental visits per email vs blast all |

---

## Business Questions

**Module A (RFM):** "Who are my customers and how much are they worth?"
→ DuckDB SQL pipeline segments 93,358 Olist customers into 7 behavioral groups

**Module B (Uplift):** "Which customers will respond to promotions — and who will buy anyway?"
→ XGBoost T-Learner on Hillstrom RCT data identifies 4 uplift segments with heterogeneous treatment effects

**Framework:** Segmentation tells you *who has value* — Uplift tells you *who to target*. Two questions a marketing manager needs to avoid burning budget on the wrong people.

---

## Methodology

### Module A — RFM Segmentation (DuckDB SQL)

```
Raw transactions (9 CSV files)
    ↓ [DuckDB JOIN — customer_unique_id, not customer_id*]
Base transactions (delivered orders only)
    ↓ [SQL CTEs]
RFM features: Recency · Frequency · Monetary
    ↓ [NTILE(5) window functions]
RFM scores → 7 segments
    ↓
rfm_segments.csv (93,358 customers)
```

**Pipeline steps:**
1. JOIN `olist_orders` + `olist_order_items` + `olist_customers` on delivered orders
2. Compute Recency (days since last purchase), Frequency (distinct orders), Monetary (total spend)
3. Score R and M via `NTILE(5)` window functions; F scored from raw value due to 97% single-purchase rate*
4. Map RFM scores to 7 business segments via `CASE WHEN` logic

**\*Schema note:** Olist assigns a new `customer_id` per order — `customer_unique_id` must be used for correct repeat-purchase identification. Using `customer_id` yields Frequency = 1 for all customers (common trap).

**\*F score design decision:** With 97% of customers having frequency = 1, `NTILE(5)` on F produces arbitrary rankings (tie-breaking on identical values). F is scored directly from raw value (1→1, 2→2, 3→3, 4→4, ≥5→5) to reflect true loyalty signal.

### Module B — Uplift T-Learner (XGBoost)

```
Hillstrom MineThatData (RCT email campaign)
    ↓ [Filter: Womens E-Mail vs No E-Mail]
Treatment (N=21,387) | Control (N=21,306)
    ↓ [XGBoost T-Learner]
Model T: P(visit | treatment, X)
Model C: P(visit | control, X)
    ↓
Uplift score = Model_T(X) − Model_C(X) for all customers
    ↓
4 segments: Persuadables · Sure Things · Needs Attention · Lost Causes
    ↓
Qini curve · AUUC evaluation · ROI simulation
```

**T-Learner logic:** Train two independent models — one on treatment group, one on control. Apply both to all customers. The difference in predicted probabilities is the individual-level uplift score (estimated incremental effect of email).

**Treatment pair selection:** Womens E-Mail vs No E-Mail chosen for moderate treatment effect (4.5pp ATE) — sufficient to demonstrate heterogeneity without trivializing the uplift modeling task. Mens E-Mail shows stronger raw ATE (7.7pp) but is used as secondary analysis.

**Evaluation:** Qini curve (not standard AUC) — correct metric for uplift models, measuring cumulative incremental conversions when targeting by descending uplift score. AUUC = 812.87.

---

## Segment Definitions

### RFM Segments

| Segment | Condition | Customers | Revenue |
|---------|-----------|-----------|---------|
| Champions | freq ≥ 2, r_score ≥ 4 | 995 (1.1%) | $293K |
| Loyal | freq ≥ 2, r_score ≥ 2 | 1,185 (1.3%) | $370K |
| At Risk | freq ≥ 2, r_score < 2 | 621 (0.7%) | $201K |
| New | freq = 1, r_score = 5 (0–92 days) | 18,234 (19.5%) | $2.9M |
| Promising | freq = 1, r_score 3–4 (92–382 days) | 36,181 (38.8%) | $5.7M |
| Needs Attention | freq = 1, r_score = 2 (268–382 days) | 18,091 (19.4%) | $3.0M |
| Lost | freq = 1, r_score = 1 (382–713 days) | 18,051 (19.3%) | $2.9M |

### Uplift Segments

| Segment | Uplift score | Visit rate T | Visit rate C | Lift | Action |
|---------|-------------|-------------|-------------|------|--------|
| Persuadables | ≥ p75 (0.079) | 23.7% | 8.4% | **+15.3pp** | ✅ Send voucher |
| Sure Things | p50–p75 | 15.7% | 9.0% | +6.7pp | 💡 Save budget |
| Needs Attention | 0–p50 | 10.5% | 9.8% | +0.7pp | ⏸ Skip |
| Lost Causes | < 0 | 10.6% | 18.0% | **−7.5pp** | 🚫 Never target |

---

## Key Insights

### 1. Olist is an acquisition market, not a loyalty market
97% of customers purchase exactly once. The RFM Frequency dimension — typically the strongest loyalty signal — carries almost no discriminating power here. This requires redesigning the scoring logic to be recency-dominant rather than standard equal-weight RFM.

### 2. Revenue is volume-driven, not whale-driven
Pareto 80/20 does not apply to Olist: customer share ≈ revenue share across all segments. There are no high-value repeat customers dominating revenue. This means acquisition efficiency matters more than retention depth for this business model.

### 3. The Sleeping Dogs effect is real and costly
Lost Causes (7,294 customers) show a −7.5pp visit rate drop when receiving email, versus 18.0% organic visit rate in the control group. Sending to this group does not just waste voucher cost — it actively suppresses conversion. Blast-all strategy incurs double cost: voucher spend + foregone organic revenue.

### 4. Naive ATE understates targeting opportunity
Averaging treatment effect across all segments hides 3.4× heterogeneity. A marketing team using only aggregate metrics (ATE = +4.5pp) would systematically under-target Persuadables and over-spend on Lost Causes.

---

## Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| SQL / data layer | DuckDB 0.9+ | RFM feature engineering — window functions, CTEs, QUALIFY |
| ML | XGBoost 1.7+ | T-Learner treatment and control models |
| Preprocessing | scikit-learn | LabelEncoder for categorical features |
| Evaluation | NumPy | Qini curve, AUUC computation |
| Dashboard | Streamlit 1.28+ | 4-page interactive app |
| Language | Python 3.11 | — |

---

## Project Structure

```
ecommerce-rfm-uplift/
├── app.py                        # Streamlit dashboard (4 pages)
├── requirements.txt
├── README.md
├── .gitignore
├── app/
│   ├── rfm_segments.csv          # Module A output (93,358 rows)
│   ├── uplift_scores.csv         # Module B output (42,693 rows)
│   ├── xgb_treatment_model.pkl   # T-Learner Model T
│   ├── xgb_control_model.pkl     # T-Learner Model C
│   └── feature_cols.json         # Feature column order
├── notebooks/
│   ├── 01_olist_rfm_duckdb.ipynb # Module A pipeline
│   └── 02_hillstrom_uplift.ipynb # Module B pipeline
├── reports/
│   ├── rfm_segment_distribution.png
│   ├── rfm_revenue_contribution.png
│   ├── rfm_recency_monetary_scatter.png
│   ├── qini_curve.png
│   └── feature_importance.png
└── data/raw/                     # gitignored
    ├── olist_*.csv               # Olist Brazilian E-commerce (Kaggle)
    └── Kevin_Hillstrom_MineThatData.csv
```

---

## Methodology Note

Module A (Olist) and Module B (Hillstrom) use different datasets from different companies and time periods. This project is a **technique demonstration** — showing how RFM segmentation and uplift modeling work together conceptually as a marketing analytics framework.

RFM answers: *"Who are my customers?"*
Uplift answers: *"Who should receive a voucher?"*

In a real company setting, both analyses would run on the same customer base with internally conducted RCT data. The two-dataset structure here reflects a real constraint: companies that publish detailed transaction logs (Olist) rarely also publish RCT experiment data, and companies that publish RCT data (Hillstrom 2008) release aggregated snapshots rather than raw transaction logs.

---

## Datasets

- **Olist Brazilian E-commerce:** [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 100K+ orders, Sep 2016–Oct 2018
- **Hillstrom MineThatData:** [Kaggle](https://www.kaggle.com/datasets/yogiyo/hillstrom-email-marketing) — 64K customers, email marketing RCT (Kevin Hillstrom, 2008)

---

## Portfolio Context

| Project | Technique | Status |
|---------|-----------|--------|
| [Telco Survival CLV](https://github.com/ahnthwu-010/telco-survival-clv) | Cox PH + CLV | ✅ Live |
| [HR Causal Survival](https://github.com/ahnthwu-010/hr-causal-survival) | PSM + Cox PH + Bootstrap ATE | ✅ Live |
| **E-commerce RFM Uplift** | DuckDB SQL + T-Learner | ✅ Live |
