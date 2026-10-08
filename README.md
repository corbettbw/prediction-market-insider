# Detecting Informed Trading on Polymarket

Unsupervised machine learning pipeline to identify prediction markets structurally attractive to informed traders, validated against publicly documented insider trading cases.

**[→ Interactive Dashboard](https://portfolio.corbettwinningham.com)**

---

## Overview

Prediction markets depend on information symmetry to function correctly. When a trader has private information unavailable to other participants, they can exploit that asymmetry for guaranteed returns — at the expense of liquidity providers and uninformed traders.

This project uses K-means clustering and Isolation Forest anomaly detection to identify which markets show structural signatures of informed trading, without being told which markets were actually targeted. The methodology is validated by checking whether known insider trading cases concentrate in the high-attractiveness cluster.

---

## Known Cases

The following documented cases of informed trading on Polymarket were used to validate the model:

| Market | Bettor | Notes |
|---|---|---|
| [Will D4vd be the #1 searched person on Google this year?](https://polymarket.com/event/1-searched-person-on-google-this-year/will-d4vd-be-the-1-searched-person-on-google-this-year) | AlphaRaccoon | A Google employee used insider knowledge of Year End Search Rankings to bet that musician D4vd would appear in the top results. |
| [Will Maduro be out of power by January 31, 2026?](https://polymarket.com/market/will-maduro-be-out-of-power-by-january-31-2026) | Van Dyke | US Army soldier Gannon Ken Van Dyke allegedly used classified information about planned military operations to bet on outcomes including the capture of Nicolas Maduro. |
| [US forces in Venezuela by January 31, 2026?](https://polymarket.com/market/us-forces-in-venezuela-by-january-31-2026) | Van Dyke | See above. |
| [Israel military action against Iran before July?](https://polymarket.com/event/israel-military-action-against-iran-before-july) | ricosuave666 / rundeep | An Israeli military reservist and a civilian were indicted for using classified information to bet on Israeli military operations against Iran. |
| [Israel strike on Iran on June 24?](https://polymarket.com/market/israel-strike-on-iran-on-june-24) | ricosuave666 / rundeep | See above. |
| [Israel strikes Iran by January 31, 2026?](https://polymarket.com/market/israel-strikes-iran-by-january-31-2026) | ricosuave666 / rundeep | See above. |
| [Israel military action against Iran by Friday?](https://polymarket.com/event/israel-military-action-against-iran-by-friday-477) | ricosuave666 / rundeep | See above. |
| [Will George Santos attend the 2026 State of the Union address?](https://polymarket.com/market/will-george-santos-attend-the-state-of-the-union) | George Santos | Santos allegedly announced he would attend, then announced he could not, while simultaneously placing No bets. |

---

## Data

**Source:** Polymarket's [Gamma API](https://gamma-api.polymarket.com) for market metadata; [Dune Analytics](https://dune.com) for trade-level data (wallet concentration, trade timing, price change rates).

**Date range:** Markets that closed between June 2025 and March 1, 2026 (`closed_time`).

**Funnel:**

| Stage | Markets |
|---|---|
| Raw pull from Gamma API | 7,623 |
| After dropping missing Dune data | 6,590 |
| After feature engineering (missing opening price) | 5,822 |
| After excluding crypto, esports, NFL Playoffs | 2,310 |

Crypto price markets, esports, and NFL Playoffs are excluded because they exhibit mechanical behavioral similarity to informed trading signals — high volume, large trades, concentrated wallets — due to market structure rather than information asymmetry.

---

## Features

Six features grounded in the academic literature on informed trading:

| Feature | Description | Literature |
|---|---|---|
| `uncertainty` | Deviation from fair even-split price at open. Corrected for market structure: binary markets use 0.5; negRisk multi-outcome markets use 1/n_outcomes. | Kyle (1985) |
| `log_trading_intensity` | Log of average dollars traded per hour. Log-transformed to reduce right skew. | Fishe & Robe (2004) |
| `log_avg_trade_size` | Log of mean dollars per trade. Log-transformed for the same reason. | Barclay & Warner (1993) |
| `late_capital_ratio` | Fraction of total volume entering in the final 48 hours. | Kyle (1985) |
| `top3_wallet_concentration` | Fraction of volume controlled by the three most active wallets. | Holden & Subrahmanyam (1992) |
| `velocity_ratio` | Ratio of late-period to early-period hourly price change rate. | Kyle (1985) |

---

## Results

### Clustering

K-means (k=2) separated 2,310 markets into two clusters. k=2 is theoretically motivated: markets are either structurally attractive to informed traders or they are not.

| Feature | Cluster 0 — Low | Cluster 1 — High | Cohen's d |
|---|---|---|---|
| Uncertainty at open | 0.247 | 0.442 | 0.546 (medium) |
| Trading intensity ($/hr) | 261 | 12,298 | 1.871 (very large) |
| Avg trade size ($) | 43 | 181 | 0.858 (large) |
| Late capital ratio | 0.310 | 0.516 | 0.574 (medium) |
| Top 3 wallet concentration | 0.833 | 0.589 | −2.019 (very large) |
| Velocity ratio | 1.064 | 1.182 | 0.424 (moderate) |

Cluster 1 scored higher on 5 of 6 features. The exception — wallet concentration — is higher in Cluster 0 because thin markets with few participants naturally concentrate volume among a small number of wallets, not because of informed trading.

Silhouette score: **0.248** — moderate separation, consistent with a continuous underlying distribution rather than two discrete natural groups.

### Validation

All 8 documented insider trading cases landed in Cluster 1 after a blind run.

### Isolation Forest

Running Isolation Forest on Cluster 1 (contamination=0.05) flagged 53 anomalous markets. Only 1 of the 8 known cases was captured (d4vd), indicating that known insider traders operated within normal parameters for high-attractiveness markets rather than at structural extremes. The Isolation Forest is better understood as identifying markets with unusually extreme feature combinations.

Notable flagged markets beyond the known cases:
- **Tesla FSD launch** — most anomalous; velocity ratio 5.3, avg trade size $832
- **Melania "Career" speech** — 98% late capital ratio, avg trade size $2,257
- **Trump speech markets** — consistent with the Gabriel Perez teleprompter case (under investigation as of mid-2026)
- **Trump diplomatic visit markets** — velocity ratios 2.4–3.8 across multiple countries
- **Poland and South Korea presidential elections** — high volume, large trades, elevated velocity ratios

### Feature Weights

Logistic regression, random forest, and permutation importance all produced the same ranking: `top3_wallet_concentration`, `log_trading_intensity`, and `log_avg_trade_size` form a leading group, followed by `late_capital_ratio`, `uncertainty`, and `velocity_ratio`. The three leading features are correlated with each other (|r| = 0.50–0.74) but only weakly to moderately correlated with raw market volume and trade count (|r| ≤ 0.38), so the clustering does not appear to be a pure market-size sort.

---

## Limitations

- Only 8 confirmed cases — too few for supervised learning; validation is qualitative rather than statistical
- Late capital ratio does not distinguish between "late because insiders knew" and "late because outcome became publicly obvious near resolution"
- Isolation Forest threshold (contamination=0.05) is conventional, not derived from a natural break in the score distribution — markets scoring below −0.05 are more robustly anomalous than the full 53
- Wallet-level analysis required to confirm individual insider behavior (Phase 2)

---

## Notebooks

| Notebook | Description |
|---|---|
| `notebooks/01_data_collection.ipynb` | Pulls market metadata from Gamma API, queries Dune for trade-level features, saves raw parquet |
| `notebooks/02_phase1_market_clustering.ipynb` | Feature engineering, K-means clustering, Isolation Forest, validation against known cases |

---

## Repository Structure

```
prediction-market-insider/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   └── 02_phase1_market_clustering.ipynb
├── data/
│   ├── raw/
│   │   └── markets_polymarket.parquet
│   └── processed/
│       ├── features_stage1.parquet
│       └── cluster_assignments.parquet
└── outputs/
    ├── clusters_pca.png
    ├── cluster_profiles.png
    └── isolation_forest_pca.png
```

---

## Next Steps

- **Phase 2:** Wallet-level network analysis on flagged markets
- Build a larger confirmed case set to enable supervised learning
- Include additional prediction market platforms (Kalshi) for cross-platform comparison
- More granular trade features: frequency, outlying individual trades, timing signals

---

## References

- Kyle, A.S. (1985). Continuous Auctions and Insider Trading. *Econometrica*, 53(6), 1315–1335.
- Barclay, M.J. & Warner, J.B. (1993). Stealth Trading and Volatility. *Journal of Financial Economics*, 34(3), 281–305.
- Holden, C.W. & Subrahmanyam, A. (1992). Long-Lived Private Information and Imperfect Competition. *Journal of Finance*, 47(1), 247–270.
- Fishe, R.P.H. & Robe, M.A. (2004). The Impact of Illegal Insider Trading in Dealer and Specialist Markets. *Journal of Financial Economics*, 71(3), 461–488.