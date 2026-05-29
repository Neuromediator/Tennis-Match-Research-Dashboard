# Methodology — honest evaluation

This is a dashboard people will look at and form expectations from. A tool that misleads its users is worse than no tool at all. We optimise for *trustworthy* probabilistic predictions, not for "look how high our accuracy is."

## Why calibration matters more than accuracy

A predictor that says "Player A wins with probability 0.55" should be right roughly 55% of the time across many such predictions. That's calibration. Accuracy alone is a poor lens — a model that always picks the higher-ranked player gets ~65% accuracy in tennis just from rank ordering, while being badly miscalibrated. We report accuracy but never lead with it.

## Headline metrics

| Metric | Why |
|---|---|
| **Brier score** | Mean squared error between predicted probability and outcome. Standard scoring rule for probabilistic forecasts. Lower is better. |
| **Log loss** | Penalises confident wrong predictions more heavily. Sensitive to calibration. |
| **Calibration plot** | Bins predictions by predicted probability, plots empirical win rate vs predicted. Visualises where the model is over- or under-confident. |
| Accuracy | Reported, not headlined. |
| Accuracy by ranking gap | Sanity check that the model isn't only good on lopsided matchups. |

## Walk-forward validation

Tennis data is a time series. Random k-fold cross-validation leaks future information into past predictions and produces overconfident metrics.

Walk-forward: train on years up to `Y-1`, calibrate on `Y-1`, validate on `Y`, advance `Y`, repeat. Every reported metric is computed on data the model has never seen, in the order the model would encounter it in production.

## The market is the benchmark, not a feature

Bookmaker closing prices aggregate information no individual model can match. If we trained on them, our "performance" would be a lower bound on what the market already provides — meaningless as evaluation.

The market shows up in two places:
- **Historical:** tennis-data.co.uk closing prices in `market_implied_probabilities`, overlaid on every calibration plot.
- **Live:** The Odds API pre-match odds in `pre_match_odds`, rendered next to the model on the Match dashboard so the user sees the gap at the per-match level, not just in aggregate.

Average Brier last-5-fold (post-calibration):

| Tour | Surface-Elo baseline | LightGBM v3 | Market |
|---|---|---|---|
| ATP | 0.2220 | 0.2087 | ~0.20 |
| WTA | 0.2180 | 0.1959 | ~0.20 |

The honest framing is "approaching the market in aggregate but not beating it." That averaged number masks tail failures on top matches with inactive / returning players — a known structural limit of vanilla Elo. The dashboard surfaces these gaps explicitly via the "why model differs" panel rather than hiding them.

## The LLM does not emit a probability and does not narrate

The model emits the probability. The LLM emits a typed list of `NewsItem`s (each with title, URL, snippet, source domain, published date, category from a closed whitelist) and a `news_lookup_status` enum. Nothing else.

**No probability**, because a probability emitted by the LLM has no calibration guarantee. Letting it adjust the model's number turns evaluation into a mess (which number do we score?) and creates incentives for the LLM to fabricate adjustment rationale.

**No prose**, because freeform narrative dissolves the dates and sources of facts. Earlier iterations of the project showed structural failures no prompt iteration could fix: articles from prior seasons read as "current form," "defended his title" appeared without backing, year-mixing across multiple sources in one paragraph. The fix was structural — the view layer renders H2H, surface Elo, and recent form deterministically from typed tool outputs; the LLM's only job is news discovery, returning each item as a structured record with its source + date as adjacent metadata.

The LLM's remaining value-add is **discovery + categorisation**: searching for injuries, withdrawals, recent results, coach changes, illness within a 32-day window, tagging each item with a category from a closed whitelist (`injury / withdrawal / illness / result / coach_change / personal`).

## Calibration as post-processing

Isotonic regression on a held-out calibration set of ≥ 1000 samples; Platt (sigmoid) below that. Isotonic on small samples produces overfit, non-smooth curves; Platt is biased but stable.

Both pre- and post-calibration Brier scores are reported. If calibration *worsens* the validation Brier (rare but possible), we investigate before shipping.

## Anti-leakage discipline

Point-in-time correctness is enforced by `tests/test_feature_leakage.py`, not by convention. The test contract: if a future row is tampered with, no feature value as of an earlier date may change.

A failing leakage test is never "fixed" by adjusting the test. The bug is in the feature.

## Why predictions are tour-level only (and feature computation is not)

The dashboard surfaces only **ATP/WTA tour-level singles** — Grand Slams, ATP/WTA 1000/500/250, and the year-end Finals. Challenger and ITF Futures matches are present in the data and **do** feed the feature layer (Elo, recent form, fatigue — players move between tiers, so ratings computed across the full pool are sharper).

Three reasons predictions stop at tour-level:

1. **No market benchmark below tour-level.** tennis-data.co.uk closing prices and matchstat's `tournament/results` odds are tour-only. Without them we can compute Brier on Challenger predictions but cannot say what it means.
2. **The LLM has nothing to surface.** News search against an M15 Futures match returns essentially nothing. The contextualisation layer — the product's user-facing differentiator — has no signal.
3. **Data quality degrades.** Higher retirements/walkovers, more debutants with sparse histories, less data discipline.

The first two are binding. A future Challenger-level odds source would relax point 1, but point 2 still keeps us out.

## What we do not claim

- We do not claim to beat the market. The dashboard renders the market consensus alongside the model and shows the gap.
- We do not claim the model is a "calibrated predictor" in user-facing copy. The product is described as a *research dashboard* / *match dashboard*.
- We do not claim the LLM can predict outcomes the model cannot. The LLM surfaces news; it does not synthesise predictions.
- We do not claim the model is suitable for betting. It is not.
- We do not claim coverage of every tour-level match — features depend on data, and matches with insufficient history are excluded from training.
- We do not claim coverage of Challenger or ITF matches.
