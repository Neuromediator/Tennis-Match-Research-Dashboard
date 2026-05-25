# Methodology — honest evaluation

This is a product people will actually look at. They'll see "Player A wins with probability 0.62" and form an expectation. We optimize for *trustworthy* probabilistic predictions, not for "look how high our accuracy is." A tool that misleads its users is worse than no tool at all.

**Phase 6.2 reframe.** Phase 6.1 close-out (2026-05-25) surfaced that the trained LightGBM model is unreliable on top matches even when its average Brier (~0.21) is competitive with the market's (~0.20). Asymmetric matchups (one player active and in form, the other inactive or returning from injury) produced 14-26pp gaps versus market consensus and, in one case, an inverted favourite. The structural cause is vanilla Elo: it cannot decay for inactive players, cannot accelerate for hot streaks, cannot handle injury returns. Rather than overpromise calibration we cannot deliver, the product was re-scoped from "predictor" to "research dashboard" — the model output is rendered as **one signal alongside the market** instead of the headline answer. This methodology document is unchanged at the model-evaluation level (Brier, calibration plots, walk-forward) but updated below to reflect what the product claims about that evaluation.

## Why calibration matters more than accuracy

A predictor that says "Player A wins with probability 0.55" should be right roughly 55% of the time across many such predictions. That's calibration. Accuracy alone is a poor lens — a model that always picks the favorite gets ~65% accuracy in tennis just from rank ordering, while being badly miscalibrated.

We report accuracy. We do not lead with it.

## Headline metrics

| Metric | Why |
|---|---|
| **Brier score** | Mean squared error between predicted probability and outcome. The standard scoring rule for probabilistic forecasts. Lower is better. |
| **Log loss** | Penalizes confident wrong predictions more heavily. Sensitive to calibration. |
| **Calibration plot** | Bins predictions by predicted probability; plots empirical win rate vs predicted. Visualizes where the model is over- or under-confident. |
| Accuracy | Reported, not headlined. |
| Accuracy by ranking gap | Sanity check that the model isn't only good when one player is much higher-ranked. |

## Walk-forward validation

Tennis data is a time series. Random k-fold cross-validation leaks future information into past predictions and produces overconfident metrics.

Walk-forward: train on years up to `Y`, validate on year `Y+1`, advance `Y`, repeat. Every metric reported is computed on data the model has never seen, in the order the model would encounter it in production. There is no scenario where this protocol over-reports performance.

## The LLM does not emit a probability — and no longer emits prose

The ML model produces the probability. The LLM produces a typed list of `NewsItem`s (each with title + url + snippet + source domain + published date + category from a whitelist) plus a `news_lookup_status` enum. Nothing else. No narrative paragraph, no key-factors list, no caveats list, no qualitative confidence band.

Why no probability: a probability emitted by the LLM has no calibration guarantee. Letting it adjust the model's number — even "modestly," "only with justification," etc. — turns the evaluation pipeline into a mess (which number do we score?) and creates incentives for the LLM to fabricate adjustment rationale.

Why no prose (Phase 6.1 lesson): freeform `narrative` dissolves the dates and sources of facts. Live use showed structural failures no prompt iteration could fix — articles from prior seasons read as "current form," "defended his title" appeared without backing, year-mixing across multiple sources within one paragraph. The fix was structural: take prose synthesis out of the LLM's hands. The view layer renders H2H, surface Elo, and recent form deterministically from typed tool outputs; the LLM's only job is news discovery, returning each item as a structured record with its source + date as adjacent metadata (not buried inside a sentence).

The LLM's remaining value-add is **discovery + categorisation**: searching the open web for injuries, withdrawals, recent results, coach changes, illness within a 32-day window, and tagging each item with a category from a closed whitelist (`injury / withdrawal / illness / result / coach_change / personal`). The agent does not synthesise — it lists, dates, attributes, and tags.

## Why we use market-implied probabilities as a benchmark, not a feature

Bookmaker closing prices are the strongest single signal for tennis match outcomes — they aggregate information no individual model can match. If we used them as a feature, our model would primarily be regressing on the market, and any reported "performance" would be a lower bound on what the market already provides.

The benchmark applies in two places:

1. **Historical evaluation (Phase 4):** tennis-data.co.uk closing prices loaded into `market_implied_probabilities`, overlaid on our calibration plots in `metadata.json` / `report.md` for every model artifact. The walk-forward Brier reads: ATP 0.2105 (our LightGBM) vs 0.2220 (Elo baseline) vs ~0.20 (market) on the most recent 5 folds. *"Our model is approaching but does not match market calibration, using only public match-level data"* — that's the defensible claim. *"We beat Vegas"* is not.

2. **Live dashboard (Phase 6.2):** The Odds API pre-match odds loaded into `pre_match_odds` for every currently-active tour-level match. The Prediction page renders a **comparison row** — market consensus, our model, surface-Elo — so the user sees the gap at the per-match level, not just in aggregate. When |model − market| > 10pp, a deterministic "why model differs" panel surfaces the structural cause (stale surface-Elo, activity asymmetry, returning veteran). Live odds are also not a training feature; same rule as historical.

The averaged-Brier "approaching the market" framing was technically true but operationally misleading — it masked the tail-failure modes Phase 6.2 was created to address. The 10-match reality acceptance test (CLAUDE.md hard rule #12) is the corrective: no phase touching the prediction surface closes without a per-match walkthrough that includes the market price for each fixture.

## Calibration as post-processing

We fit isotonic regression (or Platt scaling on small samples) on a **held-out** calibration set. The choice of method follows a hard rule:

- Held-out calibration set ≥ 1000 → isotonic.
- Smaller → Platt (sigmoid).

Isotonic on small samples produces overfit, non-smooth calibration curves. Platt is biased but stable.

Both pre- and post-calibration Brier scores are reported. If calibration *worsens* the validation Brier score (rare but possible), we investigate before shipping.

## Anti-leakage discipline

Point-in-time correctness is enforced by tests in `tests/test_feature_leakage.py`, not by convention. The test contract: if a future row is tampered with, no feature value as of an earlier date may change. This catches the entire class of bugs where a window is computed off the wrong side of the timestamp.

A failing leakage test is never "fixed" by adjusting the test. The bug is in the feature.

## Why predictions are tour-level only (and feature computation is not)

The product predicts only **ATP/WTA tour-level singles** matches — Grand Slams, ATP/WTA 1000/500/250, and the year-end Finals. Challenger and ITF Futures matches are present in our data sources and **do** feed the feature layer (Elo ratings, recent form, fatigue) — players move between tiers, and ratings computed across the full pool are sharper than ratings computed on tour-only history. We just don't surface predictions for those tiers in the UI.

Three reasons the prediction surface stops at tour-level:

1. **No market benchmark exists below tour-level.** tennis-data.co.uk closing prices and matchstat's `tournament/results` odds are tour-level only. Without them we can compute our Brier score on Challenger / ITF predictions, but we cannot say what it *means*. The headline narrative "our calibration is approaching the market's" — the only honest framing we have — collapses. We'd be left with "trust this number," which is exactly what this document rejects.

2. **The LLM analyst has nothing to surface.** `search_tennis_news` against an M15 Futures match returns essentially zero results — there is no media coverage of injuries, withdrawals, or personal events at that level. The contextualization layer — the product's user-facing differentiator — has no signal to work with. A prediction with no `key_factors` and no `caveats` is just a number, which we explicitly do not ship.

3. **Data quality degrades at lower tiers.** Higher rates of retirements and walkovers, more debutants with sparse histories, and reduced data discipline (the matchstat probe surfaced "Unknown Player" placeholders for older lower-tier records). Recent-form windows become noisier; feature variance rises faster than feature signal.

The first two points are the binding constraints; the third is a contributing one. If a future Challenger-level odds source appeared, point 1 would relax for that tier — but point 2 would still keep us from a useful product surface there.

## What we do not claim

- We do not claim to beat the market. Phase 6.2 makes this explicit at the UI level — the market consensus is rendered alongside the model and the gap is shown, not hidden.
- We do not claim the model is a "calibrated predictor" in UI copy (Phase 6.2 hard rule #11). The product is described as a "research dashboard" / "match dashboard." Model output is one signal, not the answer.
- We do not claim our LLM can predict outcomes the model cannot. The LLM surfaces news; it does not synthesise predictions.
- We do not claim the model is suitable for betting. It is not. The Sinner-Djokovic / Cina-Opelka / Kasatkina cases from the Phase 6.1 close-out smoke (model 14-26pp away from market, one inverted favourite) are documented in `phase_6_2_notes.md` Part 1 precisely so future readers understand the limits.
- We do not claim coverage of every tour-level match — feature availability depends on data sources, and matches with insufficient history are deliberately excluded from training.
- We do not claim coverage of Challenger or ITF matches. See [Why predictions are tour-level only](#why-predictions-are-tour-level-only-and-feature-computation-is-not).
