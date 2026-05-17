# Methodology — honest evaluation

This is a product people will actually look at. They'll see "Player A wins with probability 0.62" and form an expectation. We optimize for *trustworthy* probabilistic predictions, not for "look how high our accuracy is." A tool that misleads its users is worse than no tool at all.

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

## The LLM does not emit a probability

The ML model produces the probability. The LLM produces narrative, key factors, caveats, and a qualitative `confidence_band`.

Why this rule exists: a probability emitted by the LLM has no calibration guarantee. Letting it adjust the model's number — even "modestly," "only with justification," etc. — turns the evaluation pipeline into a mess (which number do we score?) and creates incentives for the LLM to fabricate adjustment rationale. The cleaner design is also the more honest one.

The LLM's value-add is **contextualization**: surfacing recent news (injuries, withdrawals), explaining the model's reasoning in plain language, and flagging when the prediction should be trusted less (e.g., a returning player with no recent matches). This is genuinely useful and genuinely safe.

## Why we use market-implied probabilities as a benchmark, not a feature

Bookmaker closing prices are the strongest single signal for tennis match outcomes — they aggregate information no individual model can match. If we used them as a feature, our model would primarily be regressing on the market, and any reported "performance" would be a lower bound on what the market already provides.

Instead, we overlay the market's calibration on our calibration plot. The honest story becomes: *"our model is approaching (but does not match) market calibration, using only public match-level data."* That's a defensible claim. "We beat Vegas" is not.

## Calibration as post-processing

We fit isotonic regression (or Platt scaling on small samples) on a **held-out** calibration set. The choice of method follows a hard rule:

- Held-out calibration set ≥ 1000 → isotonic.
- Smaller → Platt (sigmoid).

Isotonic on small samples produces overfit, non-smooth calibration curves. Platt is biased but stable.

Both pre- and post-calibration Brier scores are reported. If calibration *worsens* the validation Brier score (rare but possible), we investigate before shipping.

## Anti-leakage discipline

Point-in-time correctness is enforced by tests in `tests/test_feature_leakage.py`, not by convention. The test contract: if a future row is tampered with, no feature value as of an earlier date may change. This catches the entire class of bugs where a window is computed off the wrong side of the timestamp.

A failing leakage test is never "fixed" by adjusting the test. The bug is in the feature.

## What we do not claim

- We do not claim to beat the market.
- We do not claim our LLM can predict outcomes the model cannot.
- We do not claim the model is suitable for betting. It is not.
- We do not claim coverage of every tour-level match — feature availability depends on data sources, and matches with insufficient history are deliberately excluded from training.
