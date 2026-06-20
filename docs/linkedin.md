# LinkedIn drafts

Three angles for a post about the tennis research dashboard. Pick one, don't merge them.
Keep the live URL and a screenshot in the post itself, not in the text below.

---

## Draft 1 — The build (overview + workflow)

I spent the last few weeks building an end-to-end ML + LLM system, mostly to
learn the parts I hadn't done before. It's a research dashboard for tennis
matches: for any upcoming fixture it shows a trained model's win probability, a
simple Elo baseline, and a news block surfaced by an LLM agent, side by side.

The interesting part wasn't the model, it was everything around it:

- Three messy public data sources reconciled into one DuckDB file, with a
  canonical ID per player and a manual-review step for low-confidence name matches.
- A feature layer (44 features) built by replaying every historical match in order
  and recording the state of the world *before* each match. Point-in-time
  correctness is enforced by a test that tampers a future row and asserts no earlier
  feature changes, not by me being careful.
- Walk-forward validation instead of random k-fold, because the data is a time
  series and k-fold leaks the future. Probabilities are calibrated and scored with
  Brier / log-loss, not accuracy.
- An LLM agent with hard limits (token, time, and tool-call budgets), structured
  output via tool calls, prompt caching, and a log of every call.
- Deployed on a single small machine for a few dollars a month. No microservices,
  no model server, no queue.

The honest takeaway: the model is one signal among several, not an oracle. Building
the evaluation rigorously enough to *see* its limits was more useful than the model
itself.

Live demo and code in the comments.

---

## Draft 2 — One decision (the LLM that doesn't talk)

A small design decision I'm glad I made on my last project: I did not let the LLM
generate a probability, and I did not let it write prose.

The first version had the agent write a paragraph summarising each match. In live
use it failed in ways no amount of prompt tuning fixed: snippets from last season
read as current form, it would smooth over contradictions in the sources, and it
occasionally stated things that simply weren't backed by anything it had read.

So I removed the freeform channel entirely. Now the LLM does one narrow job:
search recent news and return a typed list of items, each with a source, a date, a
URL, and a category from a fixed list. The schema rejects anything else, including
any number the model might try to emit. The deterministic parts (head-to-head,
ratings, recent form) are rendered directly from typed tool outputs, not narrated.

It's more constrained than most LLM features. That's the point. A free-form
generator demos well and fails quietly in production. A typed schema with a
budget and a log of every call is boring and holds up.

Part of a tennis research dashboard I built to practice this end to end. Link below.

---

## Draft 3 — What I learned (short)

Finished a side project: an end-to-end ML + LLM system that predicts probabilities
for tennis matches and shows them next to a few baselines. A few things I'm taking
with me:

1. On a time series, walk-forward validation is the only honest option. Random
   k-fold quietly leaks the future and hands you metrics that fall apart in
   production.

2. Calibration beats accuracy. A model that's 65% accurate but says "75%" when it's
   really 60% will mislead anyone who trusts it. The calibration plot was the
   artefact I actually cared about.

3. Give the LLM a budget and a schema. A single agent with no caps can burn a
   month of API spend in an afternoon. Hard token/time/tool limits plus structured
   output plus a log of every call are unglamorous and save you.

4. A negative result, measured properly, is still a result. The model didn't turn
   out to be the strong forecaster I'd hoped for. Building the evaluation well
   enough to know that, instead of fooling myself, was the real outcome.

Code and a live demo in the comments.

---

## Notes for posting

- Lead the comment with the live URL, then the GitHub link. LinkedIn suppresses
  reach on posts with external links in the body.
- One screenshot of the dashboard does more than any sentence here.
- Don't add the word "production-grade." It's a side project; let the detail speak.
- Hashtags, if any: `#MachineLearning #LLM #MLOps`. Three is plenty.
