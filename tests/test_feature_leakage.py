"""Anti-leakage tests for the feature layer. MANDATORY before any model trains.

The contract: compute_features(..., as_of_date=D) must NEVER read rows whose
match_date >= D. These tests assert that by tampering with future rows and
confirming feature values do not change.

TODO(phase-3): implement once compute_features lands. Tests must run in CI.
"""
