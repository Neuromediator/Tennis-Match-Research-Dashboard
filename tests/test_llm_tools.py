"""LLM tool-calling schema and structured-output validation.

TODO(phase-5): assert each tool's JSON schema validates against Anthropic's
tool-definition format, assert the structured output schema rejects an LLM
response that contains a probability field (LLM is not allowed to emit one),
assert confidence_band is one of {"low", "medium", "high"}.
"""
