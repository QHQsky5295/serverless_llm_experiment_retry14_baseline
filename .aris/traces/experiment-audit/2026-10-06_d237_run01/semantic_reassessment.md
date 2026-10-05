# Semantic reassessment trace (2026-10-06)

Reviewer follow-up: `/root/d237_integrity_review` (read-only).

Conclusion: the IEEE PDF defines `E_i(t)` as a proactive residency budget and
explicitly gives request-driven loading priority over proactive preparation.
P1-D4/D29 and `residency_manager.py` preserve ordinary demand loading as native
LRU/backend allocation, not soft E(t). D227's remote/NVMe demand-only prefix
therefore shows that no proactive candidate was exercised; it does not prove a
request-path semantic defect. The next diagnostic must submit a legitimate
HOST-to-GPU proactive candidate and keep hard request-capacity evidence separate.

This follow-up corrects the causal interpretation only; D227 hashes, raw
observations, cleanup evidence and non-ranking boundary remain unchanged.
