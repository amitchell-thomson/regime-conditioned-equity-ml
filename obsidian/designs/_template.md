# [Proposal Title]

**Status:** 📝 Draft
**Date:** YYYY-MM-DD
**Phase:** Phase X

---

## Approach

What is being proposed and how does it work?

## Integration points

Which modules does this touch? What are the inputs and outputs?

- Module: `src/regime_ml/...`
- Inputs:
- Outputs:

## Leakage / causality risks

- Does any step use future data?
- Does this touch the IS/OOS boundary? If so, how is it respected?
- Does this use `smooth_proba()` anywhere? (must not — trading logic only)
- Are all parameters in YAML config?

## Open questions

- [ ] ...

## Decision

**Status:** (fill when reviewed)
**Approved / Rejected by:**
**Reason:**
