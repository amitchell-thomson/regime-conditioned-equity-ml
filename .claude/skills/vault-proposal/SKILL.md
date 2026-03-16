---
name: vault-proposal
description: Write a design proposal to the project Obsidian vault. Use when the user asks to design, plan, or propose changes to regime inference, evaluation, model selection, or any major architectural component. Required by CLAUDE.md before implementing anything touching regime inference or evaluation. Writes to obsidian/designs/.
---

# Vault Design Proposal Skill

CLAUDE.md requires a written design proposal before implementing anything touching regime inference, evaluation, or major architecture. This skill writes that proposal to the vault.

## When to use

- User asks to design or plan a new feature touching regime inference or evaluation
- User asks "how should we..." / "what's the approach for..." a Phase 3 or Phase 4 component
- Any change to `regimes/hmm.py`, `regimes/evaluation.py`, `regimes/selection.py`, `regimes/labeling.py`
- Any new model architecture or portfolio evaluation approach

## Workflow

1. **Discuss** approach, integration points, and leakage risks with the user first.
2. **Write the proposal** to `obsidian/designs/{kebab-case-title}.md` using the template below.
3. **Add to the index** in `obsidian/designs/README.md`.
4. **Add to phase notes** — link from the relevant `obsidian/phases/phase-X.md` under "Design proposals".
5. **Stop. Do not implement** until the user explicitly approves the proposal.

## Proposal file format

```markdown
# {Proposal Title}

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
- Does this use `smooth_proba()` anywhere? (must not appear in trading logic)
- Are all parameters destined for YAML config?

## Open questions

- [ ] ...

## Decision

**Status:** (fill when reviewed)
**Approved / Rejected:**
**Reason:**
```

## Status legend

- 📝 Draft — written, not yet reviewed
- ✅ Approved — ready to implement
- 🚫 Rejected — with reason recorded
- ✅ Implemented — code exists

## Rules

- Do not begin implementation until the user says "approved" or equivalent.
- If the user approves verbally, update the proposal status to ✅ Approved before writing code.
- If a proposal is rejected, record the reason — future Claude instances should not re-propose the same approach.
- Link to related decisions using `[[wikilinks]]`.
