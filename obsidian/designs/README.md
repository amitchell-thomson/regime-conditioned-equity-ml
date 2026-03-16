# Design Proposals

Proposals for changes touching regime inference, evaluation, or major architecture. Per CLAUDE.md: write proposal first, implement only after approval.

## Template
Each proposal should cover:
1. Approach
2. Integration points
3. Leakage / causality risks

## Status legend
- 📝 Draft
- ✅ Approved — ready to implement
- 🚫 Rejected
- ✅ Implemented

## Index

| Proposal | Status | Summary |
|---|---|---|
| [[five-state-surgical-split]] | 📝 Draft | Initialise 5-state HMM by splitting the most ambiguous 4-state model state. Resolves CV churn root cause. |
