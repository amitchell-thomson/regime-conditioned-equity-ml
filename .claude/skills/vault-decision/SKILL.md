---
name: vault-decision
description: Log an architectural or design decision to the project Obsidian vault. Use when the user confirms a design choice, settles a debate, or makes a configuration decision that should be recorded with its rationale. Writes to obsidian/decisions/ and updates obsidian/context/now.md.
---

# Vault Decision Logging Skill

When a decision is made that should be permanently recorded, write it to the Obsidian vault so it survives future conversations.

## When to use

- User confirms an approach after discussing alternatives
- A design question from `context/now.md` is resolved
- A configuration choice is settled (e.g. hyperparameter, threshold, data source)
- User says "let's go with X" / "we'll use X" / "that's settled"

## Workflow

1. **Check for an existing file** in `obsidian/decisions/` — update it rather than creating a duplicate.
2. **Write the decision file** at `obsidian/decisions/{kebab-case-title}.md` using the format below.
3. **Update `obsidian/decisions/README.md`** — add the new entry to the Index.
4. **Update `obsidian/context/now.md`** — move the question from "Active open questions" to "Recently settled decisions" with a ✅.

## Decision file format

```markdown
# {Decision Title}

**Decision:** One sentence stating what was decided.

**Why:** The reasoning — what problem this solves, what alternatives were rejected and why.

**How to apply:** When does this rule kick in? What would violate it?

**Enforced by / Config:** Where is this reflected in code or YAML?
```

## Rules

- Lead with the rule/fact, not the background.
- Include the *why* — future Claude instances need to judge edge cases, not just follow the rule blindly.
- Link to related decisions using `[[wikilinks]]`.
- If a commit is relevant, include the short hash.
- Never duplicate content already in CLAUDE.md hard constraints — cross-reference instead.
