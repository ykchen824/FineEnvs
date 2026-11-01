# Reward-asset audit

This is the first iteration of the reward-hacking audit requested in
[issue #1](https://github.com/ykchen824/FineEnvs/issues/1). It is a gate on
*changes* to the assets that decide an environment's reward, not a runtime
trace; execution logging is the next iteration.

## What it protects

`manifest.json` lists the tests, graders, rubrics, reward modules, evaluators,
reference data and audit tooling in this repository, grouped by category:

| Category | Examples |
|---|---|
| `tests` | every `tests/` directory and `test_*.py` |
| `reward_logic` | rubrics, rewards, scoring, gates, judges, graders and the servers that apply them |
| `evaluation` | `eval/` trees, `eval_*.py`, `evaluate.py`, `verify*.py` |
| `reference_data` | task packs, evaluation sets, corpus manifests, fixtures that define the reward |
| `audit_tooling` | this directory and the workflows that run the checks |

The patterns are deliberately broad: a change to any test that could be
weakened needs the approval label, not only the tests that already mention
"reward". A new reward-critical file is covered if it matches an existing
pattern; add a pattern when it does not.

## How a pull request is judged

The workflow runs `check.py --base origin/<base>` on every pull request:

1. It reads the manifest from the **base revision**, not from the pull
   request. A pull request cannot hide a protected change by narrowing the
   manifest in the same commit; the base manifest still names the file.
2. It lists the files changed between the merge base and the pull request.
3. Every changed path that matches a base-manifest pattern is a protected
   change. If the pull request carries the `audit/approved` label, the gate
   passes; otherwise it fails and prints the paths.
4. A pull request that introduces the manifest for the first time has no base
   manifest. That bootstrap is accepted after the new manifest validates; the
   pull request itself is reviewed like any other.

The manifest's own change is protected too, and so is every file in
`.github/workflows/`; otherwise a pull request could remove the gate or skip
the tests it runs.

## Setup (maintainer, once)

1. Create the label `audit/approved` in the repository settings.
2. Make the **Reward audit** workflow a required status check on `main`.
3. Only people with triage or write permission can apply labels; keep it that
   way. The label is the review record for a reward-critical change.

## Run it locally

From the repository root:

```bash
# Validate the manifest itself; every pattern must match at least one tracked file.
python3 tools/reward_audit/check.py --validate

# Compare the current branch with main.
python3 tools/reward_audit/check.py --base origin/main

# A dry run of the approved path.
python3 tools/reward_audit/check.py --base origin/main --labels audit/approved

# Unit tests (standard library only).
python3 -m unittest discover -s tools/reward_audit/tests -v
```

`--base` can be any git revision. The script needs only Python 3.10+ and git.

## Resolution when the gate fails

- If the protected change is legitimate, a maintainer reviews it and applies
  `audit/approved`; the gate passes on the next run.
- If the change should not need review (for example, a new file that is not
  reward-critical but matches a broad pattern), narrow the pattern in a
  separate, reviewed change rather than approving the file.
- If a protected asset moved or was deleted deliberately, update the manifest
  in the same reviewed pull request. The base manifest still requires the label
  for the old path; the new manifest must continue to match the asset's new
  location.

## Limits

- The gate records that a maintainer reviewed a change. It cannot prove what
  was reviewed; branch protection and the reviewer list are the rest of that
  control.
- The label is not a cryptographic signature, and a maintainer can apply it by
  mistake. The CI output lists every protected path so the record is explicit.
- This iteration does not log runtime access to these assets. A sandbox can
  still read a public reference answer at run time; that is the execution-trace
  and independent-verification iteration.
- A new reward-critical file that matches no pattern is not protected. The
  `--validate` check catches dead patterns, not missing ones; update the
  manifest when adding a new reward path.
