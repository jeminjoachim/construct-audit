# Do text classifiers measure what they claim? Testing validity with AI-generated counterfactuals

Prototype and data for the Stage 1 proposal, "Do text classifiers measure what they claim? Testing validity with AI-generated counterfactuals," submitted to the *Organization Science* special issue "AI-Enabled Frontiers in Organizational Science".

A counterfactual construct audit edits real texts in ways a codebook says should, or should not, change their label, and checks whether a classifier's score moves as the codebook predicts. This repository contains the two audits reported in the proposal:

- **Study 1:** a classifier of due-diligence questions in forum posts (Bitcointalk).
- **Study 2:** a published classifier of environmental claims in company reports
  (Stammbach et al., 2023).

Design, results and limitations are described in the technical appendix ([link]).

## Repository layout

```
features.yaml                 Study 1 rules, edit instructions and predictions
features_envclaims.yaml       Study 2 rules, edit instructions and predictions
prompts/                      System prompts used to generate edits
scripts/                      Generation, review, scoring and export scripts
data/forum_public/            Study 1: edits (changed words only), post links, scores
data/envclaims/               Study 2: sentences, edits, review decisions, scores
results/                      Score tables and figures reported in the appendix
config.example.yaml           Template for local paths (copy to config.yaml)
```

## Reproducing Study 2 (environmental claims)

Everything needed is public. Python 3.12.

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/envclaims_prepare.py
python scripts/score_envclaims.py --reproduce --variants
```

`--reproduce` scores the published test set and compares it with the paper.
`--variants` scores the original sentences, placebo edits, neutral clauses and accepted
edits, and writes `results/envclaims/envclaims_score_table.csv` (Appendix Table A4.3).

## Study 1 (forum posts)

The forum posts are public but are not copied here, because a post's text leads back
to its author. `data/forum_public/edits.csv` gives a link to each post and the words
each edit changed; `data/forum_public/scores.csv` gives every score (Appendix Table A4.2).

The classifier belongs to an unpublished doctoral study and is not released. It is
identified by its SHA-256 hash (Appendix A2), so the scores can be checked against it
once it is published.

## Generating edits

Edits were generated with an OpenAI model (gpt-6-luna).
Regenerating them requires an OpenAI API key and will not give identical edits, which is why the accepted edits and all review decisions are published.

```
$env:OPENAI_API_KEY = "<your key>"     # PowerShell
python scripts/generate_variants.py --dry-run     # shows the prompts without calling the API
```

The review step is interactive: `python scripts/review_variants.py` shows each edit next
to its original and records your judgement (`--redo <post_id>:<edit>` reviews one edit
again; type `q` at any prompt to save and quit).

## Which file supports which table

| Appendix table | File |
|---|---|
| A3.1 Review outcomes | `data/forum_public/edits.csv`, `data/envclaims/variant_checks_envclaims.csv` |
| A4.2 Study 1 scores | `data/forum_public/scores.csv` |
| A4.3 Study 2 scores | `results/envclaims/envclaims_score_table.csv` |
| A2 Reproduced performance | `results/envclaims/reproduction.csv` |

## Use of AI tools

Code was developed with the assistance of Claude (Anthropic). Edits were generated
with an OpenAI model as described in Appendix A3. The author reviewed and tested all
code and is responsible for all results.

## Licences

- Code: MIT (see `LICENSE`).
- Study 2 data: CC BY-NC-SA 4.0, following the source dataset (see `data/envclaims/LICENSE_DATA.md`).
- Study 1 post IDs, edit descriptions and scores: CC BY 4.0. Post texts are not included.

## Citation

Jemin Joachim (2026). "Do text classifiers measure what they claim? Testing validity with AI-generated counterfactuals
Stage 1 proposal, Organization Science special issue. Version v1.0-stage1, https://github.com/jeminjoachim/construct-audit.
