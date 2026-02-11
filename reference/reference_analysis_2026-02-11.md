# AIMO3 Reference Analysis - 2026-02-11

Source: `reference/ai-mathematical-olympiad-progress-prize-3/reference.csv`

## Observed structure

- Problem mix (10 items):
  - `algebra`: 4
  - `geometry`: 2
  - `number_theory`: 2
  - `combinatorics`: 1
  - `general`: 1
- Archetype concentration:
  - `functional_equation`: 3
  - `geometry_configuration`: 2
  - `divisor_arithmetic`: 2
  - `combinatorial_count`: 1
  - `general_olympiad`: 2

## Modulus patterns encountered

- Explicit formats:
  - `10^{5}`
  - `99991`
  - `5^7`
- Some items do not explicitly declare a modulus and require default handling.

## Implications for prompts/solver

- Single-template prompting is brittle; category/archetype routing is necessary.
- Geometry and functional-equation prompts require explicit contradiction checks and independent-route verification.
- Modulus parsing must support LaTeX and expression forms; normalization quality directly impacts score.
- Final-answer extraction remains sensitive to truncated long completions, so compact extractor passes are required.

## Current status

- Parsing pipeline now handles LaTeX modulus forms used in this reference set.
- Strategy stack now includes:
  - verification
  - consistency audit
  - adversarial probe
  - geometry recheck
  - small-answer guard
  - selector

Next reference pass should run with full `aimo120b` budget to evaluate these upgrades under realistic settings.
