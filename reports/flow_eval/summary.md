# Flow evaluation: confirmation question and other languages

6 of 17 fixed conversations run, in 1 of 8 languages, 15 runs in all (each conversation run several times: Gemini varies from run to run). The ML models make every prediction; these checks are about what reaches them.

**Runs fully right: 15/15 (100%)**

## Per check (all runs)

| Check | Passed |
|---|---|
| Asked to confirm the expected quantity (or nothing, for a named one) | 15/15 |
| Question shown before the models ran names no type | 15/15 |
| Scale given to the models matches the answer | 15/15 |
| "No" left that scale out of the averaging (and only then) | 15/15 |
| Country | 15/15 |
| Start month | 15/15 |
| Duration | 15/15 |
| Figure (where the user gave one) | 3/3 |

## Per conversation

| Conversation | Language | Runs fully right | Asked as expected | Scale | Ruled out | Country | Month | Duration | Figure | RF prediction (intended) |
|---|---|---|---|---|---|---|---|---|---|---|
| th-shake-yes | Thai | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | — | Earthquake ×1 (Earthquake) |
| th-shake-no | Thai | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | — | Storm ×3 (Earthquake) |
| th-shake-unsure | Thai | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | — | Storm ×3 (Earthquake) |
| th-shake-yes-figure | Thai | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | Earthquake ×3 (Earthquake) |
| th-wind-yes | Thai | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | — | Storm ×3 (Storm) |
| th-heat-no | Thai | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | — | Storm ×2 (Extreme Temperature) |

## Per language

| Language | Runs fully right |
|---|---|
| Thai | 15/15 |

## Runs with a failed check: 0


Gemini requests per run: 4.3 on average.
Small sample: each rate is out of a few runs per conversation; treat one or two runs as noise.
