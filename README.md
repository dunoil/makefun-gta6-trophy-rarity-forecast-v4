# makefun-gta6-trophy-rarity-forecast-v4

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dunoil/makefun-gta6-trophy-rarity-forecast-v4/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/dunoil/makefun-gta6-trophy-rarity-forecast-v4/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Trophy rarity forecaster v2: dilution-aware refresh model, calibrated intervals, 55 trophies, backtest-validated

Exophase refreshes GTA VI trophy counts in batches, and each refresh dilutes every earn rate as newly tracked, less-progressed players join. This entry models the refresh process itself instead of trending the rates: per-tier dilution alpha fitted by grid search against the exact discrete printed rates, earners re-anchored from the last 15 page states, future tracked players on a saturating adoption curve fitted on the log-players scale, and delta-method spreads. The practice analogue backtest scores about 0.017 mean pinball loss (house baseline about 1.51); a synthetic 4.5-day final scores 0.048-0.065 with 0.85-1.00 coverage.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
