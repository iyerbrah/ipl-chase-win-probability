# IPL Chase Win Probability

Win probability for every ball of an IPL run chase, from three models compared on seasons they never saw, and a leverage score that shows which moments of a match carry the most tension.

I built this to answer two questions:

1. At any ball of a chase, how likely is the chasing team to win?
2. Which over-breaks come right before the tensest moments? Those are the ad slots worth the most to a broadcaster.

## The app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pick any chase since 2008. The **Win probability** tab shows the three models ball by ball, the biggest turning points, and how the models score on the test seasons. The **Leverage** tab shows how much each ball could swing the match and ranks the over-breaks.

## Data

Ball-by-ball data for every IPL match from 2008 to 2026, from [Cricsheet](https://cricsheet.org/).

- 1,243 matches read, 1,185 kept
- 137,887 second-innings deliveries
- Dropped: 16 ties, 9 no-results, 23 rain-adjusted (D/L) matches, 10 other shortened matches

One row is the state of the match just before a delivery, labelled with whether the chasing team went on to win.

## Features

| Feature | Meaning |
|---|---|
| `runs_needed` | Target minus current score |
| `balls_remaining` | Legal balls left, out of 120 |
| `wickets_in_hand` | 10 minus wickets fallen |
| `required_run_rate` | Runs needed per over from here |
| `current_run_rate` | Runs per over so far |
| `target` | First-innings score plus one |
| `venue_avg_score` | Average first-innings score at the ground |
| `league_avg_score` | Average first-innings score over the previous 100 matches |

I left out team and player names on purpose. Squads change every year, and I wanted the models to judge the situation, not the franchise.

The two scoring-level features only use matches played before the one being predicted. The venue average is pulled toward the league average when a ground has hosted few games.

## Models

| Model | Notes |
|---|---|
| Logistic regression | Standardised features |
| LightGBM | 206 trees, 7 leaves. Constrained so that needing more runs can never raise the win probability, and having more balls or wickets can never lower it |
| Neural network | Two hidden layers (64, 32), 3 epochs |

The split is by season: train on 2008-2023, tune on 2024, test on 2025-2026. A random split would put ball 50 of a match in training and ball 51 in testing, which leaks the result, so whole seasons stay together. After tuning, each model is refit on train plus validation.

## Results

Test seasons 2025-2026: 138 matches, 15,914 deliveries.

| Model | Accuracy | Log loss | Brier |
|---|---|---|---|
| No model (always back the chasing team) | 55.9% | 0.687 | 0.247 |
| Logistic regression | 81.9% | 0.382 | 0.123 |
| LightGBM | 80.9% | 0.429 | 0.138 |
| Neural network | 83.1% | 0.382 | 0.121 |

Accuracy by phase of the chase:

| Model | Overs 1-6 | Overs 7-15 | Overs 16-20 |
|---|---|---|---|
| Logistic regression | 74.1% | 83.8% | 90.1% |
| LightGBM | 75.5% | 80.7% | 90.2% |
| Neural network | 75.2% | 84.9% | 92.2% |

![Calibration on the test seasons](reports/calibration.png)

I expected LightGBM to win and it didn't. Logistic regression and the neural network are tied, and the gap between them is much smaller than the swing from one season to the next.

My best explanation is that the game changed. Average first-innings scores went from about 165 before 2023 to about 190 in 2025-2026, so the test seasons are full of targets the models rarely saw in training. A linear model carries its trend into that range. Trees can't predict beyond what they were trained on. I haven't proved this is the cause.

I use logistic regression for the leverage step. It had the best validation log loss and it is the easiest of the three to explain.

## Leverage

Leverage isn't a model. For a given match state I ask the win-probability model what the probability would be after each possible next ball: dot, 1, 2, 3, 4, 6, wicket, wide or no-ball. Leverage is the average absolute change, weighted by how often each outcome happens in that phase of the innings, divided by the same number for an average ball. So 1.0 is an average ball.

An over-break is scored by the leverage of the first ball after it.

Across 2,572 over-breaks in the test seasons:

- the top 10% hold 29% of all leverage, and the top 20% hold 44%
- average leverage is about 0.7 before overs 7-15 and 2.2 before the final over

The match with the highest leverage in the test set is Chennai Super Kings chasing 214 against Royal Challengers Bengaluru on 3 May 2025. The flattest ones are a chase of 76 and two chases of 279 or more, all decided early.

## Limitations

- Leverage is a proxy for viewer attention, and I have no viewership data to check it against. With minute-level concurrency numbers I would compare viewer drop-off in high- and low-leverage ad breaks, controlling for teams, time slot and match stage.
- Leverage ignores other things that drive viewership: which teams are playing, star players, weekday or weekend.
- The next-ball outcome frequencies are fixed per phase of the innings, so leverage steps slightly at overs 7 and 16.
- 138 test matches is a small sample. Small gaps between models are noise.
- I looked at test scores once before the final setup. After that I lowered the neural network's learning rate and added the refit on train plus validation, both chosen on the 2024 season. The features did not change.
- Second innings only. Rain-affected matches and player-level effects are out of scope.

## Rebuilding from scratch

The processed data and trained models are in the repo, so the app runs as is. To rebuild everything, download `ipl_json.zip` from Cricsheet, unzip it into `data/raw/ipl_json/`, and run:

```bash
python src/build_dataset.py
python src/train.py
python src/evaluate.py
python src/leverage.py
```

## Layout

```
app.py                 Streamlit app
src/features.py        Feature definitions shared by every script
src/build_dataset.py   Cricsheet JSON to one row per delivery
src/train.py           Trains the three models
src/evaluate.py        Scores them on the test seasons
src/leverage.py        Win probability for any state, leverage, over-break value
models/                Trained models and their settings
reports/               Metrics and the calibration chart
data/processed/        The dataset and the predictions the app reads
```
