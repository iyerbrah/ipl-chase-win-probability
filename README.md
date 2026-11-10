# IPL Win Probability: who wins the chase, ball by ball

Predicts the chasing team's chance of winning before every ball of an IPL run chase, compares three machine learning models on seasons they never saw, and uses the best one to measure which moments of a match carry the most tension: the over-breaks a broadcaster would most want to sell.

[![CI](https://github.com/iyerbrah/ipl-chase-win-probability/actions/workflows/ci.yml/badge.svg)](https://github.com/iyerbrah/ipl-chase-win-probability/actions/workflows/ci.yml)
[![Live app](https://img.shields.io/badge/live-Streamlit%20app-3987e5)](https://ipl-win-probability-rxgwz3kbdl6nwuk3rmyjwe.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/license-MIT-199e70)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab)](requirements.txt)

**[Try the live app →](https://ipl-win-probability-rxgwz3kbdl6nwuk3rmyjwe.streamlit.app/)** · Built by [Ram Iyer](https://github.com/iyerbrah) · *Hosted on Streamlit Community Cloud: if it has gone to sleep, click to wake it and give it a minute.*

![The app replaying a 2026 chase ball by ball, with all three models](docs/win_probability.png)

## Highlights

- **82% accuracy on unseen seasons.** The models pick the eventual winner on 82% of deliveries in 2025-2026, against 56% for always backing the chasing team.
- **The simplest model was as good as the most complex.** Logistic regression matched the neural network and beat LightGBM; the likely reason is a rise in scoring that trees can't extrapolate to (see [Results](#results)).
- **A new measure of tension.** Leverage scores how much the next ball can swing the match. The top 10% of over-breaks hold 29% of all leverage, mostly late in close chases.
- **No leaks by design.** Train, tune and test are split by season, team and player names are left out, and LightGBM is constrained so needing more runs can never help.
- **Tested like software.** 25 offline tests in CI check the features, that every model responds to the match the right way, that no match crosses splits, that the README's numbers match the reports, and that both pages render.

**Stack:** Python · pandas · scikit-learn · LightGBM · Plotly · Streamlit · pytest · GitHub Actions

| Pick any chase | Leverage, ball by ball | The over-breaks that matter |
| --- | --- | --- |
| ![The app: match picker, headline numbers and the win probability chart](docs/app.jpg) | ![The Leverage page: flat for 15 overs, then a spike in the last over](docs/leverage_page.jpg) | ![The most valuable over-breaks in a match, with win probability and leverage](docs/over_breaks.jpg) |

## Why I built it

A broadcaster earns from attention, and attention during a cricket match is uneven. I wanted to answer two questions:

1. At any ball of a chase, how likely is the chasing team to win?
2. Which over-breaks come right before the tensest moments? Those are the most valuable ad slots.

## How it works

```mermaid
flowchart LR
  C[Cricsheet JSON<br/>every IPL match] --> B[build_dataset.py<br/>one row per ball of the chase]
  B --> T[train.py<br/>3 models, split by season]
  T --> E[evaluate.py<br/>scored on 2025-2026]
  T --> L[leverage.py<br/>win probability after<br/>every possible next ball]
  E --> A[Streamlit app]
  L --> A
```

| Step | What happens |
|---|---|
| 1. Data | Every IPL match from 2008 to 2026 (from [Cricsheet](https://cricsheet.org/)) becomes one row per ball of the chase: 1,185 matches, 137,887 deliveries |
| 2. Models | Logistic regression, LightGBM and a neural network each predict win probability from the match situation |
| 3. Testing | All three are scored on the 2025 and 2026 seasons, which were held back from training |
| 4. Leverage | The model with the lowest log loss on the 2024 validation season (logistic regression) is used to score how much each ball could swing the match |
| 5. App | A Streamlit app replays any chase with the predictions and leverage |

## What the models see

Eight numbers, all known before the ball is bowled:

- runs needed, balls remaining, wickets in hand
- required run rate and current run rate
- the target
- the average first-innings score at the ground, and across the league recently

Team and player names are left out on purpose. Squads change every year, and the models should judge the situation, not the franchise.

## Results

Tested on 138 matches (15,914 deliveries) from 2025 and 2026.

| Model | Accuracy | Log loss | Brier score |
|---|---|---|---|
| No model (always back the chasing team) | 55.9% | 0.687 | 0.247 |
| Logistic regression | 81.9% | 0.382 | 0.123 |
| LightGBM | 80.9% | 0.429 | 0.138 |
| Neural network | 83.1% | 0.382 | 0.121 |

Accuracy is how often the model's favourite went on to win. Log loss and Brier score measure how far the predicted probabilities were from the results; lower is better.

**What I found**

- All three models are far better than guessing: about 82% against 56%.
- The simplest model, logistic regression, matches the neural network and beats LightGBM.
- Accuracy climbs through the chase, from about 75% in overs 1-6 to about 90% in overs 16-20, because less of the result is still open.

I expected LightGBM to win. My best explanation for why it didn't is that the game changed: average first-innings scores rose from about 165 before 2023 to about 190 in 2025-2026, so the test seasons contain targets the models rarely saw in training. A linear model extends its trend into that range, while tree models cannot predict beyond what they were trained on. I have not proved this is the cause.

## Leverage

Leverage measures how much the next ball can change the win probability.

For a given situation, I ask the model what the win probability would be after each possible next ball (dot, single, four, six, wicket and so on). Leverage is the average size of those changes, weighted by how often each outcome happens, divided by the same figure for an average ball. A leverage of 1.0 is an average ball; 3.0 means the ball matters three times as much.

An over-break is scored by the leverage of the first ball after it.

**What I found** across 2,572 over-breaks in the test seasons:

- The top 10% of over-breaks hold 29% of all leverage; the top 20% hold 44%.
- Average leverage is about 0.7 before overs 7-15 and 2.2 before the final over.

So match tension is concentrated in a small number of breaks, mostly late in close chases.

![The Leverage page for the same chase: flat for 15 overs, then a spike in the last over](docs/leverage.png)

## Key decisions

- **Split by season, never at random.** A random split would put ball 50 of a match in training and ball 51 in testing, which leaks the result. Train on 2008-2023, tune on 2024, test on 2025-2026.
- **The situation, not the franchise.** No team or player names, so the models can't learn last year's squads.
- **Constraints that encode cricket.** LightGBM can't make needing more runs raise the win probability, or more balls and wickets lower it. The tests check all three models on runs and wickets.
- **Leverage is computed, not learned.** It reuses the win-probability model on every possible next ball, so it needs no labels and can be explained in one sentence.
- **Choose on validation, report on test.** The leverage model is the one with the best 2024 log loss; the test seasons were only used to report results.

## Limitations

- **Leverage is a proxy for viewer attention.** I have no viewership data to check it against. With minute-by-minute viewer numbers, I would compare viewer drop-off in high- and low-leverage ad breaks, controlling for teams, time slot and match stage.
- **The models underrate the chasing team in the test seasons.** When logistic regression gave the chasing team about 45%, it won 60% of the time; when LightGBM gave about 55%, it won 77%. The likely cause is the same rise in scoring. Leverage is built on these probabilities, so it inherits the error. In the chart below, a perfect model would sit on the dashed line; all three sit above it.

  <img src="reports/calibration.png" alt="Calibration chart: predicted win probability against actual win rate for the three models" width="420">

- 138 test matches is a small sample, so small gaps between models are noise.
- Only the second innings is modelled. Rain-affected matches are excluded.
- I looked at test scores once before the final setup. After that I lowered the neural network's learning rate and added a refit on train plus validation data, both chosen using the 2024 season. The features did not change.

## How the models were trained

- **Split by season:** train on 2008-2023, tune on 2024, test on 2025-2026.
- **Logistic regression:** standardised features.
- **LightGBM:** 206 trees with 7 leaves each, constrained so that needing more runs can never raise the win probability.
- **Neural network:** two hidden layers of 64 and 32 units.

After tuning, each model is refit on the training and validation seasons together.

## Repository tour

| Path | What's there |
| --- | --- |
| [`src/features.py`](src/features.py) | The eight features, shared by every script and the app |
| [`src/build_dataset.py`](src/build_dataset.py) | Cricsheet JSON to one row per delivery of the chase |
| [`src/train.py`](src/train.py), [`src/evaluate.py`](src/evaluate.py) | Trains the three models by season; scores them on the test seasons |
| [`src/leverage.py`](src/leverage.py) | Win probability for any situation, and leverage on top of it |
| [`app.py`](app.py), [`views/`](views/), [`charts.py`](charts.py) | The Streamlit app: match picker, the two pages and their charts |
| [`models/`](models/), [`reports/`](reports/) | The trained models and their settings; metrics and the calibration chart |
| [`data/processed/`](data/processed/) | The dataset and the predictions the app reads |
| [`tests/`](tests/) | 25 offline tests: features, model behaviour, no leaks, README numbers, both pages |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Lint and tests on every pull request |

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/streamlit run app.py
.venv/bin/pytest -q          # 25 tests, all offline
```

The processed data and trained models are in the repo, so the app and the tests run as they are. (On macOS, LightGBM needs OpenMP: `brew install libomp`.) To rebuild everything, download `ipl_json.zip` from Cricsheet, unzip it into `data/raw/ipl_json/`, and run:

```bash
python src/build_dataset.py
python src/train.py
python src/evaluate.py
python src/leverage.py
```

## License

The code is under the [MIT license](LICENSE). The ball-by-ball data comes from [Cricsheet](https://cricsheet.org/); credit Cricsheet if you reuse it, under the terms on its site.
