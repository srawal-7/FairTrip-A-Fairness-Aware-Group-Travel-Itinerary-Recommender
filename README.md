# FairTrip: A Fairness-Aware Group Travel Itinerary Recommender

A group itinerary recommender that treats disagreement between group members as a signal to optimize around, rather than averaging it away into a single blended preference.

## Problem

Existing group travel tools (TripAdvisor, Google Travel) are built for individual users and have no mechanism for modeling conflicting preferences within a group. Applied naively to groups, they effectively optimize for one person or default to popularity ranking, leaving other members underserved.

*Originally developed as a team project for a graduate course at Drexel University.*

## Approach

**Data:** Filtered the Yelp Open Dataset (6.99M reviews, 150K businesses) down to 578,777 Philadelphia reviews, keeping only users with 5+ reviews for reliable collaborative filtering signal. Engineered check-in counts, tip counts, and text-based sentiment flags (`has_positive`, `has_negative`) independent of star ratings, to detect venues where reviewer sentiment and star ratings disagree.

**Collaborative filtering:** Originally proposed LightFM, substituted with Implicit ALS due to environment compatibility issues. Trained on the sparse user-business ratings matrix (users average ~15 reviews across 14,344 venues).

**Conflict detection:** Scored each candidate venue by how much the group disagreed on it, classifying each member's preferences as must-have or nice-to-have before optimization.

**Fairness-aware optimization:** Implemented a greedy Nash Social Welfare optimizer that, at each selection step, picks the venue maximizing the sum of log-utilities across all group members, prioritizing whoever is currently least satisfied.

**Evaluation:** Simulated groups of 3-5 real users from their historical ratings, measuring Nash Social Welfare (group fairness), AUC (individual recommendation quality against held-out ratings), and Mean Reciprocal Rank (how early a user's top preference appears in the 5-stop itinerary).

## Results

| Model | NSW | AUC | MRR |
|---|---|---|---|
| ALS (no fairness) | 0.098 | 0.111 | 0.356 |
| GPT-4o-mini zero-shot | 0.003 | 0.022 | 0.091 |
| GPT-4o-mini improved prompt | 0.018 | 0.078 | 0.170 |
| GPT-5.4-mini | 0.007 | 0.020 | 0.126 |
| **FairTrip (NSW optimizer)** | **0.114** | **0.111** | **0.352** |

FairTrip improved group fairness (NSW) by 16% over the ALS baseline while keeping individual recommendation quality (AUC and MRR) within about 1% of the baseline, showing that fairness can be optimized for without materially sacrificing recommendation quality.

**Notable finding:** an LLM prompted zero-shot, without access to the underlying ratings matrix, performed far worse than structured collaborative filtering on every metric. Improving the prompt (filtering to only highly-rated venues per user, adding explicit fairness instructions) helped substantially, but even the improved LLM variant stayed well below both ALS and FairTrip.

## Limitations

- Both LLM baselines were given a random pool of 100 candidate venues as opaque IDs with no names or descriptions, which limits how meaningful their recommendations could be
- Ground truth for evaluation comes from simulated group scenarios using real individual ratings, not real group travel outcomes
- Scoped to a single city (Philadelphia) due to Yelp's uneven review density across metros

## Tech Stack

Python, Implicit (ALS), pandas, NumPy, OpenAI API (GPT-4o-mini benchmark)

## Repository Structure

```
fairtrip/
├── README.md
├── requirements.txt
├── notebooks/
│   └── FairTrip.ipynb
├── data/
│   └── (Yelp Open Dataset not included due to license terms; see Data Sources below)
└── outputs/
    └── (evaluation results, model comparison tables)
```

## Data Sources

[Yelp Open Dataset](https://www.yelp.com/dataset), licensed for non-commercial academic use.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/FairTrip.ipynb
```
