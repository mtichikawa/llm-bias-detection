# Bias Detection in Language Models

A pipeline for detecting demographic bias in text responses, using two prompt datasets and cross-dataset statistical comparison. It is validated on synthetic responses with deliberately planted bias. It does not call a real LLM yet, so it makes no claim about how any real model behaves.

## Methodology

### Dual-Dataset Approach

1. **Demographic Prompts** (Dataset 1) — 5 templates testing recommendation letters, leadership, career advice, technical evaluation, and stress response across gender/age/race dimensions.

2. **Occupational Prompts** (Dataset 2, Winogender-style) — 5 templates testing role congruity, competence judgments, social perception, performance evaluation, and promotion decisions across 12 stereotyped occupations (nurse, engineer, CEO, etc.) combined with gender/age/race dimensions.

### Analysis Pipeline

- **Bias Lexicon Scoring** — Agentic, communal, competence, warmth, and negative language detection
- **Sentiment Analysis** — Positive/negative language ratio per demographic group
- **Statistical Testing** — T-tests, ANOVA, Cohen's d effect sizes
- **Cross-Dataset Comparison** — Pattern consistency scoring, bias amplification detection, per-category statistical tests between datasets

### Key Design: Mock Responses as Ground Truth

Both datasets use intentionally biased mock response generators that embed known stereotypes (e.g., males get agentic language, females get communal language, role-incongruent pairings get warmth-over-competence framing). This validates the detection pipeline against known ground-truth bias patterns without requiring paid API calls.

## Features

- Two independent prompt datasets (demographic + occupational/Winogender-style)
- 12 stereotyped occupations with role congruity scoring
- Cross-dataset comparison with pattern consistency metrics
- Statistical significance testing (t-test, ANOVA, effect sizes)
- Automated multi-dataset reporting
- 22 pytest smoke tests
- Jupyter notebook analysis

## Quick Start

```bash
pip install -r requirements.txt

# Single-dataset demo (original)
python examples/complete_demo.py

# Multi-dataset experiment with cross-dataset comparison
python src/multi_dataset_experiment.py
```

## Running Tests

```bash
pytest tests/ -v
```

## What the Validation Shows

These results come from the mock responses, so they show what the pipeline can detect, not how real LLMs behave.

- The pipeline recovers the planted gender pattern. In `examples/complete_demo.py`, mock male responses score 1.0 on agentic language and mock female responses 0.0, and the reverse holds for communal language.
- Cross-dataset pattern consistency comes out at 33-67% depending on dimension. That gap reflects the two sets of mock templates differing, not prompt framing in a real model.
- The cross-dataset comparison flags statistically significant category differences between datasets, which is the check that would matter once real responses are plugged in.

## Project Structure

```
src/
  experiment.py              # Core BiasExperiment, PromptGenerator, BiasAnalyzer
  occupational_prompts.py    # Dataset 2: Winogender-style occupational prompts + mock responder
  multi_dataset_experiment.py # Multi-dataset runner + comparison reporting
  statistical_analysis.py    # StatisticalAnalyzer + CrossDatasetComparator
tests/
  test_smoke.py              # 22 smoke tests
examples/
  complete_demo.py           # Single-dataset demo
results/                     # JSON output from experiments
```

## What I Learned

- Systematic bias testing frameworks with multiple prompt datasets
- Cross-dataset statistical comparison methodology
- Role congruity theory applied to NLP evaluation
- Validating a detector against planted ground truth

Contact: Mike Ichikawa - projects.ichikawa@gmail.com

> _Note: current responses are deterministic mock templates designed to validate the detection pipeline. With `mock_responses=False` the code returns a placeholder string; there is no LLM API integration._
