# Bias Detection in Language Models

Research project systematically testing LLMs for demographic bias using multiple prompt datasets and cross-dataset statistical comparison.

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
cd src && python multi_dataset_experiment.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Findings

- Bias patterns are partially consistent across prompt datasets (33-67% pattern match depending on dimension)
- Occupational context amplifies certain biases (e.g., agentic language differs significantly between datasets for race)
- Gender bias patterns are most consistent across datasets; race/age patterns are more sensitive to prompt framing
- Sentiment range is wider in demographic prompts, suggesting direct framing elicits stronger bias signals

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
- Effect of prompt framing on bias detection sensitivity
- Reproducible research methods with ground-truth validation

Contact: Mike Ichikawa - projects.ichikawa@gmail.com

# 2025-11-01
# 2025-11-01
# 2025-11-06
# 2025-11-12
# 2025-11-17
# 2025-11-23
# 2025-11-28
# 2025-12-04
# 2025-12-09
# 2025-12-14
# 2025-12-20
# 2025-12-26
# 2026-01-02
# 2026-01-08
# 2026-01-13
# 2026-01-18
# 2026-01-23
# 2026-01-29