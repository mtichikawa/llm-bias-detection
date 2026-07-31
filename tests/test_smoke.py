'''
Smoke tests for llm-bias-detection
Validates imports work, core classes instantiate, and experiments run end-to-end.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import random
from typing import Any


class TestImports:
    '''Verify all modules import cleanly'''

    def test_import_experiment(self):
        from experiment import BiasExperiment, PromptGenerator, BiasAnalyzer
        assert BiasExperiment is not None
        assert PromptGenerator is not None
        assert BiasAnalyzer is not None

    def test_import_statistical_analysis(self):
        from statistical_analysis import StatisticalAnalyzer, CrossDatasetComparator
        assert StatisticalAnalyzer is not None
        assert CrossDatasetComparator is not None

    def test_import_occupational_prompts(self):
        from occupational_prompts import OccupationalPromptGenerator, OccupationalMockResponder
        assert OccupationalPromptGenerator is not None
        assert OccupationalMockResponder is not None

    def test_import_multi_dataset(self):
        from multi_dataset_experiment import MultiDatasetExperiment
        assert MultiDatasetExperiment is not None


class TestPromptGenerator:
    '''Test original prompt generation'''

    def test_generates_prompts_for_gender(self):
        from experiment import PromptGenerator
        gen = PromptGenerator()
        prompts = gen.generate_prompts('gender', samples_per_demo=1)
        assert len(prompts) > 0
        assert all('demographic' in p for p in prompts)
        assert all('dimension' in p for p in prompts)

    def test_generates_prompts_for_all_dimensions(self):
        from experiment import PromptGenerator
        gen = PromptGenerator()
        for dim in ['gender', 'age', 'race']:
            prompts = gen.generate_prompts(dim, samples_per_demo=1)
            assert len(prompts) > 0

    def test_invalid_dimension_raises(self):
        from experiment import PromptGenerator
        gen = PromptGenerator()
        with pytest.raises(ValueError):
            gen.generate_prompts('invalid_dimension')


class TestOccupationalPrompts:
    '''Test occupational prompt dataset'''

    def test_generates_prompts(self):
        from occupational_prompts import OccupationalPromptGenerator
        gen = OccupationalPromptGenerator()
        prompts = gen.generate_prompts('gender', samples_per_demo=1)
        assert len(prompts) > 0

    def test_prompts_have_occupation_field(self):
        from occupational_prompts import OccupationalPromptGenerator
        gen = OccupationalPromptGenerator()
        prompts = gen.generate_prompts('gender', samples_per_demo=1)
        assert all('occupation' in p for p in prompts)
        assert all('stereotype_gender' in p for p in prompts)
        assert all('prestige' in p for p in prompts)

    def test_prompts_tagged_as_occupational(self):
        from occupational_prompts import OccupationalPromptGenerator
        gen = OccupationalPromptGenerator()
        prompts = gen.generate_prompts('gender', samples_per_demo=1)
        assert all(p['dataset'] == 'occupational' for p in prompts)

    def test_mock_responder_returns_strings(self):
        from occupational_prompts import OccupationalPromptGenerator, OccupationalMockResponder
        gen = OccupationalPromptGenerator()
        responder = OccupationalMockResponder()
        prompts = gen.generate_prompts('gender', samples_per_demo=1)
        for prompt in prompts[:5]:
            response = responder.generate_response(prompt)
            assert isinstance(response, str)
            assert len(response) > 10

    def test_invalid_dimension_raises(self):
        from occupational_prompts import OccupationalPromptGenerator
        gen = OccupationalPromptGenerator()
        with pytest.raises(ValueError):
            gen.generate_prompts('invalid')


class TestBiasAnalyzer:
    '''Test bias analysis'''

    def test_analyze_response(self):
        from experiment import BiasAnalyzer
        analyzer = BiasAnalyzer()
        result = analyzer.analyze_response("This person is confident, assertive, and caring.")
        assert 'bias_scores' in result
        assert 'sentiment' in result
        assert 'word_counts' in result
        assert result['word_counts']['agentic'] >= 2
        assert result['word_counts']['communal'] >= 1


class TestBiasExperiment:
    '''Test full experiment pipeline'''

    def test_run_experiment_gender(self):
        random.seed(42)
        from experiment import BiasExperiment
        exp = BiasExperiment()
        exp.run_experiment('gender', mock_responses=True, dataset_label='demographic')
        assert len(exp.results) > 0

    def test_generate_report(self):
        random.seed(42)
        from experiment import BiasExperiment
        exp = BiasExperiment()
        exp.run_experiment('gender', mock_responses=True)
        report = exp.generate_report('gender')
        assert 'comparison' in report
        assert 'findings' in report
        assert len(report['demographics']) > 0

    def test_dataset_filtering(self):
        random.seed(42)
        from experiment import BiasExperiment
        exp = BiasExperiment()
        exp.run_experiment('gender', mock_responses=True, dataset_label='demographic')
        exp.run_experiment('gender', mock_responses=True, dataset_label='other')
        demo_results = exp.get_results_by_dataset('demographic')
        other_results = exp.get_results_by_dataset('other')
        assert len(demo_results) > 0
        assert len(other_results) > 0
        assert len(demo_results) + len(other_results) == len(exp.results)


class TestCrossDatasetComparator:
    '''Test cross-dataset comparison'''

    def test_compare_datasets(self):
        from statistical_analysis import CrossDatasetComparator
        comparator = CrossDatasetComparator()

        ds1 = {
            'male': {'avg_sentiment': 0.8, 'avg_length': 20, 'n': 10,
                     'bias_scores': {'agentic': 0.4, 'communal': 0.1, 'competence': 0.3, 'warmth': 0.1, 'negative': 0.0}},
            'female': {'avg_sentiment': 0.5, 'avg_length': 18, 'n': 10,
                       'bias_scores': {'agentic': 0.1, 'communal': 0.4, 'competence': 0.2, 'warmth': 0.3, 'negative': 0.0}},
        }
        ds2 = {
            'male': {'avg_sentiment': 0.75, 'avg_length': 22, 'n': 10,
                     'bias_scores': {'agentic': 0.35, 'communal': 0.15, 'competence': 0.35, 'warmth': 0.05, 'negative': 0.0}},
            'female': {'avg_sentiment': 0.45, 'avg_length': 19, 'n': 10,
                       'bias_scores': {'agentic': 0.05, 'communal': 0.45, 'competence': 0.15, 'warmth': 0.35, 'negative': 0.0}},
        }

        comparison = comparator.compare_datasets(ds1, ds2)
        assert 'sentiment_comparison' in comparison
        assert 'bias_score_comparison' in comparison
        assert 'pattern_consistency' in comparison
        assert 'summary' in comparison

    def test_generate_report_text(self):
        from statistical_analysis import CrossDatasetComparator
        comparator = CrossDatasetComparator()

        ds1 = {
            'male': {'avg_sentiment': 0.8, 'avg_length': 20, 'n': 10,
                     'bias_scores': {'agentic': 0.4, 'communal': 0.1, 'competence': 0.3, 'warmth': 0.1, 'negative': 0.0}},
            'female': {'avg_sentiment': 0.5, 'avg_length': 18, 'n': 10,
                       'bias_scores': {'agentic': 0.1, 'communal': 0.4, 'competence': 0.2, 'warmth': 0.3, 'negative': 0.0}},
        }
        ds2 = dict(ds1)  # Same data for simplicity

        comparison = comparator.compare_datasets(ds1, ds2)
        text = comparator.generate_report_text(comparison)
        assert 'Cross-Dataset' in text
        assert 'demographic' in text
        assert 'occupational' in text


class TestMultiDatasetExperiment:
    '''Test the full multi-dataset pipeline'''

    def test_full_run(self):
        random.seed(42)
        from multi_dataset_experiment import MultiDatasetExperiment
        runner = MultiDatasetExperiment()
        results = runner.run_full_experiment(
            dimensions=['gender'],
            samples_per_demo=1
        )
        assert 'reports_by_dimension' in results
        assert 'cross_dataset_comparisons' in results
        assert 'gender' in results['reports_by_dimension']

    def test_save_results(self, tmp_path: Any):
        random.seed(42)
        from multi_dataset_experiment import MultiDatasetExperiment
        runner = MultiDatasetExperiment()
        results = runner.run_full_experiment(
            dimensions=['gender'],
            samples_per_demo=1
        )
        filepath = str(tmp_path / 'test_results.json')
        runner.save_results(results, filepath)
        assert os.path.exists(filepath)


class TestStatisticalAnalyzer:
    '''Test statistical analysis functions'''

    def test_t_test(self):
        from statistical_analysis import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        result = analyzer.t_test_two_groups([0.8, 0.7, 0.9, 0.75], [0.4, 0.5, 0.3, 0.45])
        assert 'p_value' in result
        assert 'effect_size' in result
        assert result['significant'] == True  # Clear difference

    def test_anova(self):
        from statistical_analysis import StatisticalAnalyzer
        import numpy as np
        np.random.seed(42)
        analyzer = StatisticalAnalyzer()
        result = analyzer.anova_multiple_groups(
            [0.8, 0.7, 0.9], [0.4, 0.5, 0.3], [0.6, 0.5, 0.7]
        )
        assert 'f_statistic' in result
        assert 'p_value' in result
