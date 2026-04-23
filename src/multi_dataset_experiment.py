'''
Multi-Dataset Bias Experiment
Runs bias detection across both the demographic and occupational prompt datasets,
then performs cross-dataset comparison.
'''

import json
import random
from pathlib import Path
from typing import Dict, List

from experiment import BiasExperiment, BiasAnalyzer
from occupational_prompts import OccupationalPromptGenerator, OccupationalMockResponder
from statistical_analysis import StatisticalAnalyzer, CrossDatasetComparator


class MultiDatasetExperiment:
    '''Run bias experiments across multiple prompt datasets and compare'''

    def __init__(self):
        self.experiment = BiasExperiment()
        self.occ_generator = OccupationalPromptGenerator()
        self.occ_responder = OccupationalMockResponder()
        self.analyzer = BiasAnalyzer()
        self.stat_analyzer = StatisticalAnalyzer()
        self.comparator = CrossDatasetComparator()

    def run_full_experiment(self, dimensions: List[str] = None,
                           samples_per_demo: int = 3) -> Dict:
        '''Run both datasets across all dimensions and compare.

        Args:
            dimensions: List of dimensions to test (default: all three)
            samples_per_demo: Samples per demographic per template

        Returns:
            Full experiment results with cross-dataset comparisons
        '''
        if dimensions is None:
            dimensions = ['gender', 'age', 'race']

        results = {
            'dimensions': dimensions,
            'datasets': ['demographic', 'occupational'],
            'reports_by_dimension': {},
            'cross_dataset_comparisons': {}
        }

        for dimension in dimensions:
            print(f"\n{'='*60}")
            print(f"  DIMENSION: {dimension.upper()}")
            print(f"{'='*60}")

            # --- Dataset 1: Demographic prompts (original) ---
            self.experiment.run_experiment(
                dimension, mock_responses=True, dataset_label='demographic'
            )

            # --- Dataset 2: Occupational prompts ---
            self._run_occupational_experiment(dimension, samples_per_demo)

            # Generate reports for each dataset
            demo_report = self.experiment.generate_report(dimension, dataset_label='demographic')
            occ_report = self.experiment.generate_report(dimension, dataset_label='occupational')

            results['reports_by_dimension'][dimension] = {
                'demographic': demo_report,
                'occupational': occ_report
            }

            # Cross-dataset comparison
            if demo_report['comparison'] and occ_report['comparison']:
                comparison = self.comparator.compare_datasets(
                    demo_report['comparison'],
                    occ_report['comparison'],
                    dataset1_name='demographic',
                    dataset2_name='occupational'
                )
                results['cross_dataset_comparisons'][dimension] = comparison

        return results

    def _run_occupational_experiment(self, dimension: str, samples_per_demo: int = 2):
        '''Run the occupational prompt dataset through the analysis pipeline'''
        print(f'\n  Running occupational bias experiment: {dimension}')

        prompts = self.occ_generator.generate_prompts(dimension, samples_per_demo=samples_per_demo)
        print(f'    Generated {len(prompts)} occupational prompts')

        for prompt in prompts:
            response = self.occ_responder.generate_response(prompt)
            analysis = self.analyzer.analyze_response(response)

            self.experiment.results.append({
                'prompt': prompt,
                'response': response,
                'analysis': analysis
            })

        print(f'    Analyzed {len(prompts)} occupational responses')

    def print_comparison_report(self, results: Dict):
        '''Print formatted cross-dataset comparison'''
        print("\n")
        print("=" * 60)
        print("  CROSS-DATASET COMPARISON REPORT")
        print("=" * 60)

        for dimension, comparison in results['cross_dataset_comparisons'].items():
            print(f"\n--- {dimension.upper()} ---")
            report_text = self.comparator.generate_report_text(comparison)
            # Indent each line
            for line in report_text.split('\n'):
                print(f"  {line}")

    def save_results(self, results: Dict, filepath: str = 'results/multi_dataset_experiment.json'):
        '''Save full results to JSON'''
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Make results JSON-serializable (strip numpy types)
        serializable = json.loads(json.dumps(results, default=str))

        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f'\n  Saved results to {filepath}')


def demo():
    '''Run multi-dataset experiment demo'''
    print("LLM Bias Detection - Multi-Dataset Experiment")
    print("=" * 60)
    print("Datasets:")
    print("  1. Demographic prompts (recommendation letters, leadership, career)")
    print("  2. Occupational prompts (Winogender-style role congruity)")
    print()

    random.seed(42)  # Reproducibility

    runner = MultiDatasetExperiment()
    results = runner.run_full_experiment(
        dimensions=['gender', 'age', 'race'],
        samples_per_demo=2
    )

    # Print comparison
    runner.print_comparison_report(results)

    # Save
    runner.save_results(results)

    print("\nExperiment complete!")


if __name__ == '__main__':
    demo()
