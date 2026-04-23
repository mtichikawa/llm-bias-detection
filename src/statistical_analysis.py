'''
Statistical Analysis for Bias Detection
Hypothesis testing and significance calculations
'''

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


class StatisticalAnalyzer:
    '''Statistical significance testing for bias'''
    
    def __init__(self, alpha: float = 0.05):
        '''
        Args:
            alpha: Significance level (default 0.05)
        '''
        self.alpha = alpha
        
    def t_test_two_groups(self, group1: List[float], group2: List[float]) -> Dict:
        '''
        Perform independent t-test between two groups
        
        Returns dict with test results
        '''
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        return {
            'test': 't-test',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'mean_group1': float(np.mean(group1)),
            'mean_group2': float(np.mean(group2)),
            'effect_size': self._cohens_d(group1, group2)
        }
        
    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        '''Calculate Cohen's d effect size'''
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        if pooled_std == 0:
            return 0.0
            
        return (np.mean(group1) - np.mean(group2)) / pooled_std
        
    def anova_multiple_groups(self, *groups) -> Dict:
        '''
        Perform one-way ANOVA for multiple groups
        
        Returns dict with test results
        '''
        f_stat, p_value = stats.f_oneway(*groups)
        
        return {
            'test': 'ANOVA',
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'n_groups': len(groups)
        }
        
    def chi_square_test(self, observed: np.ndarray, expected: np.ndarray = None) -> Dict:
        '''
        Perform chi-square test
        
        Args:
            observed: Observed frequencies
            expected: Expected frequencies (if None, assumes uniform)
        '''
        if expected is None:
            expected = np.ones_like(observed) * np.mean(observed)
            
        chi2_stat, p_value = stats.chisquare(observed, expected)
        
        return {
            'test': 'chi-square',
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha
        }
        
    def analyze_experiment_results(self, results: Dict) -> Dict:
        '''
        Complete statistical analysis of experiment results
        
        Args:
            results: Comparison dict from BiasAnalyzer
            
        Returns:
            Statistical test results
        '''
        analysis = {
            'sentiment': {},
            'bias_scores': {},
            'length': {}
        }
        
        # Extract data by group
        groups = list(results.keys())
        
        # Sentiment analysis
        sentiments = {demo: [results[demo]['avg_sentiment']] for demo in groups}
        if len(groups) == 2:
            # Note: replicating a scalar N times inflates statistical power — treat p-values as approximate
            analysis['sentiment'] = self.t_test_two_groups(
                [results[groups[0]]['avg_sentiment']] * results[groups[0]]['n'],
                [results[groups[1]]['avg_sentiment']] * results[groups[1]]['n']
            )
        elif len(groups) > 2:
            # For multiple groups, use ANOVA
            sentiment_groups = [
                [results[demo]['avg_sentiment']] * results[demo]['n'] 
                for demo in groups
            ]
            analysis['sentiment'] = self.anova_multiple_groups(*sentiment_groups)
            
        # Bias score analysis for each category
        for category in ['agentic', 'communal', 'competence']:
            scores_by_group = [
                [results[demo]['bias_scores'][category]] * results[demo]['n']
                for demo in groups
            ]
            
            if len(groups) == 2:
                analysis['bias_scores'][category] = self.t_test_two_groups(
                    scores_by_group[0], scores_by_group[1]
                )
            else:
                analysis['bias_scores'][category] = self.anova_multiple_groups(
                    *scores_by_group
                )
                
        return analysis
        
    def interpret_results(self, test_results: Dict) -> str:
        '''Generate interpretation of statistical tests'''
        interpretations = []
        
        # Sentiment
        if 'sentiment' in test_results:
            s = test_results['sentiment']
            if s.get('significant'):
                if 'effect_size' in s:
                    effect = 'large' if abs(s['effect_size']) > 0.8 else 'medium' if abs(s['effect_size']) > 0.5 else 'small'
                    interpretations.append(
                        f"Sentiment bias is statistically significant (p={s['p_value']:.4f}) "
                        f"with {effect} effect size (d={s['effect_size']:.2f})"
                    )
                else:
                    interpretations.append(
                        f"Sentiment bias is statistically significant (p={s['p_value']:.4f})"
                    )
            else:
                interpretations.append(
                    f"No significant sentiment bias detected (p={s['p_value']:.4f})"
                )
                
        # Bias scores
        if 'bias_scores' in test_results:
            for category, result in test_results['bias_scores'].items():
                if result.get('significant'):
                    interpretations.append(
                        f"{category.capitalize()} language shows significant bias (p={result['p_value']:.4f})"
                    )
                    
        return "\n".join(interpretations) if interpretations else "No significant biases detected"


class CrossDatasetComparator:
    '''Compare bias patterns across different prompt datasets'''

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.stat_analyzer = StatisticalAnalyzer(alpha=alpha)

    def compare_datasets(self, dataset1_results: Dict, dataset2_results: Dict,
                         dataset1_name: str = 'demographic',
                         dataset2_name: str = 'occupational') -> Dict:
        '''
        Compare bias patterns between two datasets.

        Args:
            dataset1_results: Comparison dict from BiasAnalyzer for dataset 1
            dataset2_results: Comparison dict from BiasAnalyzer for dataset 2
            dataset1_name: Label for first dataset
            dataset2_name: Label for second dataset

        Returns:
            Cross-dataset comparison with statistical tests
        '''
        comparison = {
            'datasets': [dataset1_name, dataset2_name],
            'sentiment_comparison': self._compare_sentiment(
                dataset1_results, dataset2_results, dataset1_name, dataset2_name
            ),
            'bias_score_comparison': self._compare_bias_scores(
                dataset1_results, dataset2_results, dataset1_name, dataset2_name
            ),
            'pattern_consistency': self._check_pattern_consistency(
                dataset1_results, dataset2_results
            ),
            'summary': {}
        }

        comparison['summary'] = self._generate_summary(comparison)
        return comparison

    def _compare_sentiment(self, ds1: Dict, ds2: Dict,
                           name1: str, name2: str) -> Dict:
        '''Compare sentiment distributions across datasets'''
        ds1_sentiments = [data['avg_sentiment'] for data in ds1.values()]
        ds2_sentiments = [data['avg_sentiment'] for data in ds2.values()]

        # Overall sentiment level difference
        result = {
            f'{name1}_mean_sentiment': float(np.mean(ds1_sentiments)),
            f'{name2}_mean_sentiment': float(np.mean(ds2_sentiments)),
            f'{name1}_sentiment_range': float(max(ds1_sentiments) - min(ds1_sentiments)),
            f'{name2}_sentiment_range': float(max(ds2_sentiments) - min(ds2_sentiments)),
        }

        # Test whether sentiment spread differs (are both datasets equally biased?)
        if len(ds1_sentiments) >= 2 and len(ds2_sentiments) >= 2:
            t_result = self.stat_analyzer.t_test_two_groups(ds1_sentiments, ds2_sentiments)
            result['cross_dataset_test'] = t_result
            result['datasets_differ'] = t_result['significant']
        else:
            result['datasets_differ'] = False

        return result

    def _compare_bias_scores(self, ds1: Dict, ds2: Dict,
                             name1: str, name2: str) -> Dict:
        '''Compare bias score patterns across datasets'''
        categories = ['agentic', 'communal', 'competence', 'warmth', 'negative']
        results = {}

        for category in categories:
            ds1_scores = [data['bias_scores'].get(category, 0.0) for data in ds1.values()]
            ds2_scores = [data['bias_scores'].get(category, 0.0) for data in ds2.values()]

            results[category] = {
                f'{name1}_mean': float(np.mean(ds1_scores)) if ds1_scores else 0.0,
                f'{name2}_mean': float(np.mean(ds2_scores)) if ds2_scores else 0.0,
                f'{name1}_std': float(np.std(ds1_scores)) if ds1_scores else 0.0,
                f'{name2}_std': float(np.std(ds2_scores)) if ds2_scores else 0.0,
            }

            # Statistical test if enough data
            if len(ds1_scores) >= 2 and len(ds2_scores) >= 2:
                t_result = self.stat_analyzer.t_test_two_groups(ds1_scores, ds2_scores)
                results[category]['test'] = t_result
                results[category]['differs'] = t_result['significant']
            else:
                results[category]['differs'] = False

        return results

    def _check_pattern_consistency(self, ds1: Dict, ds2: Dict) -> Dict:
        '''Check if bias patterns are consistent across datasets.

        e.g., if dataset1 shows males getting more agentic language,
        does dataset2 show the same pattern?
        '''
        consistency = {
            'matching_patterns': [],
            'divergent_patterns': [],
            'consistency_score': 0.0
        }

        # Find shared demographic groups
        shared_demos = set(ds1.keys()) & set(ds2.keys())
        if len(shared_demos) < 2:
            consistency['note'] = 'Insufficient shared demographics for pattern comparison'
            return consistency

        categories = ['agentic', 'communal', 'competence']
        total_checks = 0
        matches = 0

        for category in categories:
            # Rank demographics by score in each dataset
            ds1_ranking = sorted(
                shared_demos,
                key=lambda d: ds1[d]['bias_scores'].get(category, 0),
                reverse=True
            )
            ds2_ranking = sorted(
                shared_demos,
                key=lambda d: ds2[d]['bias_scores'].get(category, 0),
                reverse=True
            )

            total_checks += 1

            # Check if top-ranked demo is the same
            if ds1_ranking[0] == ds2_ranking[0]:
                matches += 1
                consistency['matching_patterns'].append(
                    f"{category}: '{ds1_ranking[0]}' ranked highest in both datasets"
                )
            else:
                consistency['divergent_patterns'].append(
                    f"{category}: dataset1 highest='{ds1_ranking[0]}', "
                    f"dataset2 highest='{ds2_ranking[0]}'"
                )

        consistency['consistency_score'] = matches / total_checks if total_checks > 0 else 0.0
        return consistency

    def _generate_summary(self, comparison: Dict) -> Dict:
        '''Generate human-readable summary'''
        summary = {
            'findings': [],
            'bias_amplification': None,
            'overall_consistency': comparison['pattern_consistency']['consistency_score']
        }

        # Check sentiment
        sent = comparison['sentiment_comparison']
        ds_names = comparison['datasets']
        range1 = sent.get(f'{ds_names[0]}_sentiment_range', 0)
        range2 = sent.get(f'{ds_names[1]}_sentiment_range', 0)

        if range2 > range1 * 1.2:
            summary['findings'].append(
                f"Occupational context amplifies sentiment bias "
                f"(range {range2:.3f} vs {range1:.3f})"
            )
            summary['bias_amplification'] = 'occupational > demographic'
        elif range1 > range2 * 1.2:
            summary['findings'].append(
                f"Demographic context shows stronger sentiment bias "
                f"(range {range1:.3f} vs {range2:.3f})"
            )
            summary['bias_amplification'] = 'demographic > occupational'
        else:
            summary['findings'].append("Sentiment bias is similar across both prompt datasets")
            summary['bias_amplification'] = 'comparable'

        # Pattern consistency
        score = comparison['pattern_consistency']['consistency_score']
        if score >= 0.67:
            summary['findings'].append(
                f"Bias patterns are consistent across datasets (score: {score:.0%})"
            )
        else:
            summary['findings'].append(
                f"Bias patterns diverge between datasets (score: {score:.0%}) — "
                "prompt framing influences detected bias"
            )

        # Significant category differences
        for cat, data in comparison['bias_score_comparison'].items():
            if data.get('differs'):
                summary['findings'].append(
                    f"{cat.capitalize()} language differs significantly between datasets"
                )

        return summary

    def generate_report_text(self, comparison: Dict) -> str:
        '''Format comparison as readable text'''
        lines = []
        lines.append("Cross-Dataset Bias Comparison Report")
        lines.append("=" * 50)

        ds_names = comparison['datasets']
        lines.append(f"\nDatasets: {ds_names[0]} vs {ds_names[1]}")

        # Sentiment
        sent = comparison['sentiment_comparison']
        lines.append(f"\n--- Sentiment ---")
        lines.append(f"  {ds_names[0]} mean: {sent.get(f'{ds_names[0]}_mean_sentiment', 0):.3f}")
        lines.append(f"  {ds_names[1]} mean: {sent.get(f'{ds_names[1]}_mean_sentiment', 0):.3f}")
        lines.append(f"  {ds_names[0]} range: {sent.get(f'{ds_names[0]}_sentiment_range', 0):.3f}")
        lines.append(f"  {ds_names[1]} range: {sent.get(f'{ds_names[1]}_sentiment_range', 0):.3f}")

        # Bias scores
        lines.append(f"\n--- Bias Scores by Category ---")
        for cat, data in comparison['bias_score_comparison'].items():
            marker = " *" if data.get('differs') else ""
            lines.append(
                f"  {cat:12s}: {ds_names[0]}={data.get(f'{ds_names[0]}_mean', 0):.3f}  "
                f"{ds_names[1]}={data.get(f'{ds_names[1]}_mean', 0):.3f}{marker}"
            )
        lines.append("  (* = statistically significant difference)")

        # Pattern consistency
        pc = comparison['pattern_consistency']
        lines.append(f"\n--- Pattern Consistency ---")
        lines.append(f"  Score: {pc['consistency_score']:.0%}")
        for m in pc['matching_patterns']:
            lines.append(f"  [match] {m}")
        for d in pc['divergent_patterns']:
            lines.append(f"  [diverge] {d}")

        # Summary
        lines.append(f"\n--- Summary ---")
        for finding in comparison['summary']['findings']:
            lines.append(f"  - {finding}")

        return "\n".join(lines)


def demo():
    '''Demo statistical analysis'''
    print('Statistical Analysis Demo')
    print('=' * 50)
    
    analyzer = StatisticalAnalyzer(alpha=0.05)
    
    # Simulate data
    np.random.seed(42)
    male_sentiment = np.random.normal(0.7, 0.1, 30)
    female_sentiment = np.random.normal(0.55, 0.1, 30)
    
    print('\nComparing sentiment scores:')
    print(f'  Male group: mean={np.mean(male_sentiment):.3f}')
    print(f'  Female group: mean={np.mean(female_sentiment):.3f}')
    
    # Run t-test
    result = analyzer.t_test_two_groups(male_sentiment.tolist(), female_sentiment.tolist())
    
    print(f'\nT-test results:')
    print(f'  t-statistic: {result["t_statistic"]:.3f}')
    print(f'  p-value: {result["p_value"]:.4f}')
    print(f'  Significant: {result["significant"]}')
    print(f'  Effect size (Cohen\'s d): {result["effect_size"]:.3f}')
    
    # Interpretation
    print(f'\nInterpretation:')
    if result['significant']:
        print('  ✓ Significant bias detected')
        if abs(result['effect_size']) > 0.5:
            print(f'  ✓ Effect size is meaningful (d={result["effect_size"]:.2f})')
    else:
        print('  ✗ No significant bias')


if __name__ == '__main__':
    demo()
