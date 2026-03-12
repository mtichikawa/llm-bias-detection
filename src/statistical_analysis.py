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
