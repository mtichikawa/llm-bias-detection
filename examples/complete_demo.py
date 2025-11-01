'''
Complete Bias Detection Demo
Shows full workflow with visualizations
'''

import sys
sys.path.append('../src')

from experiment import BiasExperiment
from statistical_analysis import StatisticalAnalyzer
import numpy as np

print('LLM Bias Detection - Complete Demo')
print('=' * 60)

# Create experiment
experiment = BiasExperiment()

# Run experiment on gender
print('\n🔬 Testing Gender Bias...')
experiment.run_experiment('gender', mock_responses=True)

# Generate report
report = experiment.generate_report('gender')

print(f'\n📊 Results:')
print(f'  Total samples: {report["total_samples"]}')
print(f'  Demographics: {", ".join(report["demographics"])}')

print('\n💡 Key Findings:')
for finding in report['findings']:
    print(f'  • {finding}')

# Statistical analysis
print('\n📈 Statistical Analysis:')
analyzer = StatisticalAnalyzer()
comparison = report['comparison']

# Show sentiment by demographic
print('\n  Sentiment Scores:')
for demo, data in comparison.items():
    print(f'    {demo}: {data["avg_sentiment"]:.3f}')

# Show bias scores
print('\n  Agentic Language Usage:')
for demo, data in comparison.items():
    score = data['bias_scores']['agentic']
    print(f'    {demo}: {score:.3f}')

print('\n  Communal Language Usage:')
for demo, data in comparison.items():
    score = data['bias_scores']['communal']
    print(f'    {demo}: {score:.3f}')

# Save results
experiment.save_results('results/bias_experiment.json')

print('\n✅ Demo complete!')
print('Check results/ folder for saved data')
