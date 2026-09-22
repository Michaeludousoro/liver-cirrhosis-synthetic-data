"""Regression checks for fold isolation and truthful uncertainty reporting."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from src.statistical_analysis import cross_validate_scenarios


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'x': np.arange(40), 'Status': [0, 1] * 20})
        self.classifiers = patch('src.statistical_analysis._build_classifiers',
                                 return_value={'LR': LogisticRegression()})
        self.classifiers.start()
        self.addCleanup(self.classifiers.stop)

    def test_rejects_full_training_pool(self):
        with self.assertRaisesRegex(ValueError, 'precomputed'):
            cross_validate_scenarios(self.data, {'bad':self.data}, feat_cols=['x'])

    def test_factory_sees_only_training_and_runs_once_per_fold(self):
        calls = []
        expected = list(StratifiedKFold(5, shuffle=True, random_state=42).split(self.data, self.data.Status))
        def factory(real, fold, seed):
            tr, va = expected[fold - 1]
            self.assertEqual(set(real.index), set(tr))
            self.assertFalse(set(real.index) & set(va))
            calls.append(fold)
            return {'fresh':real.copy()}
        result = cross_validate_scenarios(self.data, {'A':None, 'B':'fresh', 'C':'fresh'},
                                         feat_cols=['x'], fold_factory=factory)
        self.assertEqual(calls, [1,2,3,4,5])
        self.assertEqual(len(result.attrs['fold_results']), 15)
        self.assertNotIn('CI Lower', result)
        self.assertEqual(sorted(sum(result.attrs['validation_indices'], [])), list(range(40)))

    def test_no_silent_empty_pool_fallback(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            cross_validate_scenarios(self.data, {'B':'fresh'}, feat_cols=['x'],
                                     fold_factory=lambda *args: {'fresh':self.data.iloc[:0]})

    def test_rejects_target_as_predictor(self):
        with self.assertRaisesRegex(ValueError, 'target'):
            cross_validate_scenarios(self.data, {'A':None}, feat_cols=['x','Status'])


if __name__ == '__main__':
    unittest.main()
