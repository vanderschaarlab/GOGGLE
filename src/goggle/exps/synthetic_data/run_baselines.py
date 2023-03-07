import sys
import json
import numpy as np
import pandas as pd

import synthcity.logger as log
from synthcity.plugins import Plugins
from synthcity.metrics import eval_statistical
from synthcity.metrics import eval_performance
from synthcity.metrics import eval_detection
from synthcity.plugins.core.dataloader import GenericDataLoader

from sklearn.model_selection import train_test_split

generators = Plugins()


def evaluate_baselines(data, seed):
    quality_evaluator = eval_statistical.AlphaPrecision()
    xgb_evaluator = eval_performance.PerformanceEvaluatorXGB()
    linear_evaluator = eval_performance.PerformanceEvaluatorLinear()
    mlp_evaluator = eval_performance.PerformanceEvaluatorMLP()

    xgb_detector = eval_detection.SyntheticDetectionXGB()
    mlp_detector = eval_detection.SyntheticDetectionMLP()
    gmm_detector = eval_detection.SyntheticDetectionGMM()

    X_train, X_test = train_test_split(
        data, random_state=seed+42, test_size=0.33, shuffle=False)

    results = {}

    for model in ['nflow', 'ctgan', 'tvae', 'bayesian_network', 'copulagan']:
        gen = generators.get(model)
        gen.fit(X_train)
        X_synth = gen.generate(count=X_test.shape[0])

        X_test_loader = GenericDataLoader(
            X_test,
            target_column="target",
        )
        X_synth_loader = GenericDataLoader(
            X_synth,
            target_column="target",
        )

        xgb_score = xgb_evaluator.evaluate(X_test_loader, X_synth_loader)
        linear_score = linear_evaluator.evaluate(X_test_loader, X_synth_loader)
        mlp_score = mlp_evaluator.evaluate(X_test_loader, X_synth_loader)
        xgb_det = xgb_detector.evaluate(X_test_loader, X_synth_loader)
        mlp_det = mlp_detector.evaluate(X_test_loader, X_synth_loader)
        gmm_det = gmm_detector.evaluate(X_test_loader, X_synth_loader)
        data_qual = quality_evaluator.evaluate(X_test_loader, X_synth_loader)

        gt_perf = np.mean([xgb_score['gt'],
                           linear_score['gt'],
                           mlp_score['gt']])
        synth_perf = np.mean([xgb_score['syn_ood'],
                              linear_score['syn_ood'],
                              mlp_score['syn_ood']])
        det_score = np.mean([xgb_det['mean'],
                             gmm_det['mean'],
                             mlp_det['mean']])
        qual_score = np.mean(list(data_qual.values()))

        results[model] = [gt_perf, synth_perf, det_score, qual_score]

    return results


if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer

    dataset = 'breast'
    X = pd.read_csv('../data/breast_cancer_2.csv')

    ind = list(range(len(X.columns)))
    ind = [x for x in ind if x != X.columns.get_loc("target")]
    col_list = X.columns[ind]
    ct = ColumnTransformer([('scaler', StandardScaler(), col_list)], remainder='passthrough')

    X_ = ct.fit_transform(X)
    X = pd.DataFrame(X_, index=X.index, columns=X.columns)

    X.head()

    print(evaluate_baselines(X, 0))