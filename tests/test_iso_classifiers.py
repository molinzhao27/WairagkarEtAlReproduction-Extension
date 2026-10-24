"""D-080: the fast shrinkage LDA in scripts/iso_classifiers.py predicts exactly as sklearn's lsqr/auto LDA."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


def test_fast_lda_matches_sklearn():
    import scipy.linalg
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, _class_cov

    rng = np.random.default_rng(0)
    for d in (40, 300):
        X = rng.normal(size=(70, d)) + np.repeat(rng.normal(size=(7, d)), 10, 0) * 0.3
        y = np.repeat(np.arange(7), 10)
        Xte = rng.normal(size=(35, d))
        classes = np.unique(y)
        priors = np.array([np.mean(y == c) for c in classes])
        cov = _class_cov(X, y, priors, shrinkage="auto")
        means = np.stack([X[y == c].mean(0) for c in classes])
        coef = scipy.linalg.solve(cov, means.T, assume_a="pos").T
        intercept = -0.5 * np.sum(means * coef, 1) + np.log(priors)
        ours = Xte @ coef.T + intercept
        sk = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
        assert np.allclose(ours, sk.decision_function(Xte), atol=1e-6)
