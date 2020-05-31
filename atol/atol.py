# coding: utf-8
"""
@author: Martin Royer
@copyright: INRIA 2019-2020
"""

import numpy as np

from sklearn.metrics import pairwise
from sklearn.base import BaseEstimator, TransformerMixin


def _lapl_contrast(measure, centers, inertias, eps=1e-8):
    return np.exp(-np.sqrt(pairwise.pairwise_distances(measure, Y=centers) / (inertias + eps)))

def _gaus_contrast(measure, centers, inertias, eps=1e-8):
    return np.exp(-pairwise.pairwise_distances(measure, Y=centers) / (inertias + eps))

def _indicator_contrast(diags, centers, inertias, eps=1e-8):
    pair_dist = pairwise.pairwise_distances(diags, Y=centers)
    flat_circ = (pair_dist < (inertias+eps)).astype(int)
    robe_curve = np.positive((2-pair_dist/(inertias+eps))*((inertias+eps) < pair_dist).astype(int))
    return flat_circ + robe_curve

def _cloud_weighting(measure):
    return np.ones(shape=measure.shape[0])

def _measure_weighting(measure):
    return np.ones(shape=measure.shape[0]) / measure.shape[0]

def _iidproba_weighting(measure):
    return np.ones(shape=measure.shape[0]) / measure.shape[0]**2


class Atol(BaseEstimator, TransformerMixin):
    """
    This class allows to vectorise measures (e.g. point clouds, persistence diagrams, etc) after a quantisation step.

    ATOL paper: https://arxiv.org/abs/1909.13472
    """
    def __init__(self, quantiser, weighting_method="cloud", contrast="gaus"):
        """
        Constructor for the Atol measure vectorisation class.

        Parameters:
            quantiser (Object): Object with `fit` (sklearn API consistent) and `cluster_centers` and `n_clusters`
                attributes (default: MiniBatchKMeans()). This object will be fitted by the function `fit`.
            weighting_method (function): constant generic function for weighting the measure points
                (default: constant function, i.e. the measure is seen as a point cloud by default).
                This will have no impact if weights are provided along with measures all the way: `fit` and `transform`.
            contrast (string): constant function for evaluating proximity of a measure with respect to centers
                (default: laplacian contrast function, see page 3 in the ATOL paper).
        """
        self.quantiser = quantiser
        self.contrast = {
            "gaus": _gaus_contrast,
            "lapl": _lapl_contrast,
            "indi": _indicator_contrast,
        }.get(contrast, _gaus_contrast)
        self.centers = np.ones(shape=(self.quantiser.n_clusters, 2))*np.inf
        self.inertias = np.full(self.quantiser.n_clusters, np.nan)
        self.weighting_method = {
            "cloud"   : _cloud_weighting,
            "measure" : _measure_weighting,
            "iidproba": _iidproba_weighting,
        }.get(weighting_method, _cloud_weighting)

    def fit(self, X, y=None, sample_weight=None):
        """
        Calibration step: fit centers to the sample measures and derive inertias between centers.

        Parameters:
            X (list N x d numpy arrays): input measures in R^d from which to learn center locations and inertias
                (measures can have different N).
            y: Ignored, present for API consistency by convention.
            sample_weight (list of numpy arrays): weights for each measure point in X, optional.
                If None, the object's weighting_method will be used.

        Returns:
            self
        """
        if not hasattr(self.quantiser, 'fit'):
            raise TypeError("quantiser %s has no `fit` attribute." % (self.quantiser))
        if len(X) < self.quantiser.n_clusters:
            # in case there are not enough observations for fitting the quantiser, we add random points in [0, 1]^2
            random_points = np.random.rand(self.quantiser.n_clusters-len(X), X[0].shape[1])
            X.append(random_points)
        if sample_weight is None:
            sample_weight = np.concatenate([self.weighting_method(measure) for measure in X])

        measures_concat = np.concatenate(X)
        self.quantiser.fit(X=measures_concat, sample_weight=sample_weight)
        self.centers = self.quantiser.cluster_centers_
        labels = np.argmin(pairwise.pairwise_distances(measures_concat, Y=self.centers), axis=1)
        dist_centers = pairwise.pairwise_distances(self.centers)
        np.fill_diagonal(dist_centers, np.inf)
        self.inertias = np.min(dist_centers, axis=0)/2
        return self

    def __call__(self, measure, sample_weight=None):
        """
        Apply measure vectorisation on a single measure.

        Parameters:
            measure (n x d numpy array): input measure in R^d.

        Returns:
            numpy array in R^self.quantiser.n_clusters.
        """
        if sample_weight is None:
            sample_weight = self.weighting_method(measure)
        return np.sum(sample_weight * self.contrast(measure, self.centers, self.inertias.T).T, axis=1)

    def transform(self, X, sample_weight=None):
        """
        Apply measure vectorisation on a list of measures.

        Parameters:
            X (list N x d numpy arrays): input measures in R^d from which to learn center locations and inertias
                (measures can have different N).
            sample_weight (list of numpy arrays): weights for each measure point in X, optional.
                If None, the object's weighting_method will be used.

        Returns:
            numpy array with shape (number of measures) x (self.quantiser.n_clusters).
        """
        if sample_weight is None:
            sample_weight = [self.weighting_method(measure) for measure in X]
        return np.stack([self(measure, sample_weight=weight) for measure, weight in zip(X, sample_weight)])
