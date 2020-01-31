# coding: utf-8
"""
@author: Martin Royer
@copyright: INRIA 2019-2020
"""

import numpy as np

from sklearn.metrics import pairwise
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans, MiniBatchKMeans


def lapl_feats(diags, centers, inertias, eps=1e-10):
    return np.exp(-np.sqrt(pairwise.pairwise_distances(diags, Y=centers) / (inertias + eps)))


def gaus_feats(diags, centers, inertias, eps=1e-10):
    return np.exp(-pairwise.pairwise_distances(diags, Y=centers) / (inertias + eps))


def weighting_switch(viewpoint):
    return {
        "None": lambda _: None,
        "cloud": lambda diags: np.concatenate([ np.ones(shape=diag.shape[0]) for diag in diags ]),
        "measure": lambda diags: np.concatenate([ np.ones(shape=diag.shape[0]) / diag.shape[0] for diag in diags ]),
    }.get(viewpoint, "None")


class Atol(BaseEstimator, ClusterMixin, TransformerMixin):
    """Atol learning
        Read more in [https://arxiv.org/abs/1909.13472]
    """

    def __init__(self, n_centers=5, method=lapl_feats, aggreg=np.sum, viewpoint="None", batch_size=0, n_calib=-1):
        self.n_calib = n_calib
        self.n_centers = n_centers
        self.r_centers = n_centers
        self.method = method
        self.batch_size = batch_size
        self.aggreg = aggreg
        self.centers = np.ones(shape=(n_centers, 2))*np.inf
        self.inertias = np.ones(shape=(n_centers, 1))*np.inf
        self.weighting_method = weighting_switch(viewpoint)

    def fit(self, diags):
        """
        Calibration step: learn centers and inertias from current available diagrams.
        @todo: si c'est un array Nx2,

        :param diags: list of diagrams from which to learn center locations and cluster spread
        :return: None
        """
        if self.n_calib > 0:
            diags = diags[:self.n_calib]
        elif self.n_calib == 0:
            diags = np.random.rand(self.n_centers, 2)
        weights = self.weighting_method(diags)
        diags_concat = np.concatenate(diags)
        kmeans = KMeans() if not self.batch_size else MiniBatchKMeans(batch_size=self.batch_size)
        kmeans.n_clusters = self.r_centers
        kmeans.fit(diags_concat, sample_weight=weights)
        labels = np.argmin(pairwise.pairwise_distances(diags_concat, Y=kmeans.cluster_centers_), axis=1)
        self.centers  = np.array([kmeans.cluster_centers_[lab,:] for lab in np.unique(labels)])
        dist_centers = pairwise.pairwise_distances(self.centers)
        np.fill_diagonal(dist_centers, np.inf)
        self.inertias = np.min(dist_centers, axis=0)/2
        self.r_centers = np.size(np.unique(labels))
        # self.inertias = np.array([[KMeans(n_clusters=1).fit(diags_concat[labels == lab, :],
        #                     sample_weight=weights[labels == lab]).inertia_] for lab in np.unique(labels)])
        return self

    def transform(self, diag):
        """
        Vectorisation step: vectorise a diagram given the current centers and inertias.
        Without proper calibration, default value should produce NaNs.

        :param diag: entry diagram to vectorise
        :return: array of size `n_centers`
        """
        weights = self.weighting_method(diag)[0]
        diag_atol = self.aggreg(self.method(diag, self.centers, self.inertias.T)*weights, axis=0)
        return diag_atol
