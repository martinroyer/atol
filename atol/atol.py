# coding: utf-8
"""
@author: Martin Royer
@copyright: INRIA 2019
"""

import numpy as np

from sklearn.metrics import pairwise
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import KMeans

import warnings


def centers_and_inertias(diags, n_centers):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=n_centers).fit(diags)
        if np.size(np.unique(kmeans.labels_)) < n_centers:
            kmeans = KMeans(n_clusters=np.size(np.unique(kmeans.labels_))).fit(diags)
    inertias = np.array([KMeans(n_clusters=1).fit(diags[kmeans.labels_ == lab, :]).inertia_
                         for lab in np.unique(kmeans.labels_)])
    return kmeans, inertias


def lapl_feats(diags, centers, inertias, eps=1e-10):
    return np.exp(-np.sqrt(pairwise.pairwise_distances(diags, Y=centers) / (inertias + eps)))


def gaus_feats(diags, centers, inertias, eps=1e-10):
    return np.exp(-pairwise.pairwise_distances(diags, Y=centers) / (inertias + eps))


class Atol(BaseEstimator, ClusterMixin, TransformerMixin):
    """Atol learning
        Read more in [https://arxiv.org/abs/1909.13472]
    """

    def __init__(self, n_centers=5, method=lapl_feats, aggreg=np.sum):
        self.n_centers = n_centers
        self.method = method
        self.aggreg = aggreg
        self.centers = np.ones(shape=(n_centers, 2))*np.inf
        self.inertias = np.ones(shape=(n_centers, 1))*np.inf

    def fit(self, diags):
        kmeans, self.inertias = centers_and_inertias(diags=diags, n_centers=self.n_centers)
        self.centers = kmeans.cluster_centers_
        return self

    def transform(self, diag):
        diag_atol = self.aggreg(self.method(diag, self.centers, self.inertias), axis=0)
        return diag_atol
