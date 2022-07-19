# coding: utf-8
"""
@author: Martin Royer
@copyright: INRIA 2019
"""

import os

from itertools import product
import shutil
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import gudhi as gd
from scipy.sparse import csgraph
from scipy.linalg import eigh
from scipy.io import loadmat

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans

from .atol import Atol

graph_dtypes = ["dgmOrd0", "dgmExt0", "dgmRel1", "dgmExt1"]


def build_filtered_simplex(A, filtration_val, edge_threshold=0):
    num_vertices = A.shape[0]
    st = gd.SimplexTree()
    [st.insert([i], filtration=-1e10) for i in range(num_vertices)]
    for i, j in combinations(range(num_vertices), r=2):
        if A[i, j] > edge_threshold:
            st.insert([i, j], filtration=-1e10)
    for i in range(num_vertices):
        st.assign_filtration([i], filtration_val[i])
    return st


def compute_tda_for_graphs(graph_folder, filtrations):
    diag_repo = graph_folder + "diagrams/"
    if os.path.exists(diag_repo) and os.path.isdir(diag_repo):
        shutil.rmtree(diag_repo)
    [os.makedirs(diag_repo + dtype) for dtype in [""] + graph_dtypes]

    pad_size = 1
    for graph_name in os.listdir(graph_folder + "mat/"):
        A = np.array(loadmat(graph_folder + "mat/" + graph_name)["A"], dtype=np.float32)
        pad_size = np.max((A.shape[0], pad_size))
    print("Pad size for eigenvalues in this dataset is: %i" % pad_size)

    for graph_name in os.listdir(graph_folder + "mat/"):
        A = np.array(loadmat(graph_folder + "mat/" + graph_name)["A"], dtype=np.float32)
        name = graph_name.split("_")
        gid = int(name[name.index("gid") + 1]) - 1
        egvals, egvectors = eigh(csgraph.laplacian(A, normed=True))
        for filtration in filtrations:
            time = float(filtration.split("-")[0])
            filtration_val = np.square(egvectors).dot(np.diag(np.exp(-time * egvals))).sum(axis=1)
            st = build_filtered_simplex(A, filtration_val)
            st.extend_filtration()
            dgms = st.extended_persistence(min_persistence=1e-5)
            [np.savetxt(diag_repo + "%s/graph_%06i_filt_%s.csv" % (dtype, gid, filtration), [pers[1] for pers in diag],
                        delimiter=',')
             for diag, dtype in zip(dgms, ["dgmOrd0", "dgmExt0", "dgmRel1", "dgmExt1"])]
    return


def csv_toarray(file_name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diag_csv = np.loadtxt(file_name, delimiter=',', ndmin=2)
        if not diag_csv.any():
            diag_csv = np.array([[0, 0]])
    return diag_csv


def atol_feats_graphs(graph_folder, all_diags, atol_objs):
    feats = []
    for graph_name in os.listdir(graph_folder + "mat/"):
        name = graph_name.split("_")
        label = int(name[name.index("lb")+1])
        gid = int(name[name.index("gid") + 1]) - 1
        if not np.sum([np.size(_.inertias) for _ in atol_objs.values()]):
            continue
        for (dtype, filt), atol_obj in atol_objs.items():
            diag_feats = atol_obj(all_diags[(dtype, filt, gid)])
            [feats.append({"index": gid, "type": dtype+"-"+filt, "center": idx_center, "value": _, "label": label})
             for idx_center, _ in enumerate(diag_feats)]
    return feats


def _predict(learner, test_feats, score=accuracy_score):
    x_test, y_test = test_feats.apply(lambda _: _["value"].values), LabelEncoder().fit_transform(
        test_feats.apply(lambda _: _["label"].values[0]))
    y_test_pred = learner.predict(list(x_test))
    test_score = score(y_test, y_test_pred)
    print("  (Test) %s: %.2f" % (score.__name__, test_score))
    return test_score


def _fit(learner, train_feats, score=accuracy_score):
    x_train, y_train = train_feats.apply(lambda _: _["value"].values), LabelEncoder().fit_transform(
        train_feats.apply(lambda _: _["label"].values[0]))
    learner.fit(list(x_train), y_train)
    y_train_pred = learner.predict(list(x_train))
    print(" Descriptors have size:", np.unique(list(map(len, x_train)), return_counts=True))
    print("  (train) %s: %.2f" % (score.__name__, score(y_train, y_train_pred)))
    return learner


def graph_tenfold(graph_folder, filtrations, n_centers=10):
    sampling = "index"

    num_elements = len(os.listdir(graph_folder + "mat/"))
    array_indices = np.arange(num_elements)
    all_diags = {}
    for dtype, gid, filt in product(graph_dtypes, array_indices, filtrations):
        all_diags[(dtype, filt, gid)] = csv_toarray(
            graph_folder + "diagrams/%s/graph_%06i_filt_%s.csv" % (dtype, gid, filt))
    atol_objs = {}
    for dtype, filt in product(graph_dtypes, filtrations):
        atol_objs[(dtype, filt)] = Atol(quantiser=MiniBatchKMeans(n_clusters=n_centers, batch_size=2048))
    length = num_elements // 10

    np.random.shuffle(array_indices)
    test_scores, featurisation_times = [], []
    for k in range(10):
        print("-- Fold %i" % (k + 1))
        test_indices = array_indices[np.arange(start=k * length, stop=(k + 1) * length)]
        train_indices = np.setdiff1d(array_indices, test_indices)

        time1 = time.time()
        for dtype, filt in product(graph_dtypes, filtrations):
            atol_objs[(dtype, filt)].fit([all_diags[(dtype, filt, gid)] for gid in train_indices])
        feats = pd.DataFrame(atol_feats_graphs(graph_folder, all_diags, atol_objs),
                             columns=["index", "type", "center", "value", "label"])
        time2 = time.time()

        fitted_learner = _fit(learner=RandomForestClassifier(n_estimators=100),
                              train_feats=feats[np.isin(feats[sampling], train_indices)].groupby([sampling]))
        test_score = _predict(learner=fitted_learner,
                              test_feats=feats[np.isin(feats[sampling], test_indices)].groupby([sampling]))
        test_scores.append(test_score)
        featurisation_times.append(time2 - time1)
    return test_scores, featurisation_times
