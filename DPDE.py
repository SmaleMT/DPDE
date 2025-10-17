import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, KernelPCA
from imblearn.over_sampling import SMOTE, KMeansSMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, \
    accuracy_score, recall_score, precision_score, mean_squared_error, silhouette_score, v_measure_score, pairwise_distances
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split, KFold, StratifiedShuffleSplit, cross_val_score
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn import decomposition
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, ClusterCentroids, \
    EditedNearestNeighbours, InstanceHardnessThreshold
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LassoCV
import PyIFS
from sklearn.ensemble import IsolationForest
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.svm import SVC
import warnings
import numpy as np
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")

class DPDE:
    def __init__(self, sub_num=50, kernel_pca_params=None, random_state=0, resampler_alpha=0.6, stay_ratio=0.8, beta=0.5):
        self.sub_num = sub_num
        self.kernel_pca_params = kernel_pca_params or {'n_components': 10, 'kernel': 'poly', 'fit_inverse_transform': True, 'remove_zero_eig': True}
        self.random_state = random_state
        self.clf = LogisticRegression(penalty='l2', C=0.5, solver='liblinear', max_iter=10000)
        # New initialization parameters
        self.selected_counts = None
        self.models = None
        self.majority_indices_ = None
        self.resampler_alpha = resampler_alpha
        self.model_alphas = []
        self.stay_ratio = stay_ratio
        self.beta = beta

    def resampler(self, X_train, y_train):
        counts = np.bincount(y_train)
        majority_class = np.argmax(counts)
        minority_class = 1 - majority_class
        majority_samples = X_train[y_train == majority_class]
        minority_samples = X_train[y_train == minority_class]

        # Get original distribution statistics
        original_mean = np.mean(majority_samples, axis=0)
        original_std = np.std(majority_samples, axis=0) + 1e-8
        # Get current batch sample weights (aligned with majority_indices_)
        majority_mask = (y_train == majority_class)

        # Metric 1: Feature preservation score (approximate KL divergence from original distribution)
        z_scores = (majority_samples - original_mean) / original_std
        feature_preserve = np.exp(-0.5 * np.mean(z_scores ** 2, axis=1))

        # Metric 2: Dynamically adjust the number of neighbors (avoid exceeding the number of minority samples)
        n_neighbors = 10
        if len(minority_samples) < n_neighbors:
            n_neighbors = len(minority_samples)
            if n_neighbors == 0:
                raise ValueError("Minority class samples are required for boundary calculation.")

        # Calculate the distance from each majority sample to the nearest n minority samples
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(minority_samples)
        distances, _ = nn.kneighbors(majority_samples)

        # Calculate average distance and normalize to get sensitivity
        avg_distances = distances.mean(axis=1)
        min_dist, max_dist = avg_distances.min(), avg_distances.max()
        boundary_sensitivity = 1 - (avg_distances - min_dist) / (max_dist - min_dist + 1e-8)

        boundary_sensitivity_min = np.min(boundary_sensitivity)
        boundary_sensitivity_max = np.max(boundary_sensitivity)
        if boundary_sensitivity_max > boundary_sensitivity_min:
            boundary_sensitivity_normalized = (boundary_sensitivity - boundary_sensitivity_min) / (
                    boundary_sensitivity_max - boundary_sensitivity_min)
        else:
            boundary_sensitivity_normalized = np.ones_like(boundary_sensitivity)  # If all values are the same, normalize to 1
        feature_preserve_min = np.min(feature_preserve)
        feature_preserve_max = np.max(feature_preserve)
        if feature_preserve_max > feature_preserve_min:
            feature_preserve_normalized = (feature_preserve - feature_preserve_min) / (
                    feature_preserve_max - feature_preserve_min)
        else:
            feature_preserve_normalized = np.ones_like(feature_preserve)  # If all values are the same, normalize to 1

        # Historical selection penalty term
        selection_penalty = 1 / (np.sqrt(self.selected_counts) + 1)

        # Dynamically adjust weights based on class ratio
        class_ratio = counts[minority_class] / counts[majority_class]
        w1 = 1 - self.resampler_alpha
        w2 = self.resampler_alpha

        # Combined score (including penalty term)
        combined_scores = (w1 * (1 - boundary_sensitivity) + w2 * feature_preserve) * selection_penalty

        # Adaptive probability calibration
        quantiles = np.quantile(combined_scores, [0.25, 0.75])
        iqr = quantiles[1] - quantiles[0]
        scaled_scores = (combined_scores - quantiles[0]) / (iqr + 1e-8)

        selection_probs = np.maximum(scaled_scores, 1e-8)
        selection_probs /= selection_probs.sum()

        n_minority = counts[minority_class]
        n_needed = n_minority
        if np.count_nonzero(selection_probs > 0) < n_needed:
            top_k_idx = np.argsort(combined_scores)[::-1][:n_needed]
            selection_probs[:] = 0
            selection_probs[top_k_idx] = 1.0
            selection_probs /= selection_probs.sum()

        # Importance sampling
        selected_idx = np.random.choice(len(majority_samples),
                                        size=n_minority,
                                        p=selection_probs,
                                        replace=False)
        selected_indices = self.majority_indices_[selected_idx]

        return (np.vstack([majority_samples[selected_idx], minority_samples]),
                np.concatenate([np.full(n_minority, majority_class),
                                np.full(len(minority_samples), minority_class)]),
                selected_indices)

    def _process_subset(self, X_train, y_train):
        """Process subset: sampling, training classifier"""
        X_res, y_res, selected_indices = self.resampler(X_train, y_train)
        # Clone classifier to ensure independence
        clfr = clone(self.clf)
        clfr.fit(X_res, y_res)
        return {
            "sub_data": X_res,
            "classifier": clfr,
        }, X_res, y_res, selected_indices

    def _build_subset_models(self, X_low, y_train):
        # Initialize selection counter
        counts = np.bincount(y_train)
        majority_class = np.argmax(counts)
        self.majority_indices_ = np.where(y_train == majority_class)[0]
        self.selected_counts = np.zeros(len(self.majority_indices_), dtype=int)

        # Model building process
        all_candidates = []  # Store model information for all candidate subsets
        all_X_subs = []  # Store X_res for all candidate subsets
        all_y_subs = []  # Store y_res for all candidate subsets
        indices_candidates = []  # Store majority sample indices for all candidate subsets
        n_candidates = self.sub_num
        i, error_num, error_line = 0, 0, 0.45
        while i < self.sub_num:
            i += 1
            # Generate sub-model
            model_info, X_sub, y_sub, selected_indices = self._process_subset(X_low, y_train)
            sample_weights = np.ones(len(y_sub)) / len(y_sub)
            # Predict and calculate errors
            y_pred = model_info['classifier'].predict(X_sub)
            errors = (y_pred != y_sub)
            error_rate = np.sum(errors * sample_weights) / np.sum(sample_weights)
            if error_rate >= error_line:  # Modify judgment condition
                i = i - 1
                error_num += 1
                # print(error_num)
                if error_num % 100 == 0:
                    error_line += 0.01
                continue  # Skip invalid model
            model_info['ACC'] = 1 - error_rate
            all_candidates.append(model_info)
            # Update selection counter
            mask = np.isin(self.majority_indices_, selected_indices)
            self.selected_counts[mask] += 1
            # Store candidate model
            all_X_subs.append(X_sub)
            all_y_subs.append(y_sub)
            indices_candidates.append(selected_indices)
        n_candidates = len(indices_candidates)
        # Calculate diversity weight matrix
        epsilon = 1e-8
        weight_diversity = np.zeros((n_candidates, n_candidates))
        for p in range(n_candidates):
            for q in range(n_candidates):
                intersection = len(set(indices_candidates[p]) & set(indices_candidates[q]))
                union = len(set(indices_candidates[p]) | set(indices_candidates[q]))
                weight_diversity[p, q] = 1 - intersection / union
        weight_diversity = (weight_diversity - np.min(weight_diversity)) / (
                    np.max(weight_diversity) - np.min(weight_diversity))

        majority_class = np.argmax(np.bincount(y_train))
        X_majority = X_low[y_train == majority_class]

        # Calculate overall mean and standard deviation (per feature dimension)
        mean_all = np.mean(X_majority, axis=0)
        std_all = np.std(X_majority, ddof=1, axis=0)

        # Merge into a statistical feature vector: [mean_1, mean_2, ..., mean_r, std_1, std_2, ..., std_r]
        L = np.concatenate([mean_all, std_all])
        r = len(mean_all)  # Number of feature dimensions

        # Initialize similarity score array
        similarity = np.zeros(n_candidates)

        for p in range(n_candidates):
            subset_data = X_low[indices_candidates[p]]
            mean_p = np.mean(subset_data, axis=0)
            std_p = np.std(subset_data, ddof=1, axis=0)

            # Merge into current subset's statistical feature vector
            L_p = np.concatenate([mean_p, std_p])

            # Calculate sum of absolute errors per dimension and average
            avg_diff = np.sum(np.abs(L_p - L)) / (2 * r)

            # Apply min truncation + convert to similarity score
            similarity[p] = 1 - min(1.0, avg_diff)

        # Construct similarity weight matrix W^S, using average score combination method
        weight_similarity = np.zeros((n_candidates, n_candidates))
        for p in range(n_candidates):
            for q in range(n_candidates):
                if p == q:
                    # 对角线元素设为1（子集与自身完全相似）
                    weight_similarity[p, q] = 1.0
                else:
                    # 使用调和平均公式
                    sim_p = similarity[p]
                    sim_q = similarity[q]

                    # 调和平均：2*sim_p*sim_q/(sim_p + sim_q)
                    denominator = sim_p + sim_q + epsilon
                    weight_similarity[p, q] = (2 * sim_p * sim_q) / denominator


        # Comprehensive weight matrix (dynamically adjust alpha)
        beta  = self.beta
        matrix = beta  * weight_diversity + (1 - beta ) * weight_similarity

        matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))

        # Calculate comprehensive score for each candidate subset (take row mean of matrix)
        scores = np.mean(matrix, axis=1)

        # Filter TOP30% subsets
        chose_num = int(self.sub_num * 0.3)
        top_indices = np.argsort(scores)[-chose_num:][::-1]  # Get top 30% by score, descending order

        # Retrieve corresponding models from all_candidates and add subset_score field
        subset_models = []
        for i in top_indices:
            model = all_candidates[i].copy()  # Recommend copy to avoid modifying original data
            model["subset_score"] = scores[i]  # Directly attach subset-level score
            subset_models.append(model)

        # Collect training prediction results for each model
        train_preds_list = []
        for model in subset_models:
            clf = model["classifier"]  # Can still directly use model["classifier"]
            pred_probs = clf.predict_proba(X_low)[:, 1]
            pred_labels = (pred_probs >= 0.5).astype(int)
            train_preds_list.append(pred_labels)

        return subset_models

    def _weighted_predict(self, models, X_test_low):
        """Weighted prediction based on sample distance"""
        self.models = models
        y_pred = []
        for x_low in X_test_low:
            weights = []
            probs = []
            for model in models:
                sub_data = model["sub_data"]
                distances = np.linalg.norm(sub_data - x_low, axis=1)  # Euclidean distance
                # Remove the farthest samples
                num_samples_to_keep = int(len(distances) * 0.8)  # Keep stay_ratio of samples
                sorted_indices = np.argsort(distances)  # Sort from smallest to largest distance
                kept_indices = sorted_indices[:num_samples_to_keep]  # Take first stay_ratio
                kept_distances = distances[kept_indices]  # Kept distances
                avg_distance = np.mean(kept_distances)
                # Normalization
                distance_weight = 1 / (1 + avg_distance) * model['ACC']
                # Model prediction probability
                prob = model["classifier"].predict_proba(x_low.reshape(1, -1))[0][1]
                weights.append(distance_weight)
                # weights.append(combined_weight)
                probs.append(prob)

                # Normalize weights
            weights = np.array(weights)
            if weights.sum() > 0:
                weights /= weights.sum()
            else:
                weights = np.ones_like(weights) / len(weights)

            final_prob = np.dot(weights, probs)
            y_pred.append(final_prob)

        return np.array(y_pred)  # Return positive class probability for each sample

    def fit(self, X_train, y_train):
        self.pca = KernelPCA(**self.kernel_pca_params)
        # print(self.kernel_pca_params)
        X_train_low = self.pca.fit_transform(X_train)
        counts = np.bincount(y_train)
        majority_class = np.argmax(counts)
        minority_class = 1 - majority_class
        majority_count = counts[majority_class]
        minority_count = counts[minority_class]

        # Calculate imbalance ratio
        IR = majority_count / minority_count
        # IR = 500
        if 200 > IR >= 20:
            # print("Switch to SVC training")
            self.clf = SVC(kernel='rbf', probability=True, max_iter=10000)
        if IR >= 200:
            # print("Switch to XGB training")
            self.clf = XGBClassifier(
                learning_rate=0.1,
                n_estimators=200,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
            )
        subset_models = self._build_subset_models(X_train_low, y_train)
        return {
            'pca': self.pca,
            'subset_models': subset_models,
        }

    def predict(self, model_dict, X_test):
        """Prediction method"""
        X_test_low = model_dict['pca'].transform(X_test)
        y_pred_probs = self._weighted_predict(model_dict['subset_models'], X_test_low)
        return np.where(y_pred_probs >= 0.5, 1, 0)

    def predict_proba(self, model_info, X_test):
        """Predict probability"""
        X_test_low = model_info['pca'].transform(X_test)
        return self._weighted_predict(model_info['subset_models'], X_test_low)
