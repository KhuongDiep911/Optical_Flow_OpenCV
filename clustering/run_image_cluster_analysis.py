#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" J.Madge 23.04.2018 'run_image_cluster_analysis.py'.

    Performs cluster analysis on raw image data so as to determine whether a
    correlation exists between the number of clusters selected and the quality
    of the clusters produced.

    The following parameters were used for this experiment. Future work would
    look to determine the influence of each and tune then in order to obtain
    the best performance.

    Distance metric:    Image histograms are used as the distance
                        metric with a bin size of 256.
    Sample size:        1000 images.
    Min/max clusters:   2/18 respectively.
    Cluster iterations: 10
    Feature transform:  Standard scalar.

    """

# Distance measures for images need to be created so that they may be clustered.
# This involves extracting data about the images, there are a number of ways this could be achieved.
# This articles pointed to a number of established techniques.
# https://izbicki.me/blog/data-mining-images-tutorial.html

# Histogram analysis.
# https://en.wikipedia.org/wiki/Image_histogram

# Converting images into time-series data.
# https://izbicki.me/blog/converting-images-into-time-series-for-data-mining

# Shock graphs (don't really seem applicable).
# http://www.cs.toronto.edu/~sven/Papers/ijcv99.pdf

# SIFT (Scale-invarient Feature Transform).
# https://en.wikipedia.org/wiki/Scale-invariant_feature_transform

# Hough Transform.
# https://en.wikipedia.org/wiki/Hough_transform

# Bunch of other `standard' distance metrics.
# Euclidean, Manhattan, Mahalanobis, Canberra. etc.
# https://onlinelibrary.wiley.com/doi/pdf/10.1002/ima.22031

# Template matching.
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#py-template-matching

# Background subtraction to remove static elements in the game.
# https://docs.opencv.org/3.3.0/db/d5c/tutorial_py_bg_subtraction.html

from jm17290.ops import *
from jm17290.atari.game import Games
from jm17290.visualisation import ProgressBar

import os
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import AgglomerativeClustering

# Select game.
game = Games.james_bond

# Load data for the selected game.
data_files = os.listdir(game.data)

# Random color for output graphs.
# https://stackoverflow.com/questions/13998901/generating-a-random-hex-color-in-python
import random

r = lambda: random.randint(0, 255)
colour_random = '#%02X%02X%02X' % (r(), r(), r())

# Matrix in which to store the calculated histograms.
data_hist = np.empty((0, 256))

# Size of histogram bins.
bin_size = 256

# Sample size.
sample_size = 1000

# Set minimum and maximum number of clusters.
clusters_min = 2
clusters_max = 18  # Max number of actions.

# Clustering iterations. Removes element of chance caused by random cluster initialization.
cluster_iterations = 10

# Track progress using progress bar.
progressBar = ProgressBar((clusters_max - clusters_min) + 1, 20)

# Maintain the silhouette scores for each algorithm, for each number of clusters.
silhouette_scores_kmeans = {}
silhouette_scores_agglomerative = {}

# Maintain the lowest silhouette scores for each algorithm.
silhouette_score_lowest_kmeans = 1
silhouette_score_lowest_agglomerative = 1

# Maintain the cluster with the lowest silhouette score for each algorithm.
silhouette_score_lowest_cluster_kmeans = 0
silhouette_score_lowest_cluster_agglomerative = 0

for file in data_files:
    # Load images in grey scale.
    img = image_load_grey_scale('{0}/{1}'.format(game.data, file))

    # Calculate the histogram for each image.
    # https://docs.opencv.org/3.1.0/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d
    # images        Source arrays.
    # channels      List of the dims channel used to compute the histogram.
    # mask          Optional mask.
    # histSize      Array of histogram sizes in each dimension.
    # ranges        Array of the dims arrays of the histogram bin boundaries in each dimension.
    # hist          Output histogram. Default=None.
    # accumulate    Accumulation flag. If set, the histogram is not cleared in the beginning when it is allocated.
    #               Default = None
    hist = cv2.calcHist([img], channels=[0], mask=None, histSize=[bin_size], ranges=[0, 256])

    # Append the calculated histogram into a storage array.
    data_hist = np.append(data_hist, np.transpose(hist), axis=0)

# Create a pandas data frame from the histogram data, where the index is the name of the files.
df_data_hist = pd.DataFrame(data_hist, index=data_files)

# Take a samples of the data.
df_img_hist_sample = df_data_hist.values[np.random.choice(df_data_hist.values.shape[0], sample_size)]

# Scale the features uniformly and centre about zero.
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
df_img_hist_sample_transformed = StandardScaler().fit_transform(df_img_hist_sample)

# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
# TODO Should Robust scalar be preferred? Since most games are likely to hava a significant amount of a certain colour.
# df_img_hist_sample_transformed = RobustScaler().fit_transform(df_img_hist_sample)

# Cycle through different number of clusters to determine which provides the best clustering using the silhouette
# coefficient which is a 'measure of how similar an object is to its own cluster (cohesion) compared to other clusters
# (separation).

for i in range(clusters_min, clusters_max + 1):
    # Silhouette scores for K-Means and agglomerative clustering for 'i' clusters.
    results_kmeans = []
    results_agglomerative_clustering = []

    # Perform clustering multiple times to overcome randomness caused by random cluster initialization.
    for j in range(cluster_iterations):
        # K-Means clustering.
        labels_kmeans = KMeans(i).fit_predict(df_img_hist_sample_transformed)
        # Calculate silhouette score.
        results_kmeans.append(metrics.silhouette_score(df_img_hist_sample_transformed, labels_kmeans))

        # Agglomerative clustering.
        labels_agglomerative_clustering = AgglomerativeClustering(i).fit_predict(df_img_hist_sample_transformed)
        # Calculate silhouette score.
        results_agglomerative_clustering.append(
            metrics.silhouette_score(df_img_hist_sample_transformed, labels_agglomerative_clustering))

    # Maintain silhouette scores for each iteration of each number of clusters.
    silhouette_scores_kmeans[i] = results_kmeans
    silhouette_scores_agglomerative[i] = results_agglomerative_clustering

    # Calculate the mean silhouette score for each algorithm.
    silhouette_score_mean_kmeans = np.mean(results_kmeans)
    silhouette_score_mean_agglomerative = np.mean(results_agglomerative_clustering)

    # Update lowest silhouette score and cluster, k-means.
    if silhouette_score_mean_kmeans < silhouette_score_lowest_kmeans:
        silhouette_score_lowest_kmeans = silhouette_score_mean_kmeans
        silhouette_score_lowest_cluster_kmeans = i

    # Update lowest silhouette score and cluster, k-means.
    if silhouette_score_mean_agglomerative < silhouette_score_lowest_agglomerative:
        silhouette_score_lowest_agglomerative = silhouette_score_mean_agglomerative
        silhouette_score_lowest_cluster_agglomerative = i

    # Update progress bar.
    progressBar.update(i)

# Print results.
print("\n\nLowest Silhouette Coefficient KMeans: " + str(silhouette_score_lowest_kmeans))
print("Lowest Silhouette Score Cluster KMeans: " + str(silhouette_score_lowest_cluster_kmeans))

print("\nLowest Silhouette Coefficient Agglomerative: " + str(silhouette_score_lowest_agglomerative))
print("Lowest Silhouette Score Cluster Agglomerative: " + str(silhouette_score_lowest_cluster_agglomerative))

# Plot silhouette score for k-means.
data_kmeans = np.empty((0, 2))
for i in range(clusters_min, clusters_max + 1):
    for j in silhouette_scores_kmeans[i]:
        data_kmeans = np.append(data_kmeans, np.array([i, j]).reshape(1, 2), axis=0)

df = pd.DataFrame(data_kmeans, columns=['cluster_num', 'silhouette_score'])
sns.pointplot(data=df, x='cluster_num', y='silhouette_score', color=colour_random)

plt.title('Mean Silhouette Scores with Confidence Intervals\nfor Different Numbers of K-Means Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.show()

# Plot silhouette scores for agglomerative.
data_agglomerative = np.empty((0, 2))
for i in range(clusters_min, clusters_max + 1):
    for j in silhouette_scores_agglomerative[i]:
        data_agglomerative = np.append(data_agglomerative, np.array([i, j]).reshape(1, 2), axis=0)

df = pd.DataFrame(data_agglomerative, columns=['cluster_num', 'silhouette_score'])
sns.pointplot(data=df, x='cluster_num', y='silhouette_score', color=colour_random)

plt.title('Mean Silhouette Scores with Confidence Intervals\nfor Different Numbers of Agglomerative Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.show()
