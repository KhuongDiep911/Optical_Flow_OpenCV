# CE888 Assignment 2
# Interpreting the Actions of Atari 2600 Agents

This work builds on [CE888 Assignment 1](https://github.com/JamesMadge/ce888assignment1) in which data instances were captured that couple four sequential frames of Atari 2600 gameplay with the response of an autonomous agent trained to play a specific game. The pre-trained agent was sourced from the [Tensorpack](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/A3C-Gym) framework which trained 47 different Atari 2600 agents using a multi-GPU version of the [A3C](https://arxiv.org/pdf/1602.01783.pdf) algorithm.

The objective of this work is to interpret the captured data instances and to provide insights as to why the agents took the actions they did and therefore discover what the agent learnt about the game in order to play successfully. This work conducts serval experiments involving [LIME](https://github.com/marcotcr/lime) (Local Interpretable Model-Agnostic Explanations), OpenCV V3.4.0 Lucas-Kanade and Gunnar Farneback [optical flow](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_lucas_kanade.html) implementations and the scikit learn implementations of [K-Means](http://scikit-learn.org/stable/modules/clustering.html#k-means) and [Agglomerative clustering](http://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering) techniques.

This file briefly explains the experiments conducted and provides a sample of the obtained results.

## Experiments

Each experiment approaches the problem from a different perspective. This section introduces the motives behind these experiments and presents a sample of results.

### Frame Perturbation

Previously captured observations are passed to the [LIME](https://github.com/marcotcr/lime) framework which perturbs the sequence of frames to determine which regions contributed most significantly towards the selection of the associated action. A data instance for each game is shown below with highlights representing the regions determined by LIME to contribute most significantly towards the selected action.

Astroids (UP_FIRE) | Battle Zone (DOWN_FIRE)
:-------:|:----------:
![LIME_Asteroids](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/asteroids/frames/4394-6-23-8.png "LIME, Asteroids") | ![LIME_Battle_Zone](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/battle_zone/frames/2177-0-2177-13.png "LIME, Battle Zone")

Breakout (RIGHT) | Gopher (RIGHT)
:-------:|:----------:
![LIME_Breakout](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/breakout/frames/1911-0-1911-2.png "LIME, Breakout")  |  ![LIME_Gopher](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/gopher/frames/1131-0-1131-3.png "LIME, Gopher")

James Bond (DOWN_LEFT_FIRE) | Ms. Pacman (RIGHT)
:---------:|:----------:
![LIME_James_Bond](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/james_bond/frames/4851-2-494-17.png "LIME, James Bond")  |  ![LIME_Ms_Pacman](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/ms_pacman/frames/4475-1-2048-2.png "LIME, Ms. Pacman")

Road Runner (UP_LEFT) | Tennis (DOWN_LEFT)
:----------:|:----------:
![LIME_Road_Runner](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/road_runner/frames/4877-3-886-7.png "LIME, Road Runner")  |  ![LIME_Tennis](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/tennis/frames/662-0-662-9.png "LIME, Tennis")

### Movement

Optical flow techniques were used to reveal and extract the movement of artefacts in the gameplay frames which the agent may use to select the associated action.

#### Lucas-Kanade, sparse optical flow.

The OpenCV V3.4.0 python implementation of the Lucas-Kanade ([`calcOpticalFlowPyrLK`](https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk)) sparse optical flow algorithm. Optical flow vectors for each frame transition are calculated and visualised on top of the final frame in the sequence.

![Lucas_Kanade_Asteroids](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/asteroids/frames/1030-1-547-4.png "Lucas Kanade, Asteroids")
![Lucas_Kanade_Battle Zone](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/battle_zone/frames/103-0-103-4.png "Lucas Kanade, Battle Zone")
![Lucas_Kanade_Breakout](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/breakout/frames/78-0-78-2.png "Lucas Kanade, Breakout")
![Lucas_Kanade_Gopher](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/gopher/frames/90-0-90-7.png "Lucas Kanade, Gopher")
![Lucas_Kanade_James Bond](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/james_bond/frames/27-0-27-8.png "Lucas Kanade, James Bond")
![Lucas_Kanade_Ms. Pacman](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/ms_pacman/frames/98-0-98-0.png "Lucas Kanade, Ms. Pacman")
![Lucas_Kanade_Road Runner](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/road_runner/frames/105-0-105-9.png "Lucas Kanade, Road Runner")
![Lucas_Kanade_Tennis](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/tennis/frames/29-0-29-15.png "Lucas Kanade, Tennis")

#### Gunnar Farneback, dense optical flow.

The OpenCV V3.4.0 python implementation of the Gunnar Farneback’s ([`calcOpticalFlowFarneback`](https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback)) dense optical flow algorithm. The dense optical flow image is calculated for each frame transition, merged and visualised on top of the final frame in the sequence.

[![Asteroids](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/asteroids/frames/76-0-76-5.png "TEXT")](https://youtu.be/ptdM1Kqk_Lg)
![Battle Zone](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/battle_zone/frames/97-0-97-4.png "TEXT")
![Breakout](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/breakout/frames/31-0-31-0.png "TEXT")
![Gopher](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/gopher/frames/98-0-98-4.png "TEXT")
![James Bond](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/james_bond/frames/7-0-7-7.png "TEXT")
![Ms. Pacman](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/ms_pacman/frames/93-0-93-0.png "TEXT")
![Road Runner](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/road_runner/frames/90-0-90-9.png "TEXT")
![Tennis](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/tennis/frames/10-0-10-16.png "TEXT")

#### Combining LIME Output and Dense Optical Flow

The output of LIME and results of the dense optical flow calculations were combined to create new insights.

Asteroids (UP_FIRE) | Battle Zone (UP_FIRE)
:-------:|:----------:
![LIME_DOF_Merge_Asteroids](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/13-0-13-8.png "LIME & Dense Optical Flow Merge, Asteroids")  |  ![LIME_DOF_Merge_Battle_Zone](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/263-0-263-10.png "LIME & Dense Optical Flow Merge, Battle Zone")

Breakout (LEFT) | Gopher (RIGHT_FIRE)
:-------:|:----------:
![LIME_DOF_Merge_Breakout](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/1469-0-1469-3.png "LIME & Dense Optical Flow Merge, Breakout")  |  ![LIME_DOF_Merge_Gopher](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/1334-0-1334-6.png "LIME & Dense Optical Flow Merge, Gopher")

James Bond (DOWN_RIGHT) | Ms. Pacman (UP_RIGHT)
:---------:|:----------:
![LIME_DOF_Merge_James_Bond](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/1404-0-1404-8.png "LIME & Dense Optical Flow Merge, James Bond")  |  ![LIME_DOF_Merge_Ms_Pacman](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/1390-0-1390-5.png "LIME & Dense Optical Flow Merge, Ms. Pacman")

Road Runner (UP_LEFT) | Tennis (UP_LEFT_FIRE)
:----------:|:----------:
![LIME_DOF_Merge_Road_Runner](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/140-0-140-7.png "LIME & Dense Optical Flow Merge, Road Runner")  |  ![LIME_DOF_Merge_Tennis](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/760-0-760-15.png "LIME & Dense Optical Flow Merge, Tennis")

### Image Cluster Analysis

Cluster analysis of the previously captured observations was performed to determine whether specific actions were taken for frames that contain similar artefacts. The analysis was performed using the [K-Means](https://projecteuclid.org/download/pdf_1/euclid.bsmsp/1200512992) and [Agglomerative Hierarchical](https://books.google.co.uk/books/about/Numerical_Taxonomy.html?id=iWWcQgAACAAJ&redir_esc=y) clustering algorithms. The results of both techniques are compared using [silhouette coefficients](https://ac.els-cdn.com/0377042787901257/1-s2.0-0377042787901257-main.pdf?_tid=45f93935-07e9-4d91-9c07-f887d75d4283&acdnat=1524558319_227f4e120f76072443bc235ab08a6d55) which score the quality of the resulting clusters based on the coherence of instances in the same cluster and the separation of different clusters.

Below is an example of the graphical results obtained from clustering observations from the Asteroids Atari game.

Asteroids, K-Means Clustering |  Asteroids, Agglomerative Clustering
:---------------------------:|:------------------------------------:
![K-Means_Asteroids](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/clustering/results/asteroids/graph_kmeans_silhouette_asteroids.png "Astroids, K-Means Clustering")  |  ![Agglomerative_Asteroids](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/clustering/results/asteroids/graph_agglomerative_silhouette_asteroids.png "Asteroids, Agglomerative Clustering")

The table below summarises the graphical results across the eight games used in this work; it presents the number of actions available to the agent in each game, the lowest silhouette score obtained by each technique and the number of clusters used to obtain the lowest silhouette score.

| Game          | Actions | SC K-Means (3SF) | # Clusters K-Means | SC Agglomerative (3SF) | # Clusters Agglomerative |
| ------------- |:-------:|:----------------:|:------------------:|:----------------------:|:------------------------:|
| Asteroids     | 14      | 0.0421           | 18                 | 0.0558                 | 6                        |
| Battle Zone   | 18      | 0.0473           | 16                 | 0.0141                 | 18                       |
| Breakout      | 4       | 0.109            | 16                 | 0.0900                 | 4                        |
| Gopher        | 8       | 0.0555           | 9                  | 0.0406                 | 10                       |
| James Bond    | 18      | 0.0404           | 16                 | 0.0352                 | 6                        |
| Ms. Pacman    | 9       | 0.108            | 4                  | 0.0903                 | 3                        |
| Road Runner   | 18      | 0.0115           | 8                  | 0.00126                | 13                       |
| Tennis        | 18      | 0.0799           | 8                  | 0.0685                 | 8                        |
