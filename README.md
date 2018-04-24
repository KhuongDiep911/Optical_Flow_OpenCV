# CE888 Assignment 2
# Interpreting the Actions of Atari 2600 Agents

This work builds upon [ce888assignment1](https://github.com/JamesMadge/ce888assignment1) which captured data from agents playing eight Atari 2600 games. This data was the observations in the form of sequences of four frames from the Atari games and the resulting action that the agent has been train to output.

This work uses the data to conduct a series of experiments aimed at interpreting or providing explanations for the actions taken by the agent so that insights may be obtained about the what the agent has learnt about each of the game in order to be successful.

This file briefly introduces the conducted experiments and some of the results obtained.


<!-- This repository has been created for the purpose of the CE888 Assigment 2 deliverable and contain all code developed for the purpose of this project and a subset of experimental results.-->

## Experiments

A number of tecchniques were used to gain insights into the reason why the Atari agents took the actions it did given a certain input. This study tackled the problem from a number of perspectives.

This file briefly describes the results of the study and the impact of the experiments.

### LIME (Local Interpretable Model-Agnostic Explanations)

[LIME](https://github.com/marcotcr/lime)

![Asteroids](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/asteroids/frames/4394-6-23-8.png "TEXT")
![Battle Zone](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/battle_zone/frames/2177-0-2177-13.png "TEXT")

![Breakout](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/breakout/frames/1911-0-1911-2.png "TEXT")
![Gopher](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/gopher/frames/1131-0-1131-3.png "TEXT")

![James Bond](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/james_bond/frames/4851-2-494-17.png "TEXT")
![Ms. Pacman](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/ms_pacman/frames/4475-1-2048-2.png "TEXT")

![Road Runner](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/road_runner/frames/4877-3-886-7.png "TEXT")
![Tennis](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/explanation/results/tennis/frames/662-0-662-9.png "TEXT")

### Optical Flow

Optical flow was used to identify the movement of actors in the scene which is seen been the Atari agent during the sequence of input frames. Below are the results of running the spare optical flow and the dense optical flow.

#### Lucas Kanade, sparse optical flow.

The OpenCV V3.4.0 python implementation of the Lucas-Kanade  ([calcOpticalFlowPyrLK](https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk)) sparse optical flow algorithm.

![Asteroids](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/asteroids/frames/1030-1-547-4.png "TEXT")
![Battle Zone](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/battle_zone/frames/103-0-103-4.png "TEXT")
![Breakout](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/breakout/frames/78-0-78-2.png "TEXT")
![Gopher](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/gopher/frames/90-0-90-7.png "TEXT")
![James Bond](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/james_bond/frames/27-0-27-8.png "TEXT")
![Ms. Pacman](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/ms_pacman/frames/98-0-98-0.png "TEXT")
![Road Runner](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/road_runner/frames/105-0-105-9.png "TEXT")
![Tennis](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/lucas_kanade/tennis/frames/29-0-29-15.png "TEXT")

#### Gunnar Farneback, dense optical flow.

The OpenCV V3.4.0 python implementation of the Gunnar Farneback’s ([calcOpticalFlowFarneback](https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback)) dense optical flow algorithm.

![Asteroids](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/asteroids/frames/76-0-76-5.png "TEXT")
![Battle Zone](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/battle_zone/frames/97-0-97-4.png "TEXT")
![Breakout](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/breakout/frames/31-0-31-0.png "TEXT")
![Gopher](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/gopher/frames/98-0-98-4.png "TEXT")
![James Bond](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/james_bond/frames/7-0-7-7.png "TEXT")
![Ms. Pacman](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/ms_pacman/frames/93-0-93-0.png "TEXT")
![Road Runner](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/road_runner/frames/90-0-90-9.png "TEXT")
![Tennis](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/optical_flow/results/dense/tennis/frames/10-0-10-16.png "TEXT")

#### LIME Cross Dense Optical Flow

![Asteroids](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/13-0-13-8.png "TEXT")
![Battle Zone](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/263-0-263-10.png "TEXT")

![Breakout](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/1469-0-1469-3.png "TEXT")
![Gopher](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/1334-0-1334-6.png "TEXT")

![James Bond](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/1404-0-1404-8.png "TEXT")
![Ms. Pacman](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/1390-0-1390-5.png "TEXT")

![Road Runner](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/140-0-140-7.png "TEXT")
![Tennis](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/merged_lime_optical_flow/760-0-760-15.png "TEXT")

### Image Cluster Analysis

Image cluster analysis is performed to determine whether the observations could be clustered into groups that contain features which reveal the relavent action to be taken.

The analysis was performed using the [K-Means](https://projecteuclid.org/download/pdf_1/euclid.bsmsp/1200512992) and [Agglomerative Hierarchical](https://books.google.co.uk/books/about/Numerical_Taxonomy.html?id=iWWcQgAACAAJ&redir_esc=y) clustering algorithms. The results of both techniques are compared using [silhouette coefficients](https://ac.els-cdn.com/0377042787901257/1-s2.0-0377042787901257-main.pdf?_tid=45f93935-07e9-4d91-9c07-f887d75d4283&acdnat=1524558319_227f4e120f76072443bc235ab08a6d55) which reveal the coherance of instance and the same cluster and the separation between clusters which is indicative of good clusters being formed.

Image histograms were uses as the distance metric with a bin size of 256. 1000 images samples were taken for each game. The number of clusters were varied between 2 and 18 the number of iterations was set to 10 to overcome variability caused be random cluster initialization. The [`StandardScalar`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) feature transform was use over the [`RobustScalar`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html).

Below is an example of the graphical results obtained from clustering observations from the Astroids Atari game.

![K-Means_Asteroids](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/clustering/results/asteroids/graph_kmeans_silhouette_asteroids.png | width=48 "TEXT")
![Agglomerative_Asteroids](https://raw.githubusercontent.com/JamesMadge/ce888assignment2/master/clustering/results/asteroids/graph_agglomerative_silhouette_asteroids.png "TEXT")

The table below summarises the graphics results by showing the lowest silhouette score for each technique and the number of clusters for which that score was obtained.

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


<!--

8 [data sets](https://github.com/JamesMadge/ce888assignment1/tree/master/data) have been captured, each comprised of 5000 instances formed from 4 concatenated sequential frames of [agents](http://models.tensorpack.com/OpenAIGym/) playing one of eight Atari games, namely; Asteroids, Battle Zone, *Breakout*, *Gopher*, *James Bond*, *Ms. Pacman*, *Road Runner* and *Tennis*. The first 50 instances of each data set have been uploaded to GitHib for the purposes of demonstration. The entire data is 0.5GiB and hence is stored and maintained locally.

Each data instance has the following descriptive name: 

<**observation**>-<**episode**>-<**tick**>-<**action**>.png

Where, **observation** is the number of the observation from 0->4999, **episode** is the game number incremented from zero if a new game is started while observations are being captured, **tick** observation number for the current episode, **action** the resulting action taken by the agent.

![Asteroids](https://raw.githubusercontent.com/JamesMadge/ce888assignment1/master/data/asteroids/49-0-49-2.png "TEXT")
![Battle Zone](https://raw.githubusercontent.com/JamesMadge/ce888assignment1/master/data/battle_zone/49-0-49-9.png "TEXT")

![Breakout](https://raw.githubusercontent.com/JamesMadge/ce888assignment1/master/data/breakout/49-0-49-1.png "TEXT")
![Gopher](https://raw.githubusercontent.com/JamesMadge/ce888assignment1/master/data/gopher/49-0-49-4.png "TEXT")

![James Bond](https://raw.githubusercontent.com/JamesMadge/ce888assignment1/master/data/james_bond/49-0-49-11.png "TEXT")
![Ms. Pacman](https://raw.githubusercontent.com/JamesMadge/ce888assignment1/master/data/ms_pacman/49-0-49-6.png "TEXT")

![Road Runner](https://raw.githubusercontent.com/JamesMadge/ce888assignment1/master/data/road_runner/49-0-49-17.png "TEXT")
![Tennis](https://raw.githubusercontent.com/JamesMadge/ce888assignment1/master/data/tennis/49-0-49-8.png "TEXT")

## Code

Minimal code was required to be written for Assignment 1, the provided sample code shown below was modified and incorporated into the `play_one_episode` function of TensorPack's [common.py](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/DeepQNetwork/common.py) file within the DeepQNetwork example to capture data instances named in the format specified above. The resulting implementation can be found [here](https://github.com/JamesMadge/ce888assignment1/blob/master/common.py).

```python
from PIL import Image

stacker = np.empty((84, 0, 3),dtype="uint8")

for it in range(4):
    im = Image.fromarray(s[:, :, it*3:3*(it+1)])
    q = np.asarray(im)
    stacker = np.hstack((stacker, q))

im = Image.fromarray(stacker)
im.save("game_name-" + str(t) + ".png") # you need to define (t) somewhere so that you know which part of the game you are in. 


```

-->
