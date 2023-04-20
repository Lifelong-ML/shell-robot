# ShELL Robot Experiments

This is the main repo for the Penn + UT Austin Lifelong Self-Supervised Occupancy Anticipation ShELL Robot Experiments.

## Overview

The anticipation model takes in a single robot frame laser scan, and anticipates surrounding occluded structure. This is formulated as a self-supervised prediction problem: after driving around through a new room, the robot is able to assemble a ground truth map using an offline SLAM system, and localize each laser scan on this ground truth map. This provides laser scan input, ground truth region output pairs to train our anticipation network.


### Data preprocessing

As these processing steps require access to RosPy, we provide the Dockerfile `docker/Dockerfilepreprocess` as an environment, and `launch_preprocess.sh` as a convenience launcher.

Each group collects data from their robot containing a ground truth map, along with a list of laser scans and poses of those laser scans localized on the ground truth map. These formats are unique to each group, but they are homogenized using shared preprocessing infrastructure in `data_preprocessing/shared_structs`.

These shared data structures provide the `MapWrapper`, `LaserScan`, and `SE2` utilities needed to generate a list of input-output pairs, e.g.

```
input = laser_scan.ego_occupancy(map.resolution)
target = map.extract_region(pose)
```

These pairs are square `NxN` regions (i.e. `N = 160` as defined by `ROBOT_FRAME_SIZE` in `shared_structs`), rotated to be in robot frame. The `input` is `NxNx3` `np.array`, where all three channels at `0.33` represent unknown, the green channel represents occupied, and the blue channel represents free space. The `output` is an `NxN` `np.array` with `0` as unknown, `1` represents occupied, and `2` represents free space.

These pairs are saved as pickle file with the keys `input` for the input array and `target` for the target array. See `dataloaders/pickle_loader.py` for data type and shape details (these should be enforced by the generation utility code).

### Training

As these processing steps require access to a full torch install, we provide the Dockerfile `docker/Dockerfiletrain` as an environment, and `launch_train.sh` as a convenience launcher.

#### Data Loading

Due to the standardized format described above, Penn and UT Austin data can share a single data loader for training. This data loader, inside `dataloaders/pickle_loader.py`, indexes into a single dataset folder and loads all pickle files.

#### Train Loop

Currently, there's a simple train loop (`train.py`) that initializes the Anticipation UNet, takes the data folder from the command line, and trains the model, saving a checkpoint every 10 epochs to the specified checkpoint directory (default to `model_checkpoints/`).

## Future Work

Currently, the train loop has no awareness of the existence of multiple agents, or lifelong tasks. The training infra must be updated to sequester the data into different tasks, and train different agents on sequestered subsets. The concept of sequential tasks must be introduced, and the concept of information sharing between (weight averaging) must be implemented.