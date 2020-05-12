<img src="https://github.com/bayesgroup/bayesgroup.github.io/blob/master/tqc/assets/tqc/main_exps.svg">


This repository implements continuous reinforcement learning method TQC, described in paper ["Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics"](https://arxiv.org/abs/2005.04269).
Source code is based on [Softlearning](https://github.com/rail-berkeley/softlearning.git), and we thank the authors for a good framework. For more exaustive Readme (for example, docker usage), please refer to the original repo. 

Our method is implemented in module `${SOURCE_PATH}/softlearning/algorithms/tqc.py`.

## MuJoCo Installation

1. [Download](https://www.roboti.us/index.html) and install MuJoCo 1.50 from the MuJoCo website. We assume that the MuJoCo files are extracted to the default location (`~/.mujoco/mjpro150`). Gym and MuJoCo 2.0 have integration bug, where Gym doesn't process contanct forces correctly for environments Humanoid and Ant.
Please use MuJoCo 1.5.

2. Copy your MuJoCo license key (mjkey.txt) to ~/.mujoco/mjkey.txt:

## Conda installation

Create and activate conda environment, install softlearning to enable command line interface.
```
cd ${SOURCE_PATH}
conda env create -f environment.yml
conda activate tqc
```

## Training and simulating an agent
 1. To train the agent
    ```
    ./run_tqc.sh --alg_top_crop_quantiles=2 --domain=Walker2d
    ```
    Number of atoms to remove for each environment:
    
    | Environment        | alg_top_crop_quantiles  |
    | ------------- |:-------------:|
    | Hopper           | 5 |
    | HalfCheetah      | 0 |
    | Walker2d         | 2 |
    | Ant              | 2 |
    | Humanoid         | 2 |

    You can look at full list of parameters inside the `run_tqc.sh`.

2. To simulate the resulting policy:

First, find the path that the checkpoint is saved to. By default, the data is saved under `${SOURCE_PATH}/ray_results/<universe>/<domain>/<task>/<datatimestamp>-<exp-name>/<trial-id>/<checkpoint-id>`. 

For example: `${SOURCE_PATH}/ray_results/gym/HalfCheetah/v3/2018-12-12T16-48-37-my-experiment-1-0/mujoco-runner_0_seed=7585_2018-12-12_16-48-37xuadh9vd/checkpoint_1000/`. 

The next command assumes environment var `${CHECKPOINT_DIR}` contains `${SOURCE_PATH}/ray_results/...`.

```
python ./examples/development/simulate_policy.py \
    ${CHECKPOINT_DIR} \
    --max-path-length=1000 \
    --num-rollouts=1 \
    --render-mode=human
```

## Run curves
`tqc_curves.pkl` contains evaluation returns of TQC agent, which were used for plotting learning curves in the paper. 
