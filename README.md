# Preference-Guided-DQN-Atari
### :page_with_curl: Sampling Efficient Deep Reinforcement Learning through Preference-Guided Stochastic Exploration
:dizzy: This work proposes a **_generalized and efficient epsilon-greedy exploration policy_** to learn a **_multimodal distribution_** that aligns with landscape of the Q value.

:wrench: Realized in Ubuntu 20.04 and Pytorch over OpenAI Gym benchamark: Atari Game and Classic Control. 

# Video: Multi-Human Demonstration
Distinct genders, age groups, and driving proficiency.

https://github.com/OscarHuangWind/Learning-from-Intervention/assets/41904672/c6259c23-4a88-4283-9aa0-579e0b1ae2c9

# Video: Unprotected Left Turn
https://github.com/OscarHuangWind/Learning-from-Intervention/assets/41904672/8227ce00-73e5-4037-9192-5a837e5c2500

# User Guide

## Clone the repository.
cd to your workspace and clone the repo.
```
git clone https://github.com/OscarHuangWind/Preference-Guided-DQN-Atari.git
```

## Create a new virtual environment with dependencies.
```
cd ~/$your workspace/Preference-Guided-DQN-Atari
conda env create -f virtual_env.yml
```
You can modify your virtual environment name and dependencies in **virtual_env.yml** file.

## Activate virtual environment.
```
conda activate pgdqn
```

## Install Pytorch
Select the correct version based on your cuda version and device (cpu/gpu):
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

## If it shows error that the game is not installed, then please manually install the ROMS.
```
# Download Roms.rar from the [ Atari 2600 VCS ROM Collection](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html)
# Extract the .rar file and import them

python -m atari_py.import_roms <path to extracted folder>

```

## PGDQN Training
```
python main.py
```

## Visualization
Modify the VISUALIZATION parameter in **config.yaml** file to "True", and run:
```
python main.py
```

## DQN, DOUBLE_DQN, DUELING_DQN, D3QN, UCB_DQN Training
Modify the corresponding name of the **_algo_** in **main.py** file, and run:
```
python main.py
```

## Parameters
Feel free to play with the parameters in **config.yaml**. 