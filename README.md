# SQLGym

This is a portable Gymnasium environment of SQLite database. It is designed for platforms that are not able to use docker. (e.g. users without root privillege)

## Setup

```bash
# Clone this repository
git clone https://github.com/KYLN24/sqlgym.git
# or via SSH
# git clone git@github.com:KYLN24/sqlgym.git

cd sqlgym

# Install this package
pip install .
```

## Prepare Dataset

```
# Make a directory to save data
mkdir .data
cd .data
```

This project currently suppport the BIRD-SQL dataset.

```bash
mkdir bird
cd bird

# Download BIRD-SQL Dataset
wget -c https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
unzip train.zip
cd train
unzip train_databases.zip
cd ..

wget -c https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip dev.zip
cd dev
unzip dev_databases.zip
cd ..
```

# Usage

```python
from sqlgym import SqlGymEnv
from sqlgym.datasets import BirdDataset

dataset = BirdDataset(
    bird_path=".data/bird",
    mode="dev",
)

env = SqlGymEnv(dataset)

print(env.reset(0))
print(env.step(dataset[0].gt))
```
