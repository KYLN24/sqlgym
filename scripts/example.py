from sqlgym import SqlGymEnv
from sqlgym.datasets import BirdDataset

dataset = BirdDataset(
    bird_path="../.data/bird",
    mode="dev",
)

env = SqlGymEnv(dataset)

print(env.reset(0))
print(env.step(dataset[0].gt))
