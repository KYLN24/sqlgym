from pathlib import Path

from sqlgym.datasets import BirdDataset, make_sft_dataset

BIRD_PATH = Path("../.data/bird")

make_sft_dataset(
    BirdDataset(
        bird_path=BIRD_PATH,
        mode="dev",
    ),
    BIRD_PATH.joinpath("dev.jsonl"),
)

make_sft_dataset(
    BirdDataset(
        bird_path=BIRD_PATH,
        mode="train",
    ),
    BIRD_PATH.joinpath("train.jsonl"),
)
