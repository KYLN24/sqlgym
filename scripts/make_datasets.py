import argparse
from pathlib import Path

from sqlgym.datasets import BirdDataset, make_sft_dataset


def main(args):
    bird_path = Path(args.bird_path)
    make_sft_dataset(
        BirdDataset(
            bird_path=bird_path,
            mode="dev",
        ),
        bird_path.joinpath("dev.jsonl"),
    )

    make_sft_dataset(
        BirdDataset(
            bird_path=bird_path,
            mode="train",
        ),
        bird_path.joinpath("train.jsonl"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bird_path", type=str)
    main(parser.parse_args())
