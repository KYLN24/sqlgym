import json
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tqdm import tqdm


class SqlGymEnvModeEnum(Enum):
    SINGLE = "single"
    MULTI_TURN = "multi_turn"


@dataclass
class DbDatasetItem:
    path: Path
    gt: str
    query: str
    info: dict


class DbDataset:
    sql_gym_env_mode: SqlGymEnvModeEnum

    @abstractmethod
    def __getitem__(self, idx: int) -> DbDatasetItem:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


def make_sft_dataset(dataset: DbDataset, save_path: Path):
    with open(save_path, "w", encoding="utf8") as f:
        for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
            f.write(
                json.dumps(
                    {
                        "id": idx,
                        "input": data.query,
                        "output": data.gt,
                    }
                )
                + "\n"
            )
