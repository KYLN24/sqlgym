import json
from pathlib import Path
from typing import Dict, List

from .utils import DbDataset, DbDatasetItem, SqlGymEnvModeEnum


class BirdDataset(DbDataset):
    def __init__(self, bird_path, mode):
        self.bird_path = Path(bird_path)
        self.sql_gym_env_mode = SqlGymEnvModeEnum.SINGLE
        self.mode = mode
        with open(
            self.bird_path.joinpath(self.mode, f"{self.mode}.json"),
            "r",
            encoding="utf8",
        ) as f:
            self._data = json.load(f)

        with open(
            self.bird_path.joinpath(self.mode, f"{self.mode}_tables.json"),
            "r",
            encoding="utf8",
        ) as f:
            self.tables = json.load(f)
        self.tables = {table["db_id"]: table for table in self.tables}

    @staticmethod
    def serialize_schema_natural_language(
        db_id: str,
        db_column_names: Dict[str, str],
        db_table_names: List[str],
        db_primary_keys,
        db_foreign_keys,
        normalize_query: bool = True,
    ) -> str:
        """
        This function is adopted from https://github.com/AlibabaResearch/DAMO-ConvAI/blob/611141c7e9846b82224d0f2e0a6cedbb54ce09e8/bird/finetuning/seq2seq_construction/bird.py#L137
        """
        overall_description = (
            f"{db_id} contains tables such as "
            f'{", ".join([table_name.lower() if normalize_query else table_name for table_name in db_table_names])}.'
        )
        table_description_primary_key_template = (
            lambda table_name, primary_key: f"{primary_key} is the primary key."
        )
        table_description = (
            lambda table_name, column_names: f'Table {table_name} has columns such as {", ".join(column_names)}.'
        )
        value_description = (
            lambda column_value_pairs: f'{"".join(["The {} contains values such as {}.".format(column, value) for column, value in column_value_pairs])}'
        )
        foreign_key_description = (
            lambda table_1, column_1, table_2, column_2: f"The {column_1} of {table_1} is the foreign key of {column_2} of {table_2}."
        )

        descriptions = [overall_description]
        db_table_name_strs = []
        db_column_name_strs = []
        for table_id, table_name in enumerate(db_table_names):
            table_name_str = table_name.lower() if normalize_query else table_name
            db_table_name_strs.append(table_name_str)
            columns = []
            column_value_pairs = []
            primary_keys = []
            for column_id, (x, y) in enumerate(db_column_names):
                if column_id == 0:
                    continue
                column_str = y.lower() if normalize_query else y
                db_column_name_strs.append(column_str)
                if x == table_id:
                    columns.append(column_str)
                    if column_id in db_primary_keys:
                        primary_keys.append(column_str)

            table_description_columns_str = table_description(table_name_str, columns)
            descriptions.append(table_description_columns_str)
            table_description_primary_key_str = table_description_primary_key_template(
                table_name_str, ", ".join(primary_keys)
            )
            descriptions.append(table_description_primary_key_str)
            if len(column_value_pairs) > 0:
                value_description_str = value_description(column_value_pairs)
                descriptions.append(value_description_str)

        for x, y in db_foreign_keys:
            # get the table and column of x
            x_table_name = db_table_name_strs[db_column_names[x][0]]
            x_column_name = db_column_name_strs[x]
            # get the table and column of y
            y_table_name = db_table_name_strs[db_column_names[y][0]]
            y_column_name = db_column_name_strs[y]
            foreign_key_description_str = foreign_key_description(
                x_table_name, x_column_name, y_table_name, y_column_name
            )
            descriptions.append(foreign_key_description_str)
        return " ".join(descriptions)

    def _format_instruction(self, idx: int) -> str:
        database_desciption = self.serialize_schema_natural_language(
            db_id=self._data[idx]["db_id"],
            db_column_names=self.tables[self._data[idx]["db_id"]]["column_names"],
            db_table_names=self.tables[self._data[idx]["db_id"]]["table_names"],
            db_primary_keys=self.tables[self._data[idx]["db_id"]]["primary_keys"],
            db_foreign_keys=self.tables[self._data[idx]["db_id"]]["foreign_keys"],
        )

        return f"{database_desciption}\n\n{self._data[idx]['question']}"

    def __getitem__(self, idx: int) -> DbDatasetItem:
        return DbDatasetItem(
            path="file:"
            + self.bird_path.joinpath(
                self.mode,
                f"{self.mode}_databases",
                self._data[idx]["db_id"],
                f'{self._data[idx]["db_id"]}.sqlite',
            ).as_posix()
            + "?mode=ro",
            gt=self._data[idx]["SQL"],
            query=self._format_instruction(idx),
            info={
                "evidence": self._data[idx]["evidence"],
                "difficulty": (
                    self._data[idx]["difficulty"]
                    if "difficulty" in self._data[idx]
                    else ""
                ),
            },
        )

    def __len__(self) -> int:
        return len(self._data)
