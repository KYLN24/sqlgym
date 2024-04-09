import sqlite3

from gymnasium import Env

from .datasets import DbDataset, SqlGymEnvModeEnum


class SqlGymEnv(Env):
    def __init__(self, dataset: DbDataset) -> None:
        self.idx = None
        self.dataset = dataset
        self.conn = None

    def reset(  # pylint: disable=arguments-differ
        self,
        idx,
        seed=None,
        options=None,
    ) -> str:
        super().reset(seed=seed, options=options)
        self.idx = idx
        self.conn = sqlite3.connect(self.dataset[idx].path)
        return self.dataset[idx].query

    def _get_ground_truth(self) -> str:
        cursor = self.conn.cursor()
        cursor.execute(self.dataset[self.idx].gt)
        gt = cursor.fetchall()
        return gt

    def _get_reward(self, execution_result: list | Exception) -> float:
        if isinstance(execution_result, Exception):
            return 0.0
        gt = self._get_ground_truth()
        if set(execution_result) == set(gt):
            return 1.0
        else:
            return 0.0

    def _exec_sql(self, sql: str) -> list | Exception:
        cursor = self.conn.cursor()
        try:
            cursor.execute(sql)
            predicted_res = cursor.fetchall()
            return predicted_res
        except Exception as e:  # pylint: disable=W0718:broad-exception-caught
            return e

    def step(self, action: str) -> tuple:
        """
        Action is a string of a SQL query.
        """
        execution_result = self._exec_sql(action)
        reward = self._get_reward(execution_result)
        if self.dataset.sql_gym_env_mode == SqlGymEnvModeEnum.SINGLE:
            terminated = True
            info = {"ground_truth": self._get_ground_truth()}
        else:
            raise NotImplementedError
        return execution_result, reward, terminated, info, terminated

    def render(self) -> None:
        """make gymnasium.Env happy."""
        return super().render()
