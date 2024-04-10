import json
import math
import os
from dataclasses import dataclass
from typing import Sequence

import torch
from tqdm import tqdm, trange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from sqlgym import SqlGymEnv
from sqlgym.datasets import BirdDataset

RANK = int(os.getenv("RANK", "0"))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

GENERATION_CONFIG = GenerationConfig(max_new_tokens=1024, do_sample=False)


@dataclass
class Arguments:
    model: str
    bird_path: str
    react: bool = False
    save_path: str | None = None
    batch_size: int = 1


class Evaluator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        env: SqlGymEnv,
        react: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.env = env
        self.react = react

    def parse_action(self, generated_text: str):
        return {
            "thought": (
                generated_text.split("Thought:")[-1].split("Action:")[0].strip()
                if self.react
                else None
            ),
            "action": generated_text.split("```sql")[-1].split("```")[0].strip(),
            "generated_text": generated_text,
        }

    def eval_one(self, query: str):
        if self.react:
            messages = [
                {
                    "role": "user",
                    "content": "Given you a description of a SQlite database system, I will ask you a question, then you should help me operate the SQLite database with SQL to answer the question.\n\nYou have to explain the problem and your solution to me and write down your thoughts.\nAfter thinking and explaining thoroughly, every round you can choose to operate or to answer.\n\nyour response should be like this:\nThought: I think...\n\nAction: ```sql\nSELECT * FROM table WHERE condition;\n```\n\nYou MUST put SQL in markdown format without any other comments. Your SQL should be in one line. Every time you can only execute one SQL statement.",
                },
                {"role": "assistant", "content": "Ok."},
                {"role": "user", "content": query},
            ]
        else:
            messages = [{"role": "user", "content": query}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(f"cuda:{LOCAL_RANK}")
        output_ids = self.model.generate(input_ids, generation_config=GENERATION_CONFIG)
        generated_text = self.tokenizer.decode(
            output_ids[0][len(input_ids[0]) :], skip_special_tokens=True
        )
        output = self.parse_action(generated_text)
        return output

    def eval_one_batch(self, queries: Sequence[str]):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        if self.react:
            messages_batch = [
                [
                    {
                        "role": "user",
                        "content": "Given you a description of a SQlite database system, I will ask you a question, then you should help me operate the SQLite database with SQL to answer the question.\n\nYou have to explain the problem and your solution to me and write down your thoughts.\nAfter thinking and explaining thoroughly, every round you can choose to operate or to answer.\n\nyour response should be like this:\nThought: I think...\n\nAction: ```sql\nSELECT * FROM table WHERE condition;\n```\n\nYou MUST put SQL in markdown format without any other comments. Your SQL should be in one line. Every time you can only execute one SQL statement.",
                    },
                    {"role": "assistant", "content": "Ok."},
                    {"role": "user", "content": query},
                ]
                for query in queries
            ]
        else:
            messages_batch = [[{"role": "user", "content": query}] for query in queries]
        prompts_batch = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_batch
        ]
        prompts_batch = self.tokenizer(
            prompts_batch, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(f"cuda:{LOCAL_RANK}")

        output_ids = self.model.generate(
            **prompts_batch,
            generation_config=GENERATION_CONFIG,
        )
        generated_texts = self.tokenizer.batch_decode(
            output_ids[:, prompts_batch.input_ids.shape[1] :], skip_special_tokens=True
        )

        return [self.parse_action(generated_text) for generated_text in generated_texts]

    def eval(self, batch_size: int = 1, ids: Sequence[int] | None = None):
        rewards = []
        results = []
        _iter = list[range(len(self.env.dataset))] if ids is None else ids

        if batch_size == 1:
            for _idx in tqdm(_iter, desc="Evaluating"):
                query = self.env.reset(_iter[_idx])
                output = self.eval_one(query)
                action = output["action"]
                execution_result, reward, _, info, _ = self.env.step(action)
                rewards.append(reward)
                results.append(
                    {
                        "query": query,
                        "action": action,
                        "reward": reward,
                        "execution_result": execution_result,
                        "info": info,
                        "thought": output["thought"],
                        # "generated_text": output["generated_text"],
                    }
                )
        else:
            for _idx in trange(math.ceil(len(_iter) / batch_size), desc="Evaluating"):
                _start = _idx * batch_size
                _end = min((_idx + 1) * batch_size, len(_iter))
                queries = [self.env.reset(_iter[i]) for i in range(_start, _end)]
                outputs = self.eval_one_batch(queries)
                for i, output in enumerate(outputs):
                    _ = self.env.reset(_iter[i + _idx * batch_size])
                    action = output["action"]
                    execution_result, reward, _, info, _ = self.env.step(action)
                    rewards.append(reward)
                    results.append(
                        {
                            "query": queries[i],
                            "action": action,
                            "reward": reward,
                            "execution_result": execution_result,
                            "info": info,
                            "thought": output["thought"],
                            # "generated_text": output["generated_text"],
                        }
                    )
        if WORLD_SIZE > 1:
            torch.distributed.barrier()
        return results


def main(args: Arguments):
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .to(f"cuda:{LOCAL_RANK}")
    )

    dataset = BirdDataset(
        bird_path=args.bird_path,
        mode="dev",
    )

    env = SqlGymEnv(dataset)
    evaluator = Evaluator(model, tokenizer, env, args.react)
    if WORLD_SIZE > 1:
        # _ids = list(range(len(dataset)))
        _ids = list(range(8))
        _per_device_len = math.ceil(len(_ids) / WORLD_SIZE)
        ids = _ids[RANK * _per_device_len : (RANK + 1) * _per_device_len]
    else:
        ids = None

    results = evaluator.eval(batch_size=args.batch_size, ids=ids)

    if WORLD_SIZE > 1:
        _results = [None for _ in range(WORLD_SIZE)]
        torch.distributed.gather_object(results, _results if RANK == 0 else None, dst=0)
        if RANK == 0:
            results = sum(_results, [])

    if RANK == 0:
        if args.save_path is not None:
            with open(args.save_path, "w", encoding="utf8") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
        rewards = [r["reward"] for r in results]
        print(f"Mean reward: {sum(rewards) / len(rewards)}")


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    if WORLD_SIZE > 1:
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)
    main(parser.parse_args())
