import json
import math
from dataclasses import dataclass
from typing import Sequence

import torch
from tqdm import trange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from sqlgym import SqlGymEnv
from sqlgym.datasets import BirdDataset


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
                generated_text.split("Thought:")[1].split("Action:")[0].strip()
                if self.react
                else None
            ),
            "action": generated_text.split("```sql")[-1].split("```")[0].strip(),
        }

    def eval_one(self, query: str):
        if self.react:
            messages = [
                {
                    "role": "user",
                    "content": "Given you a description of a SQlite database system, I will ask you a question, then you should help me operate the SQLite database with SQL to answer the question.\n\nYou have to explain the problem and your solution to me and write down your thoughts.\nAfter thinking and explaining thoroughly, every round you can choose to operate or to answer.\n\nyour response should be like this:\nThought: I think...\n\nAction: ```sql\nSELECT * FROM table WHERE condition;\n```\n\nYou MUST put SQL in markdown format without any other comments. Your SQL should be in one line. Every time you can only execute one SQL statement.",
                },
                {"role": "assistent", "content": "Ok."},
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
        ).input_ids.to("cuda")
        output_ids = self.model.generate(input_ids)
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
                    {"role": "assistent", "content": "Ok."},
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
        ).to("cuda")

        output_ids = self.model.generate(**prompts_batch)
        generated_texts = self.tokenizer.batch_decode(
            output_ids[:, prompts_batch.input_ids.shape[1] :], skip_special_tokens=True
        )

        return [self.parse_action(generated_text) for generated_text in generated_texts]

    def eval(self, batch_size):
        rewards = []
        results = []

        if batch_size == 1:
            for idx in trange(len(self.env.dataset)):
                query = self.env.reset(idx)
                output = self.eval_one(query)
                action = output["action"]
                thought = output["thought"]
                execution_result, reward, _, info, _ = self.env.step(action)
                rewards.append(reward)
                results.append(
                    {
                        "query": query,
                        "action": action,
                        "reward": reward,
                        "execution_result": execution_result,
                        "info": info,
                        "thought": thought,
                    }
                )
        else:
            for idx in trange(math.ceil(len(self.env.dataset) / batch_size)):
                queries = [
                    self.env.reset(i)
                    for i in range(idx * batch_size, (idx + 1) * batch_size)
                ]
                outputs = self.eval_one_batch(queries)
                for i, action in enumerate(outputs):
                    _ = self.env.reset(i + idx * batch_size)
                    action = output["action"]
                    thought = output["thought"]
                    execution_result, reward, _, info, _ = self.env.step(action)
                    rewards.append(reward)
                    results.append(
                        {
                            "query": queries[i],
                            "action": action,
                            "reward": reward,
                            "execution_result": execution_result,
                            "info": info,
                            "thought": thought,
                        }
                    )

        print(f"Average reward: {sum(rewards) / len(rewards)}")
        return results


def main(args: Arguments):
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    ).cuda()

    dataset = BirdDataset(
        bird_path=args.bird_path,
        mode="dev",
    )
    env = SqlGymEnv(dataset)
    evaluator = Evaluator(model, tokenizer, env, args.react)
    results = evaluator.eval(batch_size=args.batch_size)
    if args.save_path is not None:
        with open(args.save_path, "w", encoding="utf8") as f:
            f.writelines([json.dumps(r) for r in results])


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    main(parser.parse_args())
