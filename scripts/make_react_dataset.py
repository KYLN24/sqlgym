import argparse
import asyncio
import itertools
import json
import random

from openai import AsyncOpenAI
from tqdm import tqdm

NUM_SAMPLES = (
    3000  # The number of samples to generate. If None, all samples will be generated.
)

SEED = 3407

MAX_CONCURRENCY = 4  # The number of concurrent requests to OpenAI API.
MODEL = "gpt-3.5-turbo"  # GPT 3.5 Turbo is enough for this task. But you can try GPT 4 Turbo if you want.
# MODEL = "gpt-4-turbo-preview"
API_KEYS = itertools.cycle(
    []
)  # Add your keys here. You can add multiple keys to avoid rate limits.


# Generation parameters
MAX_TOKENS = 1024
TEMPERATURE = 0.6
TOP_P = 0.95

SYS_PROMPT = """You are teaching a course on database."""
PROMPT_TEMPLATE = """Question:
<|PLACE_HOLDER_FOR_INPUT|>

Answer:
```sql
<|PLACE_HOLDER_FOR_OUTPUT|>
```

Please provide your thought to solve the question. You should give the thought in only one paragraph with no more than 3 sentences.
"""

prompt_tokens = 0
completion_tokens = 0


async def complete(prompt, ttl=10):
    client = AsyncOpenAI(api_key=next(API_KEYS))
    if ttl == 0:
        return ""
    try:
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=SEED,
        )

        global prompt_tokens, completion_tokens  # pylint: disable=W0603:global-statement
        prompt_tokens += completion.usage.prompt_tokens
        completion_tokens += completion.usage.completion_tokens

        return completion.choices[0].message.content
    except Exception as e:  # pylint: disable=W0718:broad-exception-caught
        print(e)
        await asyncio.sleep(2)
        return await complete(prompt, ttl - 1)


async def ask_with_sem(prompt, sem, pbar):
    async with sem:
        completion = await complete(prompt)
        pbar.update(1)
        return completion


async def ask(data):
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    with tqdm(total=len(data)) as pbar:
        responses = await asyncio.gather(
            *[
                (
                    ask_with_sem(
                        PROMPT_TEMPLATE.replace(
                            "<|PLACE_HOLDER_FOR_INPUT|>", sample["input"]
                        ).replace("<|PLACE_HOLDER_FOR_OUTPUT|>", sample["output"]),
                        sem,
                        pbar,
                    )
                    if (sample.get("thought") is None or sample.get("thought") == "")
                    else sample["thought"]
                )
                for sample in data
            ]
        )
    return responses


def format_data_item(d):
    return {
        "item_id": d["id"],
        "conversation": [
            {
                "from": "human",
                "value": "Given you a description of a SQlite database system, I will ask you a question, then you should help me operate the SQLite database with SQL to answer the question.\n\nYou have to explain the problem and your solution to me and write down your thoughts.\nAfter thinking and explaining thoroughly, you should give a SQL statement to solve the question.\n\nyour response should be like this:\nThought: Your thought here.\n\nAction: ```sql\nSELECT * FROM table WHERE condition;\n```\n\nYou MUST put SQL in markdown format without any other comments. Your SQL should be in one line. Every time you can only execute one SQL statement.",
                "loss": None,
            },
            {"from": "gpt", "value": "Ok.", "loss": False},
            {"from": "human", "value": d["input"], "loss": None},
            {
                "from": "gpt",
                "value": f"Thought: {d['thought']}\n\nAction: ```sql\n{d['output']}\n```",
                "loss": True,
            },
        ],
    }


async def main(args):
    with open(args.data_path, "r", encoding="utf8") as f:
        data = [json.loads(line) for line in f]

    random.seed(SEED)
    if NUM_SAMPLES is not None and NUM_SAMPLES < len(data):
        data = random.sample(data, NUM_SAMPLES)
    responses = await ask(data)

    for sample, response in zip(data, responses):
        sample["thought"] = response
        sample["data_item"] = format_data_item(sample)

    with open(args.save_path, "w", encoding="utf8") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    asyncio.run(main(parser.parse_args()))
    print("Prompt tokens:", prompt_tokens)
    print("Completion tokens:", completion_tokens)
