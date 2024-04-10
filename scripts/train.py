import json
import os
import random
from dataclasses import dataclass

import datasets
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


@dataclass
class Arguments:
    model: str
    train_set: str
    output_dir: str
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_train_epochs: int = 1
    warmup_ratio: float = 0.1
    seed: int = 42
    gradient_checkpointing: bool = False
    zero_stage: int = 0
    bf16: bool = True
    tf32: bool = True
    num_samples: int | None = None
    chat_template: str | None = (
        "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content | trim + ' ' + eos_token }}{% endif %}{% endfor %}"
    )
    react: bool = False


def make_dataset(
    path: str,
    tokenizer: PreTrainedTokenizerBase,
    seed: int = 42,
    num_samples: int | None = None,
    react: bool = False,
):
    def format_sample(sample):
        conversation = [{"role": "user", "content": sample["input"]}]
        _input = tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
        _output = tokenizer.encode(sample["output"], add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        return {
            "input_ids": _input + _output,
            "labels": [-100] * len(_input) + _output,
        }

    def format_sample_react(sample):
        conversation = [
            {
                "role": "user" if c["from"] == "human" else "assistant",
                "content": c["value"],
            }
            for c in sample["data_item"]["conversation"]
        ]
        _input = tokenizer.apply_chat_template(
            conversation[:-1], add_generation_prompt=True
        )
        _output = tokenizer.encode(
            conversation[-1]["content"], add_special_tokens=False
        ) + [tokenizer.eos_token_id]
        return {
            "input_ids": _input + _output,
            "labels": [-100] * len(_input) + _output,
        }

    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f.readlines()]

    random.seed(seed)
    if num_samples is not None and num_samples < len(data):
        data = random.sample(data, num_samples)

    dataset = datasets.Dataset.from_list(data)

    dataset = dataset.map(
        format_sample_react if react else format_sample,
        keep_in_memory=True,
        remove_columns=dataset.column_names,
    )
    return dataset


def get_collate(tokenizer: PreTrainedTokenizerBase):
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    def collate_fn(batch):
        input_ids_with_pad = tokenizer.pad(
            {"input_ids": [b["input_ids"] for b in batch]},
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels_with_pad = tokenizer.pad(
            {"input_ids": [b["labels"] for b in batch]},
            return_attention_mask=False,
            return_tensors="pt",
        )
        return {
            "input_ids": input_ids_with_pad.input_ids,
            "attention_mask": input_ids_with_pad.attention_mask,
            "labels": labels_with_pad.input_ids,
        }

    return collate_fn


def main(args: Arguments):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.chat_template is not None:
        tokenizer.chat_template = args.chat_template

    dataset = make_dataset(
        args.train_set, tokenizer, args.seed, args.num_samples, args.react
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            do_train=True,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            num_train_epochs=args.num_train_epochs,
            lr_scheduler_type="cosine",
            warmup_ratio=args.warmup_ratio,
            logging_first_step=True,
            logging_steps=1,
            save_strategy="epoch",
            seed=args.seed,
            bf16=args.bf16,
            tf32=args.tf32,
            ddp_backend="nccl",
            dataloader_num_workers=2,
            deepspeed={
                "bf16": {"enabled": "auto"},
                "zero_optimization": {
                    "stage": args.zero_stage,
                },
                "zero_allow_untested_optimizer": True,
                "gradient_accumulation_steps": "auto",
                "gradient_clipping": "auto",
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
            },
            dataloader_persistent_workers=True,
            gradient_checkpointing=args.gradient_checkpointing,
        ),
        data_collator=get_collate(tokenizer),
        train_dataset=dataset,
        tokenizer=tokenizer,
    ).train()

    tokenizer.save_pretrained(f"{args.output_dir}/model")
    model.save_pretrained(f"{args.output_dir}/model")


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    main(parser.parse_args())
