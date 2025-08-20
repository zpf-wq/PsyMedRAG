from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
import pandas as pd
from peft import LoraConfig, TaskType
from transformers import TrainingArguments
from datasets import Dataset
from swanlab.integration.transformers import SwanLabCallback
from peft import get_peft_model
from transformers import Trainer
import os
from os.path import join
import torch

model_id = "deepseek-ai/deepseek-llm-7b-chat"

model_kwargs = {
    "torch_dtype": torch.float16,
    "use_cache": True,
    "device_map": "auto"
}
model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

print("模型：", model)
print("分词器：", tokenizer)

data_path = "/Users/zhangpengfei/PycharmProjects/RAG/DeepSeek-LLM-7B-Chat/medical_multi_data.json"
data = pd.read_json(data_path)
train_ds = Dataset.from_pandas(data)
print(train_ds)


def process_data(data, tokenizer, max_seq_length):
    input_ids, attention_mask, labels = [], [], []

    conversations = data["conversation"]
    for i,conv in enumerate(conversations):

        if "instruction" in conv:
            instruction_text = conv['instruction']
        else:
            instruction_text = ""
        human_text = conv["input"]
        assistant_text = conv["output"]

        input_text = f"{tokenizer.bos_token}{instruction_text}\n\nUser:{human_text}\n\nAssistant:"

        input_tokenizer = tokenizer(
            input_text,
            add_special_tokens=False,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        output_tokenizer = tokenizer(
            assistant_text,
            add_special_tokens=False,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids += (
                input_tokenizer["input_ids"] + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
        )
        attention_mask += input_tokenizer["attention_mask"] + output_tokenizer["attention_mask"] + [1]
        labels += ([-100] * len(input_tokenizer["input_ids"]) + output_tokenizer["input_ids"] + [tokenizer.eos_token_id]
                   )

    if len(input_ids) > max_seq_length:  # 做一个截断
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]
        labels = labels[:max_seq_length]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


train_dataset = train_ds.map(process_data,
                             fn_kwargs={"tokenizer": tokenizer, "max_seq_length": tokenizer.model_max_length},
                             remove_columns=train_ds.column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=['up_proj', 'gate_proj', 'q_proj', 'o_proj', 'down_proj', 'v_proj', 'k_proj'],
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False  # 训练模式
    )

output_dir = "./output/deepseek-mutil-test"

train_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=1,
    num_train_epochs=3,
    save_steps=5000,
    learning_rate=2e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to=None,
    seed=42,
    optim="adamw_torch",
    fp16=True,
    bf16=False,
    remove_unused_columns=False,
)

# ------------------------Vision------------------------ #
os.environ["SWANLAB_API_HOST"] = "https://swanlab.115.zone/api"
os.environ["SWANLAB_WEB_HOST"] = "https://swanlab.115.zone"
swanlab_config = {
        "dataset": data_path,
        "peft": "lora"
    }
swanlab_callback = SwanLabCallback(
    project="deepseek-finetune-test",
    experiment_name="first-test",
    description="微调多轮对话",
    workspace=None,
    config=swanlab_config,
)
# ------------------------Vision------------------------ #

model.enable_input_require_grads()

model = get_peft_model(model,lora_config)
model.print_trainable_parameters()

trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[swanlab_callback],
        )

trainer.train()

final_save_path = join(output_dir)
trainer.save_model(final_save_path)