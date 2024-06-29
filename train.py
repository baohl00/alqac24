import os
import torch
import random
import pathlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import transformers
from trl import SFTTrainer
from datasets import load_dataset
from transformers import (
        AutoModelForCausalLM, AutoTokenizer, 
        deepspeed, BitsAndBytesConfig
        )

from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

SEED = 42
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:512"
INS_PROMPT = '''You are a helpful Vietnamese legal assistant with the mission of answering the question based on the given article without explanation.

### Article: {article}

### Question: {question}\n'''

CHOICE = '''{choices}'''

INS_ANS = "### Answer: {answer}"

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class LoraArguments:
    lora_enable:bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    val_nums: int = field(default=None, metadata={"help": "Number of samples in validation set"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
#    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
#    model_max_length: int = field(
#        default=None,
#        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
#    )
#    overwrite_output_dir: bool = field(default=True)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

# Instruction

def formatting_option(options):
    return f"A) {options['A']}\nB) {options['B']}\nC) {options['C']}\nD) {options['D']}"

def formatting_func(sample):
    text = INS_PROMPT.format(article = sample['article'], question = sample['text'])
    if sample['question_type'] == 'Trắc nghiệm':
        text += CHOICE.format(choices = formatting_option(sample['choices']))

    text += INS_ANS.format(answer = sample['answer'])
#        formatted_texts.append(text)
#    text = INS_PROMPT.format(question=sample['question'], choices=choices, answer=ans)
    
    return text

def formatting_func1(sample):
    formatted_texts = []
    for article, text, question_type, choices, answer in zip(sample['article'], sample['text'], sample['question_type'], sample['choices'], sample['answer']):
        text = INS_PROMPT.format(article = article, question = text)
        if question_type == 'Trắc nghiệm':
            text += CHOICE.format(choices = formatting_option(options = choices))
        text += INS_ANS.format(answer = answer)
        formatted_texts.append(text)
    return formatted_texts



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    
    json_data = load_dataset('json', data_files=data_args.data_path)
    print(json_data)
    if data_args.val_nums is None:
        train_data = json_data['train']
        val_data = None
    else:
        dataset = json_data['train'].train_test_split(test_size=data_args.val_nums, shuffle=True, seed=42)
        train_data = dataset['train']
        val_data = dataset['test']

    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, 
            quantization_config=quantization_config,
            use_auth_token=True,
            #torch_dtype = torch.float16
            )
    if lora_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
            
        if training_args.local_rank == 0:
            print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
        model_max_length=4096,
        padding_side="right",
        use_fast=False,
    )

    #if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        dataset_text_field="text",
#        packing=True,
        tokenizer=tokenizer,
        max_seq_length=4096,
        formatting_func=formatting_func1
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
        if training_args.local_rank == 0:
            state_dict = state_dict_zero3
    else:
        # in other mode we use original code from fastchat team, to make sure our change is minimum
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), lora_args.lora_bias
        )

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)

if __name__ == "__main__":
    train()
