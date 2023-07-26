# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import copy
from tqdm import tqdm
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence
import torch.nn.functional as F
import pdb
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from entity_dataset2 import EntityDataset
from basic_dataset import BasicDataset
from torch.distributed.elastic.multiprocessing.errors import record

os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    basic_dataset : bool = field(default=False)
    pickle_path : str = field(default="data/EntityDataset_chat_usmle/int16_ep_0.pkl",
                              metadata={"help": "Path to the training data."})
    kg_path : str = field(default="data/umls_kg_filter_count_5.csv")
    data_path : str = field(default="data/kg_chat_usmle_10178.json")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


@dataclass
class DataCollatorForEntityDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [b[0] for b in instances]
        attention_mask = [b[1] for b in instances]
        labels = [b[2] for b in instances]
        
        bsz = len(input_ids)
        max_len = max([x.shape[-1] for x in input_ids])
        input_ids = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=self.tokenizer.pad_token_id) for x in input_ids]).view(bsz, max_len).long()
        attention_mask = torch.stack([F.pad(x, (max_len - x.shape[-1], 0, 0, max_len - x.shape[-1]), mode='constant', value=0) for x in attention_mask]).view(bsz, 1, max_len, max_len).float()
        attention_mask[attention_mask==0] = float('-inf')
        attention_mask[attention_mask==1] = 0
        labels = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=-100) for x in labels]).view(bsz, max_len).long() if labels else None
        
        # hard_position_type_ids = [b[3] for b in instances]
        # hard_position_type_ids = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=-1) for x in hard_position_type_ids]).view(bsz, max_len)
        # return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels, hard_position_type_ids=hard_position_type_ids)
        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

@dataclass
class DataCollatorForBasicDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [b[0] for b in instances]
        attention_mask = [b[1] for b in instances]
        labels = [b[2] for b in instances]
        
        bsz = len(input_ids)
        max_len = max([x.shape[-1] for x in input_ids])
        input_ids = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=self.tokenizer.pad_token_id) for x in input_ids]).view(bsz, max_len).long()
        attention_mask = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=0) for x in attention_mask]).view(bsz, max_len).long()
        labels = torch.stack([F.pad(x, (max_len - x.shape[-1], 0), mode='constant', value=-100) for x in labels]).view(bsz, max_len).long() if labels else None
        
        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.basic_dataset:
        train_dataset = BasicDataset(data_path=data_args.data_path, tokenizer=tokenizer, from_pickle=data_args.pickle_path)
    else:
        train_dataset = EntityDataset(data_path=data_args.data_path, kg_path=data_args.kg_path, tokenizer=tokenizer, from_pickle=data_args.pickle_path)
    
    if data_args.basic_dataset:
        data_collator = DataCollatorForBasicDataset(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForEntityDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

@record
def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    
    # 删除llama的_prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        return attention_mask
    
    if not data_args.basic_dataset:
        tok = tokenizer
        # 调整tokenizer
        tok.padding_side = 'right'
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
        tok.add_tokens(["[DASH]"])
        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
        # 在模型中添加dash_token
        old_shape = model.model.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tok))
        dash_token_id = tok.convert_tokens_to_ids("[DASH]")
        # Add the new token to the end of the embedding table
        new_shape = model.model.embed_tokens.weight.shape
        print(f"DASH token is added to tokenizer and model\nDASH token id:{dash_token_id}\nEmbedding shape change from:{old_shape} to {new_shape}")
    
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
