from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import transformers
from typing import List, Optional, Tuple
import torch
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType, PeftModelForCausalLM
from tokenizers import Tokenizer
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import random
from tqdm.auto import tqdm
import pickle
from functools import partial
import os
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from accelerate import Accelerator
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from entity_dataset2 import EntityDataset
import time
import uuid
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def main():
    mt_path = '/home/cs/yangyuchen/yushengliao/Medical_LLM/vicuna-7b'
    # mt_path = '/home/cs/yangyuchen/yushengliao/Medical_LLM/FastChat/checkpoints/medical_llama_13b_chatv1.3/checkpoint-4974/'
    kg_path = "data/umls_kg_filter_count_5.csv"
    data_path = "data/kg_chat_usmle_10178.json"
    pickle_dataset_template = "data/EntityDataset_chat_usmle/int16_ep_{}.pkl"
    lazy = False

    max_len = 2048
    dash_token = "[DASH]"
    lr = 8e-6
    warmup_ratio = 0.04
    num_epochs = 3
    batch_size = 1
    # device = 'cuda'
    out_dir = f"output/full_vicuna_7b_{str(uuid.uuid4().int)[:8]}"
    
    start_time = time.time()
    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(mt_path)
    tok = AutoTokenizer.from_pretrained(mt_path)
    
    # 调整tokenizer
    tok.padding_side = 'right'
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.add_tokens([dash_token])
    
    # 删除llama的_prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        return attention_mask

    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask

    # 在模型中添加dash_token
    old_shape = model.model.model.embed_tokens.weight.shape if isinstance(model, PeftModelForCausalLM) else model.model.embed_tokens.weight.shape
    model.resize_token_embeddings(len(tok))
    dash_token_id = tok.convert_tokens_to_ids(dash_token)
    # Add the new token to the end of the embedding table
    new_shape = model.model.model.embed_tokens.weight.shape if isinstance(model, PeftModelForCausalLM) else model.model.embed_tokens.weight.shape
    print(f"DASH token is added to tokenizer and model\nDASH token id:{dash_token_id}\nEmbedding shape change from:{old_shape} to {new_shape}")
    

    # 快速初始化数据集
    dataset = EntityDataset(data_path, kg_path, tok, max_len=max_len, lazy=lazy, from_pickle=pickle_dataset_template.format(0))
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    
    accelerator.print("preparing model")
    model = accelerator.prepare(model)
    
    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(len(dl) * num_epochs * warmup_ratio),
        num_training_steps=(len(dl) * num_epochs),
    )
    
    # device = 'cuda:0'
    # model.half()
    # model.to(device)
    accelerator.print("preparing optimizer and scheduler")
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    accelerator.print("preparing dl")
    dl = accelerator.prepare_data_loader(dl)
    accelerator.print(f"prepare time:{time.time()-start_time}")
    accelerator.print("start training")

    for epoch in range(num_epochs):
        accelerator.print("epoch start")
        for step,(input_ids, attention_mask, labels, hard_position_type_ids) in enumerate(dl):
            input_ids = input_ids.long()
            attention_mask = attention_mask.float()  
            attention_mask[attention_mask==0] = float('-inf')
            
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
            labels = labels.to(model.device)

            accelerator.print("data loaded")
            res = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            accelerator.print("forward success")
            
            labels_kg = labels.clone()
            labels_kg[hard_position_type_ids != 3] = -100
            labels_lm = labels.clone()
            labels_lm[hard_position_type_ids == 3] = -100
            
            # accelerator.print("calculating loss")
            logits = res.logits
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels_lm = labels_lm[..., 1:].contiguous()
            shift_labels_kg = labels_kg[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, len(tok))
            shift_labels_lm = shift_labels_lm.view(-1)
            shift_labels_kg = shift_labels_kg.view(-1)
            # Enable model parallelism
            shift_labels_lm = shift_labels_lm.to(shift_logits.device)
            shift_labels_kg = shift_labels_kg.to(shift_logits.device)
            loss_lm = loss_fct(shift_logits, shift_labels_lm)
            loss_kg = loss_fct(shift_logits, shift_labels_kg)
            loss = loss_lm + loss_kg
            # loss = res.loss

            # print(f"Epoch:{epoch+1}/{num_epochs} Step:{step+1}/{len(dl)} lr:{scheduler.get_lr()} loss:{loss} loss_lm:{loss_lm} loss_kg:{loss_kg}")
            # loss.backward()
            accelerator.print(f"Epoch:{epoch+1}/{num_epochs} Step:{step+1}/{len(dl)} lr:{scheduler.get_lr()} loss:{loss} loss_lm:{loss_lm} loss_kg:{loss_kg}")
            # accelerator.print(f"Epoch:{epoch+1}/{num_epochs} Step:{step+1}/{len(dl)} lr:{scheduler.get_lr()} loss:{loss}")
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        # if epoch != num_epochs -1:
        #     # dataset = EntityDataset(data_path, kg_path, tok, max_len=max_len, from_pickle=pickle_dataset_template.format(epoch+1))
        #     dataset = EntityDataset(data_path, kg_path, tok, max_len=max_len, from_pickle=pickle_dataset_template.format(0))
        #     dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        accelerator.print(f"train epoch{epoch+1} time:{time.time()-start_time}")

    # out_dir = f"output/full_vicuna_7b_chat"
    
    # save model
    accelerator.print(f"Saving model and tokenizer to {out_dir}")
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        state = accelerator.get_state_dict(model)
    model.save_pretrained(
        out_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state,
    )
    # model.save_pretrained(out_dir)
    if accelerator.is_main_process:
        tok.save_pretrained(out_dir)
    accelerator.print(f"all done time:{time.time()-start_time}")
if __name__ == "__main__":
    main()