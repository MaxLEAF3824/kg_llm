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
from basic_dataset import BasicDataset
from treelib import Tree
import time
import uuid
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def main():
    mt_path = '/home/cs/yangyuchen/yushengliao/Medical_LLM/vicuna-7b'
    # mt_path = '/home/cs/yangyuchen/guoyiqiu/kg_llm/output/full_vicuna_7b_chat_baseline_bsz2_epoch3_lr8e-06_46822816'
    # mt_path = '/home/cs/yangyuchen/yushengliao/Medical_LLM/FastChat/checkpoints/medical_llama_13b_chatv1.3/checkpoint-4974/'
    data_path = "data/kg_chat_usmle_10178.json"
    pickle_dataset_template = "data/BasicDataset_chat_usmle/ep_{}.pkl"
    max_len = 2048      
    lr = 8e-6
    warmup_ratio = 0.04
    num_epochs = 3
    batch_size = 1
    # device = 'cuda'
    out_dir = f"output/full_vicuna_7b_chat_usmle_baseline_ft_bsz{batch_size}_epoch{num_epochs}_lr{lr}_{str(uuid.uuid4().int)[:8]}"
    
    start_time = time.time()
    accelerator = Accelerator()

    model = AutoModelForCausalLM.from_pretrained(mt_path)
    tok = AutoTokenizer.from_pretrained(mt_path)
    
    # 调整tokenizer
    tok.padding_side = 'right'
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

    # 快速初始化数据集
    dataset = BasicDataset(data_path, tok, max_len=max_len,from_pickle=pickle_dataset_template.format(0))
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
    
    device = accelerator.device
    accelerator.print("preparing optimizer and scheduler")
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    accelerator.print(f"prepare time:{time.time()-start_time}")
    accelerator.print("start training")
    for epoch in range(num_epochs):
        
        accelerator.print("preparing dl")
        # dl = accelerator.prepare(dl)
        
        for step,(input_ids, attention_mask, labels) in enumerate(dl):
            # input_ids = input_ids.to(device)
            # attention_mask = attention_mask.to(device)
            # labels = labels.to(device)

            # accelerator.print("model forward")
            res = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = res.loss

            # print(f"Epoch:{epoch+1}/{num_epochs} Step:{step+1}/{len(dl)} lr:{scheduler.get_lr()} loss:{loss} loss_lm:{loss_lm} loss_kg:{loss_kg}")
            # loss.backward()
            accelerator.print(f"Epoch:{epoch+1}/{num_epochs} Step:{step+1}/{len(dl)} lr:{scheduler.get_lr()} loss:{loss}")
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if epoch != num_epochs -1:
            dataset = BasicDataset(data_path, tok, max_len=max_len, from_pickle=pickle_dataset_template.format(epoch+1))
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        accelerator.print(f"train epoch{epoch+1} time:{time.time()-start_time}")


    
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