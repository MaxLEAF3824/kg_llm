from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
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
from entity_dataset import EntityDataset
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def main():
    accelerator = Accelerator()
    
    mt_path = '/home/cs/yangyuchen/yushengliao/Medical_LLM/vicuna-7b'
    mt_path = '/home/cs/yangyuchen/yushengliao/Medical_LLM/FastChat/checkpoints/medical_llama_13b_chatv1.3/checkpoint-4974/'
    kg_path = "data/umls_kg_filter.csv"
    data_path = "data/kg_instruction_1000.json"
    max_len = 1024
    dash_token = "[DASH]"
    lr = 1e-5
    num_warmup_steps = 10
    num_epochs = 3
    batch_size = 1
    # device = 'cuda'
    

    model = AutoModelForCausalLM.from_pretrained(mt_path)
    tok = AutoTokenizer.from_pretrained(mt_path)
    n_layer = model.config.num_hidden_layers
    
    # 调整tokenizer
    tok.padding_side = 'right'
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    
    # 添加dash_token
    tok.add_tokens([dash_token])
    old_shape = model.model.model.embed_tokens.weight.shape if isinstance(model, PeftModelForCausalLM) else model.model.embed_tokens.weight.shape
    model.resize_token_embeddings(len(tok))
    dash_token_id = tok.convert_tokens_to_ids(dash_token)
    # Add the new token to the end of the embedding table
    new_shape = model.model.model.embed_tokens.weight.shape if isinstance(model, PeftModelForCausalLM) else model.model.embed_tokens.weight.shape
    print(f"DASH token is added to tokenizer and model\nDASH token id:{dash_token_id}\nEmbedding shape change from:{old_shape} to {new_shape}")
    
    # 添加peft模块
    # print("Adding peft to model")
    # peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj","v_proj"])
    # model = get_peft_model(model, peft_config)
    # model.half()
    # model.to(device)
    # model.print_trainable_parameters()

    # 快速初始化数据集
    dataset = EntityDataset(data_path, kg_path, tok, max_len=max_len)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    
    model = accelerator.prepare(model)
    
    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=(len(dl) * num_epochs),
    )
    
    device = accelerator.device
    optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    
    for epoch in range(num_epochs):
        
        dl = accelerator.prepare(dl)

        for step,(input_ids, attention_mask, labels, hard_position_type_ids) in enumerate(dl):
            # input_ids = input_ids.to(device)
            # attention_mask = attention_mask.to(device)
            # labels = labels.to(device)
            
            def pre_hook(module, args,  kwargs):
                kwargs['attention_mask'] = attention_mask.to(kwargs['attention_mask'].device)
                return args, kwargs
            
            hooks = []
            for i in range(n_layer):
                attn = model.model.model.layers[i].self_attn if isinstance(model, PeftModelForCausalLM) else model.model.layers[i].self_attn
                hooks.append(attn.register_forward_pre_hook(pre_hook, with_kwargs=True))
            
            res = model(input_ids=input_ids, attention_mask=None, labels=None)
            
            for hook in hooks:
                hook.remove()
            
            labels_kg = labels.clone()
            labels_kg[hard_position_type_ids != 3] = -100
            labels_lm = labels.clone()
            labels_lm[hard_position_type_ids == 3] = -100
            
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

            # print(f"Epoch:{epoch+1}/{num_epochs} Step:{step+1}/{len(dl)} lr:{scheduler.get_lr()} loss:{loss} loss_lm:{loss_lm} loss_kg:{loss_kg}")
            # loss.backward()
            accelerator.print(f"Epoch:{epoch+1}/{num_epochs} Step:{step+1}/{len(dl)} lr:{scheduler.get_lr()} loss:{loss} loss_lm:{loss_lm} loss_kg:{loss_kg}")
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        if epoch != num_epochs -1:
            dataset = EntityDataset(data_path, kg_path, tok, max_len=max_len)
            dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    model.save_pretrained(f"output/full_book_13b_bsz{batch_size}_epoch{num_epochs}_lr{lr}")
    tok.save_pretrained(f"output/full_book_13b_bsz{batch_size}_epoch{num_epochs}_lr{lr}")

if __name__ == "__main__":
    main()