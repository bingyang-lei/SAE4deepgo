import os
import sys
sys.path.append('./')
import numpy as np

import torch
from datasets import load_dataset
from datasets import concatenate_datasets


import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_constant_schedule
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
import logging
import random
import time
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

def custom_collate_fn(batch):
    # 假设 batch 是一个包含字典的列表，每个字典代表一个样本
    collated_batch = {}
    
    for key in batch[0].keys():
        collated_batch[key] = [d[key] for d in batch]  # 保留所有键的值
    
    collated_batch["input_ids"] = torch.tensor(collated_batch["input_ids"])
    collated_batch["labels"] = torch.tensor(collated_batch["labels"])
    collated_batch["attention_mask"] = torch.tensor(collated_batch["attention_mask"])
    collated_batch["final_pos"] = np.array(collated_batch["final_pos"])
    # 可以在这里对 collated_batch 进行进一步处理，例如将列表转换为 tensor
    # 例如，对于数值型数据，可以使用 torch.tensor() 来转换
    
    return collated_batch

def get_dataset(dataset_name, tokenizer):
    
    # dataset format: input_ids, attention_mask, tags, sequences, final_pos, 
    if dataset_name == "ctrlprot_dataset":
    
        dataset_dict = {"process/train0.csv":"GO:1",}
                        # "function/1.tsv":"GO:0003723",
                        # "process/0.tsv":"GO:0016310",
                        # "process/1.tsv":"GO:0006412",
                        # "component/0.tsv":"GO:0005737",
                        # "component/1.tsv":"GO:0005634"}
        all_dataset = []
        for path in list(dataset_dict.keys()):
            tag = dataset_dict[path]
            dataset = load_dataset('csv', data_files=os.path.join(dataset_name,path),split = 'train')
            # dataset = dataset.train_test_split(test_size=0.1)

            if tokenizer.pad_token_id is None:   # 如果不存在pad字符，则使用eos字符替代
                tokenizer.pad_token_id = tokenizer.eos_token_id


            def preprocess_function(examples):
                max_length = 400
                batch_size = len(examples["Entry"])   # map函数按batch处理时的一个batch_size
                print(batch_size)
                inputs = [x for x in examples["Sequence"]]   # 构建输入形式 "ABCDEFG" 
                entry = [x for x in examples["Entry"]]
                
                tags = [[tag]]*batch_size
                model_inputs = tokenizer(inputs)   # 将输入tokenize为标签
                labels = model_inputs
                # 取得最后一个token的位置
                final_token_pos = torch.tensor([min(max_length, len(model_inputs["input_ids"][i]))  for i in range(len(model_inputs["input_ids"]))])
                
                # 该for循环处理每个样本
                for i in range(batch_size):
                    sample_input_ids = [tokenizer.eos_token_id] + model_inputs["input_ids"][i] + [tokenizer.eos_token_id]
                    label_input_ids = [tokenizer.eos_token_id] + labels["input_ids"][i] + [tokenizer.eos_token_id]
                    labels["input_ids"][i] = label_input_ids
                    model_inputs["input_ids"][i] = sample_input_ids
                    model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

                #该for循环进行padding操作
                for i in range(batch_size):
                    sample_input_ids = model_inputs["input_ids"][i]
                    label_input_ids = labels["input_ids"][i]
                    model_inputs["input_ids"][i] = sample_input_ids + [tokenizer.pad_token_id] * (      # 补齐序列，在每个序列前面加上pad字符至max_length
                        max_length - len(sample_input_ids)
                    )
                    model_inputs["attention_mask"][i] = model_inputs["attention_mask"][i] +  [0] * (max_length - len(sample_input_ids)) 
                    labels["input_ids"][i] = label_input_ids + [0] * (max_length - len(sample_input_ids))
                    model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])   # 取[:max_length]操作为处理输入比max_length长的情况
                    model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
                    labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
                model_inputs["labels"] = labels["input_ids"]
                model_inputs["sequence"] = inputs
                model_inputs["tags"] = tags
                model_inputs["final_pos"] = final_token_pos
                model_inputs["entry"] = entry
                return model_inputs


            # dataset
            processed_datasets = dataset.map(
                preprocess_function,
                batched=True,
                num_proc=1,
                batch_size = 500,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            print("type of processed_datasets: ", type(processed_datasets))
            # print(processed_datasets[:1])
            all_dataset.append(processed_datasets)
        
        # 假设 all_dataset 是一个包含多个 processed_datasets 的列表
        final_dataset = concatenate_datasets(all_dataset)
        return final_dataset
    
    print("No dataset")
    return None