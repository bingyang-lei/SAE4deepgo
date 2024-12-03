# Pyvene method of getting activations
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch
from load_datasets import get_dataset, custom_collate_fn
from tqdm import tqdm
import numpy as np
import pickle
import sys
sys.path.append('../')
from torch.utils.data import DataLoader
from datasets import Dataset, concatenate_datasets
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import default_data_collator, get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_constant_schedule

# Specific pyvene imports
from utils import  get_protgpt2_activations_bau
# from interveners import wrapper, Collector, ITI_Intervener
# import pyvene as pv
import random

seed = 42
random.seed(seed) 
os.environ['PYTHONHASHSEED'] = str(seed) 
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) 
torch.backends.cudnn.deterministic = True

HF_NAMES = {
    # 'llama_7B': 'baffo32/decapoda-research-llama-7B-hf',
    # 'llama_7B': 'huggyllama/llama-7b',
    # 'alpaca_7B': 'circulus/alpaca-7b', 
    # 'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    # 'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    # 'llama2_chat_13B': 'meta-llama/Llama-2-13b-chat-hf', 
    # 'llama2_chat_70B': 'meta-llama/Llama-2-70b-chat-hf', 
    # 'llama3_8B': 'meta-llama/Meta-Llama-3-8B',
    # 'llama3_8B_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    # 'llama3_70B': 'meta-llama/Meta-Llama-3-70B',
    # 'llama3_70B_instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
    
    # 目前用的蛋白质模型
    "protgpt2": "/mnt/data1/xyliu/Pre_Train_Model/ProtGPT2",
}

def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='protgpt2')
    parser.add_argument('--model_prefix', type=str, default='', help='prefix of model name')
    parser.add_argument('--dataset_name', type=str, default='ctrlprot_dataset')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
    # loading model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map="auto")
    device = "cuda"

 
    if args.dataset_name == "ctrlprot_dataset":
        origin_dataset = get_dataset(args.dataset_name,tokenizer)
    
    batch_size = 64  
    dataloader = DataLoader(
        origin_dataset, shuffle=True, collate_fn=custom_collate_fn, batch_size=batch_size, pin_memory=True
    )
    
        
    all_layer_wise_activations = []
    all_head_wise_activations = []
    all_mlp_wise_activations = []
    final_dataset = [] 
    print("Getting activations")
    
    # [entry, sequence, tags, head_activation, mlp_activation]
    # final_dataset = Dataset.from_dict({
    # "entry": [],
    # "sequence": [],
    # "tags": [],
    # "head_activation": [],
    # "mlp_activation": [],
    # })
    for step, batch in enumerate(tqdm(dataloader)):
        print("type of batch:", type(batch))
        print("length of batch:", len(batch))
        a = 0
        for k, v in batch.items():
            a += 1
            print(k, v)
            if a==1:
                break
        break
    for step, batch in enumerate(tqdm(dataloader)):
        # test_batch = {k: v.to(device) for k, v in batch.items()}
        # outputs = model(**test_batch)
        
        layer_wise_activations, head_wise_activations, mlp_wise_activations = get_protgpt2_activations_bau(model, batch, device)
        
        # all_layer_wise_activations.append(layer_wise_activations.copy())
        for i in range(len(batch["sequence"])):
            final_dataset.append([batch["entry"][i],batch["sequence"][i], batch["tags"][i],
                                  head_wise_activations[i], mlp_wise_activations[i]])
        # if step>=10:
        #     break

    # 设置每个批次的大小
    concat_batch_size = 2048  # 你可以根据需要调整这个值
    total_batches = len(final_dataset) // concat_batch_size + (1 if len(final_dataset) % concat_batch_size > 0 else 0)
    final_datasets = []  # 用于存储所有的小批次数据集
    for batch_index in range(total_batches):
        start_index = batch_index * concat_batch_size
        end_index = start_index + concat_batch_size
        current_batch = final_dataset[start_index:end_index]
        
        dataset = Dataset.from_dict({
            "entry": [item[0] for item in current_batch],
            "sequence": [item[1] for item in current_batch],
            # "tags": [item[2] for item in current_batch],
            "head_activation": [item[3] for item in current_batch],
            "mlp_activation": [item[4] for item in current_batch],
        }) 

        final_datasets.append(dataset)

    # 合并所有的小数据集
    final_dataset_combined = concatenate_datasets(final_datasets)

    # 保存为缓存文件
    final_dataset_combined.save_to_disk(f"./dataset/{args.model_name}_{args.dataset_name}")


if __name__ == '__main__':
    main()