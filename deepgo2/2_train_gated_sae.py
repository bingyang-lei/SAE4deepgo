# 如果使用protgpt2激活，则需要显式设置-m protgpt2 -td test_gpt
# 如果使用SAE，则同样要设置模型-m esm_sae_gate -ont bp
# esm_sae_gate:coef=10
# esm_sae_gate2:coef=0.1,且选择模型的依据是总损失而不是分类损失
# esm_sae_gate3:coef=100,且decoder改成encoder_mag的转置,选择模型依据为分类损失
# esm_sae_gate4:coef=2000,其他和esm_sae_gate3一样
# esm_sae_gate5:coef=10000,其他和esm_sae_gate4一样

import time
import click as ck
import pandas as pd
import torch
import torch as th
import numpy as np
import random
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import copy
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from itertools import cycle
import math
from deepgo.torch_utils import FastTensorDataLoader
from deepgo.utils import Ontology, propagate_annots
from multiprocessing import Pool
from functools import partial
from deepgo.data import load_data, load_normal_forms
from deepgo.models import DeepGOModel, GateSAEModel, GateSAEModel2   #, AutoEncoderConfig
from deepgo.metrics import compute_roc
import wandb
import argparse
import os
from tqdm import tqdm
# fix the seed
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Training options")
    # 命令行参数配置
    parser.add_argument("--use_wandb", "-wandb", default=True, type=bool, help="是否启用Weights and Biases日志")
    parser.add_argument('--data-root', '-dr', default='data', help='数据文件夹路径')
    parser.add_argument('--ont', '-ont', default='mf', choices=['mf', 'bp', 'cc'], help='GO本体')
    parser.add_argument('--model-name', '-m', choices=['deepgozero_esm', 'deepgozero_esm_plus', 'esm_sae', 'esm_sae_gate4', 'gpt_sae_head36', 'protgpt2_average'], 
                        default='deepgozero_esm', help='模型名称')
    parser.add_argument('--model-id', '-mi', type=int, required=False)
    parser.add_argument('--test-data-name', '-td', default='test', choices=['test', 'nextprot', 'valid', 'test_gpt', 'test_gpt_mlp'], help='测试数据集名称')
    parser.add_argument('--batch-size', '-bs', default=-1, type=int, help='训练的batch大小')
    parser.add_argument('--epochs', '-ep', default=-1, type=int, help='训练的轮次')
    parser.add_argument("--learning_rate","-lr",default = -1, type = float, help="learning_rate")
    parser.add_argument('--load', '-ld', action='store_true', help='是否加载预训练模型')
    parser.add_argument('--device', '-d', default='cuda:0', help='训练设备')
    
    # 使用head做激活，要设置-td=test_gpt，使用mlp，要使用test_gpt_mlp

    
    # 解析参数并返回
    args = parser.parse_args()
    
    # 直接修改模型名：
    args.model_name = "gpt_sae_head_1_no_kaiming"
    if args.model_name.find('esm') != -1:
        args.d_mlp = 2560
    elif args.model_name.find('mlp') != -1:
        args.d_mlp = 5120
    elif args.model_name.find('head') != -1:
        args.d_mlp = 1280

    # 模型参数
    args.seed = 42
    args.pred_weight = 1
    args.aux_weight = 0
    args.rec_weight = 1000
    # 额外添加的参数量 
    args.extra_ratio = 1
     
    # args.d_mlp  = 1280 
    args.dtype = th.float32
    args.d_hidden = 21356
    
    # 训练参数 默认-1，不外部指定就在这里设置
    args.batch_size = 64 if args.batch_size == -1 else args.batch_size
    args.epochs = 150 if args.epochs == -1 else args.epochs
    args.learning_rate = 5e-4 if args.learning_rate==-1 else args.learning_rate
      
    # 输出参数 输出模型和结果单独放到result文件夹里
    if args.model_id is not None:
        args.model_name = f'{args.model_name}_{args.model_id}'
    output_dir = f"result/{args.ont}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.model_file = os.path.join(output_dir, f"{args.model_name}.th")
    args.out_file = os.path.join(output_dir, 'result/{args.ont}/{args.test_data_name}_predictions_{args.model_name}.pkl')
    args.performance_file = os.path.join(output_dir,f"{args.test_data_name}_predictions_{args.model_name}.pkl")
    
    args.use_wandb = True
    return args


def main(args):
    """
    This script is used to train DeepGO models
    """
    print('args.d_mlp:',args.d_mlp)

    if args.use_wandb:
        wandb.login()
        # wandb.init(project="", entity="")#your account
        wandb.init(project="protein_interpret_on_gpt2", entity = "xyl-nju-nanjing-university", name = args.model_name, config=vars(args),settings=wandb.Settings(init_timeout=120))#your account


    if args.model_name.find('plus') != -1:
        go_norm_file = f'{args.data_root}/go-plus.norm'
    else:
        go_norm_file = f'{args.data_root}/go.norm'
        
    go_file = f'./{args.data_root}/go.obo'
    terms_file = f'{args.data_root}/{args.ont}/terms.pkl'
    
    # Load Gene Ontology and Normalized axioms
    go = Ontology(go_file, with_rels=True)

    # Load the datasets
    # 在这里添加protgpt2的激活选项
    if args.model_name.find('esm') != -1:
        features_length = 2560
        features_column = 'esm2'
    elif args.model_name.find('mlp') != -1:
        features_length = 5120
        features_column = 'mlp'
    elif args.model_name.find('head') != -1:
        features_length = 1280
        features_column = 'head'

    test_data_file = f'{args.test_data_name}_data.pkl'
    iprs_dict, terms_dict, train_data, valid_data, test_data, test_df = load_data(
        args.data_root, args.ont, terms_file, features_length, features_column, test_data_file)
    
    print("data_root: ", args.data_root)
    print("ont: ", args.ont)
    print("terms_file: ", terms_file)
    print("features_length: ", features_length)
    print("features_column: ", features_column)
    print("test_data_file: ", test_data_file)
    n_terms = len(terms_dict)

    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data
    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()

    # todo 加入rule
    # nf1, nf2, nf3, nf4, relations, zero_classes = load_normal_forms(
    #     go_norm_file, terms_dict)
    # n_rels = len(relations)
    # n_zeros = len(zero_classes)

    # normal_forms = nf1, nf2, nf3, nf4
    # nf1 = th.LongTensor(nf1).to(args.device)
    # nf2 = th.LongTensor(nf2).to(args.device)
    # nf3 = th.LongTensor(nf3).to(args.device)
    # nf4 = th.LongTensor(nf4).to(args.device)
    # normal_forms = nf1, nf2, nf3, nf4
    n_zeros, n_rels=0,0
    
    # Create DataLoaders
    train_loader = FastTensorDataLoader(
        *train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=args.batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=args.batch_size, shuffle=False)
    
###################################################### training #########################################
   
    n_zeros, n=0,0
    #cfg = AutoEncoderConfig()
    net = GateSAEModel2(features_length, n_terms, n_zeros, n_rels, args.device, args).to(args.device)
    print(net)


    optimizer = th.optim.Adam(net.parameters(), lr=args.learning_rate)
    # scheduler = MultiStepLR(optimizer, milestones=[5, 20], gamma=0.1)

    best_loss = 10000.0
    if not args.load:
        print('Training the model')
        for epoch in tqdm(range(args.epochs)):

            net.train()
            train_pred_loss = 0
            train_reconstruction_loss = 0
            train_via_reconstruction_loss = 0
            train_steps = int(math.ceil(len(train_labels) / args.batch_size))
            # with ck.progressbar(length=train_steps, show_pos=True) as bar:
            #     for batch_features, batch_labels in train_loader:
            #         bar.update(1)
            #         batch_features = batch_features.to(device)
            #         batch_labels = batch_labels.to(device)
            #         reconstruction, reconstruction_loss, predition, via_reconstruction_loss \
            #         = net(batch_features,batch_labels)
            #         pred_loss = F.binary_cross_entropy(predition, batch_labels)
            #         # el_loss = net.el_loss(normal_forms)
            #         total_loss = cfg.l1_coeff*pred_loss + reconstruction_loss + via_reconstruction_loss
            #         train_pred_loss += pred_loss.detach().item()
            #         train_reconstruction_loss += reconstruction_loss.detach().item()
            #         train_via_reconstruction_loss += via_reconstruction_loss.detach().item()
            #         optimizer.zero_grad()
            #         total_loss.backward()
            #         optimizer.step()
            
            
            reconstruction_loss_grad = 0
            num_reconstruction = 0
            pred_loss_grad = 0
            num_pred = 0
            via_reconstruction_loss_grad = 0
            num_via = 0
            # with ck.progressbar(length=train_steps, show_pos=True) as bar:
            for batch_features, batch_labels in train_loader:
                # bar.update(1)
                batch_features = batch_features.to(args.device)
                batch_labels = batch_labels.to(args.device)

                # Forward pass
                reconstruction, reconstruction_loss, predition, via_reconstruction_loss \
                    = net(batch_features, batch_labels)
                # print("reconstruction: ",reconstruction[0])
                # print()
                # print("batch_features: ", batch_features[0])
                pred_loss = F.binary_cross_entropy(predition, batch_labels)
                total_loss = args.pred_weight * pred_loss + args.rec_weight * reconstruction_loss + args.aux_weight * via_reconstruction_loss
                # total_loss = reconstruction_loss
                # # # Zero gradients before backward pass
                # optimizer.zero_grad()

                # # Backward pass for pred_loss to calculate its gradient norm
                # pred_loss.backward(retain_graph=True)  # Backprop only pred_loss
                # for name, param in net.named_parameters():
                #     if param.grad is not None:
                #         num_pred+=1
                #         pred_loss_grad += param.grad.norm().item()  # Calculate L2 norm of gradient
                #         #print(f"Name: {name} | pred_loss gradient norm: {param.grad.norm().item()}")
                
                # optimizer.zero_grad()
                # # Backward pass for reconstruction_loss to calculate its gradient norm
                # reconstruction_loss.backward(retain_graph=True)  # Backprop only reconstruction_loss

                # for name, param in net.named_parameters():
                #     if param.grad is not None:
                #         num_reconstruction+=1
                #         reconstruction_loss_grad += param.grad.norm().item()  # Calculate L2 norm of gradient
                #         #print(f"Name: {name} | reconstruction_loss gradient norm: {param.grad.norm().item()}")
                
                # optimizer.zero_grad()
                        
                # # Backward pass for via_reconstruction_loss to calculate its gradient norm
                # via_reconstruction_loss.backward(retain_graph=True)  # Backprop only via_reconstruction_loss

                # for name, param in net.named_parameters():
                #     if param.grad is not None:
                #         num_via+=1
                #         via_reconstruction_loss_grad += param.grad.norm().item()  # Calculate L2 norm of gradient
                #         #print(f"Name: {name} | via_reconstruction_loss gradient norm: {param.grad.norm().item()}")
                
                # Perform the final backpropagation for the total loss
                optimizer.zero_grad()
                train_pred_loss += pred_loss.detach().item()
                train_reconstruction_loss += reconstruction_loss.detach().item()
                train_via_reconstruction_loss += via_reconstruction_loss.detach().item()
                
                total_loss.backward()  # Backprop the total loss
                optimizer.step()  # Update parameters  
                
            # reconstruction_loss_grad /= num_reconstruction
            # pred_loss_grad /= num_pred
            # via_reconstruction_loss_grad /= num_via
            print()
            # print(f"Pred loss gradient norm: {pred_loss_grad}")
            # print(f"Reconstruction loss gradient norm: {reconstruction_loss_grad}")
            # print(f"Via reconstruction loss gradient norm: {via_reconstruction_loss_grad}")
            # if args.use_wandb:
            #     wandb.log({
            #         "epoch": epoch,
            #         "pred_loss_grad_norm": pred_loss_grad,
            #         "reconstruction_loss_grad_norm": reconstruction_loss_grad,
            #         "via_reconstruction_loss_grad_norm": via_reconstruction_loss_grad,
            #     })
                

            
            train_pred_loss /= train_steps
            train_reconstruction_loss /= train_steps
            train_via_reconstruction_loss /= train_steps
            
            # ****Validation********
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / args.batch_size))
                valid_loss_recovered = 0
                valid_pred_loss = 0
                valid_reconstruction_loss = 0
                preds = []
                # with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                for batch_features, batch_labels in valid_loader:
                    #bar.update(1)
                    batch_features = batch_features.to(args.device)
                    batch_labels = batch_labels.to(args.device)
                    zero_tensor = th.zeros_like(batch_features).to(args.device)
                    reconstruction,reconstruction_loss,predition,via_reconstruction_loss = net(batch_features,batch_labels)
                    batch_loss = F.binary_cross_entropy(predition, batch_labels)
                    
                    valid_pred_loss += batch_loss.detach().item()
                    valid_reconstruction_loss += reconstruction_loss.detach().item()
                    preds = np.append(preds, predition.detach().cpu().numpy())
                # valid_loss_recovered /= valid_steps
                
                valid_pred_loss /= valid_steps
                valid_reconstruction_loss /= valid_steps

                
                # todo toc_auc计算太耗时，先注释掉
                # roc_auc = compute_roc(valid_labels, preds)
                roc_auc = 0

                print(f'''Epoch {epoch}: 
                    train_pred_Loss - {train_pred_loss}, 
                    train_reconstruction_loss: {train_reconstruction_loss},
                    train_via_reconstruction_loss: {train_via_reconstruction_loss},
                    
                    Valid_pred_Loss - {valid_pred_loss},
                    Valid_reconstruction_Loss - {valid_reconstruction_loss}, 
                    Valid_AUC - {roc_auc}''')
                
                if args.use_wandb:
                    wandb.log({
                        # "epoch": epoch,
                        "train_pred_loss": train_pred_loss ,
                        "train_reconstruction_loss": train_reconstruction_loss ,
                        "train_via_reconstruction_loss": train_via_reconstruction_loss ,
                        "valid_pred_loss": valid_pred_loss,
                        "valid_reconstruction_loss": valid_reconstruction_loss,
                        # "valid_auc": roc_auc,
                    })

                print('Test')
                test_steps = int(math.ceil(len(test_labels) / args.batch_size))
                
                test_pred_loss = 0
                test_reconstruction_loss = 0
                preds = []
                # with ck.progressbar(length=test_steps, show_pos=True) as bar:
                for batch_features, batch_labels in test_loader:
                    #bar.update(1)
                    batch_features = batch_features.to(args.device)
                    batch_labels = batch_labels.to(args.device)
                    zero_tensor = th.zeros_like(batch_features).to(args.device)
                    reconstruction,reconstruction_loss,predition,via_reconstruction_loss = net(batch_features,batch_labels)
                    batch_loss = F.binary_cross_entropy(predition, batch_labels)

                    
                    test_pred_loss += batch_loss.detach().item()
                    test_reconstruction_loss += reconstruction_loss.detach().item()
                    # test_loss += cfg.l1_coeff * batch_loss.detach().cpu().item() + reconstruction_loss.detach().cpu().item() + via_reconstruction_loss.detach().cpu().item()
                    preds.append(predition.detach().cpu().numpy())
                
                test_pred_loss /= test_steps
                test_reconstruction_loss /= test_steps
                preds = np.concatenate(preds)
                # roc_auc计算太耗时，先注释掉
                # roc_auc = compute_roc(test_labels, preds)
                roc_auc = 0
                print(f'''
                    Test pred Loss - {test_pred_loss}, 
                    Test reconstruction Loss - {test_reconstruction_loss},
                    Test AUC - {roc_auc}''')
                
                if args.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "test_pred_loss": test_pred_loss,
                        "test_reconstruction_loss": test_reconstruction_loss,
                        # "valid_auc": roc_auc,
                    })
                    

            # todo 保存模型先注释掉
            # if valid_pred_loss < best_loss:
            #     best_loss = valid_pred_loss
            #     print('Saving model')
            #     th.save(net.state_dict(), args.model_file)

            # scheduler.step()
            
    return 
    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(args.model_file))
    net.eval()

    with th.no_grad():
        valid_steps = int(math.ceil(len(valid_labels) / args.batch_size))
        valid_loss_recovered = 0
        valid_pred_loss = 0
        valid_reconstruction_loss = 0
        preds = []
        with ck.progressbar(length=valid_steps, show_pos=True) as bar:
            for batch_features, batch_labels in valid_loader:
                bar.update(1)
                batch_features = batch_features.to(args.device)
                batch_labels = batch_labels.to(args.device)
                zero_tensor = th.zeros_like(batch_features).to(args.device)
                reconstruction,reconstruction_loss,predition,via_reconstruction_loss = net(batch_features,batch_labels)
                batch_loss = F.binary_cross_entropy(predition, batch_labels)
                # 计算重建和原特征的交叉熵
                CE1 = F.cross_entropy(reconstruction, batch_features)
                CE2 = F.cross_entropy(zero_tensor, batch_features)
                valid_loss_recovered += 1-CE1/CE2
                valid_pred_loss += batch_loss.detach().item()
                valid_reconstruction_loss += reconstruction_loss.detach().item()
                # valid_loss += cfg.l1_coeff * batch_loss.detach().item() + reconstruction_loss.detach().item() + via_reconstruction_loss.detach().item()
                # preds = np.append(preds, predition.detach().cpu().numpy())
        valid_loss_recovered /= valid_steps
        valid_pred_loss /= valid_steps
        valid_reconstruction_loss /= valid_steps

    with th.no_grad():
        test_steps = int(math.ceil(len(test_labels) / args.batch_size))
        test_loss_recovered = 0
        test_pred_loss = 0
        test_reconstruction_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_features, batch_labels in test_loader:
                bar.update(1)
                batch_features = batch_features.to(args.device)
                batch_labels = batch_labels.to(args.device)
                zero_tensor = th.zeros_like(batch_features).to(args.device)
                reconstruction,reconstruction_loss,predition,via_reconstruction_loss = net(batch_features,batch_labels)
                batch_loss = F.binary_cross_entropy(predition, batch_labels)
                # 计算重建和原特征的交叉熵
                CE1 = F.cross_entropy(reconstruction, batch_features)
                CE2 = F.cross_entropy(zero_tensor, batch_features)
                test_loss_recovered += 1-CE1/CE2
                test_pred_loss += batch_loss.detach().item()
                test_reconstruction_loss += reconstruction_loss.detach().item()
                # test_loss += cfg.l1_coeff * batch_loss.detach().cpu().item() + reconstruction_loss.detach().cpu().item() + via_reconstruction_loss.detach().cpu().item()
                preds.append(predition.detach().cpu().numpy())
            test_loss_recovered /= test_steps
            test_pred_loss /= test_steps
            test_reconstruction_loss /= test_steps
        preds = np.concatenate(preds)
        roc_auc = compute_roc(test_labels, preds)
        print(f'''Valid pred Loss - {valid_pred_loss}, Valid loss recovered - {valid_loss_recovered},
              Valid reconstruction Loss - {valid_reconstruction_loss},
              Test pred Loss - {test_pred_loss}, Test loss recovered - {test_loss_recovered},
              Test reconstruction Loss - {test_reconstruction_loss},
              Test AUC - {roc_auc}''')

    # Save the performance into a file
    with open(args.performance_file, 'w') as f:
        f.write(f'''Valid pred Loss - {valid_pred_loss}, Valid loss recovered - {valid_loss_recovered},
              Valid reconstruction Loss - {valid_reconstruction_loss},
              Test pred Loss - {test_pred_loss}, Test loss recovered - {test_loss_recovered},
              Test reconstruction Loss - {test_reconstruction_loss},
              Test AUC - {roc_auc}''')
    # return
    preds = list(preds)
    # Propagate scores using ontology structure
    with Pool(32) as p:
        preds = p.map(partial(propagate_annots, go=go, terms_dict=terms_dict), preds)
    test_df['preds'] = preds
    test_df.to_pickle(args.out_file)

if __name__ == '__main__':
    args=parse_args()
    main(args)
