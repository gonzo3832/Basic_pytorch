
# coding: utf-8
# # Overview
# hydraによるパラメータ管理を行う　　
# hydraのマルチラン機能を使うためには、instanceを動的に定義する必要がある。  

import random
import numpy as np
import os
import torch
import hydra
import mlflow
import pandas as pd
import sklearn.model_selection as sms
from PIL import Image
from src.dataset import OriginalDataset
from src.model  import get_model
from src.utils import set_seed,RMSELoss
import torch.nn as nn
from torch.utils.data import DataLoader
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

config_path = "./config"
config_name = "run_config.yaml"

@hydra.main(config_path=config_path, config_name=config_name)
def run(cfg):
    set_seed()
    cwd = hydra.utils.get_original_cwd()
    train_data_dir = os.path.join(cwd,'inputs/MNIST_small/train')
    df = pd.read_csv(os.path.join(train_data_dir,'train.csv'))
    train_df,valid_df = sms.train_test_split(df,test_size = 0.2,stratify=df['y'],random_state=42)
    test_data_dir = os.path.join(cwd,'inputs/MNIST_small/test')
    test_df = pd.read_csv(os.path.join(test_data_dir,'test.csv'))
    
    # datasetの動的なinstance化
    train_dataset = OriginalDataset(train_data_dir,train_df,aug = cfg.dataset.aug)
    valid_dataset = OriginalDataset(train_data_dir,valid_df)
    test_dataset = OriginalDataset(test_data_dir,test_df)
    
    train_loader = DataLoader(train_dataset,batch_size = 4,shuffle = True)
    valid_loader = DataLoader(train_dataset,batch_size = 4,shuffle = False)
    test_loader =DataLoader(train_dataset,batch_size = 4,shuffle = False)
    
    # modelの動的なinstance化
    model = get_model(cfg.model.name)
    print(model)
    
    loss_func = RMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    train_losses = []
    valid_losses = []
    
    mlflow.set_tracking_uri('file:'+os.path.join(cwd,'mlruns'))

    # ------ 追加code ①: start runでlogging開始----

    tags = {MLFLOW_RUN_NAME:f'model_{cfg.model.name}-aug_{cfg.dataset.aug}'}
    mlflow.start_run( tags = tags)
    # -------------------------------------------------------
    
    for i in range(epochs):
        train_loss = 0
        valid_loss = 0
        valid_pred = []
        valid_true = []

        
        for X_train, y_train in train_loader:
        
            y_pred = model(X_train)  # we don't flatten X-train here
            batch_loss = loss_func(y_pred, y_train)
     
            
            # Update parameters
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            train_loss += batch_loss.item()
            
        train_loss /= len(train_loader)
            
        train_losses.append(train_loss)
        
        # ------追加code②-1: logしたい項目に対してlog_metric--------------
        mlflow.log_metric("train loss", train_loss, step = i)
        # -------------------------------------------------------

        for X_val, y_val in valid_loader:
            with torch.no_grad():
                y_pred = model(X_val)
            valid_pred.append(y_pred)
            valid_true.append(y_val)
                
            batch_loss = loss_func(y_pred, y_val)
            valid_loss += batch_loss.item()

        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)
        
        # ------追加code②-2: logしたい項目に対してlog_metric--------------
        mlflow.log_metric("valid loss", valid_loss, step = i)
        # -------------------------------------------------------
        
        print(f'epoch: {i:2}   train_loss: {train_loss:10.8f}  valid_loss: {valid_loss:10.8f}  ')
        print(f'valid_pred : {valid_pred[0]} valid_true {valid_true[0]}')
        print('-----------------------------')


    test_loss=0
    for X_test,y_test in test_loader:
        with torch.no_grad():
                y_pred = model(X_test)
                
        batch_loss = loss_func(y_pred, y_test)
        test_loss += batch_loss.item()

    test_loss /= len(test_loader)

# ------追加code②-3: logしたい項目に対してlog_metric--------------
    mlflow.log_metric("test loss", test_loss)
# -------------------------------------------------------

    print(f'test loss : {test_loss:.4f}')

# ------追加code③: 終了を明示する--------------
    mlflow.end_run()
# -------------------------------------------------------

if __name__ == "__main__":
    run()
