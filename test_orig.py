import torch
from utils.loss import regression_loss
from dataset.data import HCPOrigDataset
from tqdm import tqdm
from fusion_model.AttentionFusion import AttentionFusionModel
from fusion_model.Naive_Cat import NaiveCatModel
from fusion_model.TransformerFusion import TransformerFusionModel
from fusion_model.WrapModel import WrapModel
#logger
import logging
import os
import sys
import time
import numpy as np
import ast


def test(model, test_loader, device):
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(test_loader)):
            X0 = sample['X0'].to(device).to(torch.float32)
            X1 = sample['X1'].to(device).to(torch.float32)
            X2 = sample['X2'].to(device).to(torch.float32)

            
            y_pred = model([X0, X1, X2])
            y_pred_list.append(y_pred.cpu().numpy())

    return y_pred_list



def pearson_corr(preds, y_test):
    # preds: N x 1
    # y_test: N x 1
    preds = preds - np.mean(preds)
    y_test = y_test - np.mean(y_test)
    corr = np.sum(preds * y_test) / (np.sqrt(np.sum(preds ** 2)) * np.sqrt(np.sum(y_test ** 2)))
    return corr

def evaluate(preds, y_test):
  
    # preds: N x 1
    # y_test: N x 1
    
    mae_loss= np.mean(np.abs(preds - y_test))
    print("MAE Loss: ", mae_loss)
    # pearson correlation
    corr = pearson_corr(preds, y_test)

    print("Corr: ", corr)

    return mae_loss, corr


def main():
    exp_path="exps/2023-12-10-21-33-55"
    with open(os.path.join(exp_path, 'config.txt'), 'r') as f:
        Config = ast.literal_eval(f.read())
    test_dataset = HCPOrigDataset(Config['data_root'], Config['task_id'], mode='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=Config['batch_size'], shuffle=False, num_workers=8)
    device = torch.device(Config['device'])
    if Config['model'] == 'AttentionFusion':
        model = AttentionFusionModel([Config['input_dim'], Config['input_dim'], Config['input_dim']], Config['input_dim']*3, dropout=Config['dropout'], num_heads=Config['num_heads'],device=Config['device'])
    elif Config['model'] == 'NaiveCat':
        model = NaiveCatModel([Config['input_dim'], Config['input_dim'], Config['input_dim']], Config['input_dim']*3, dropout=Config['dropout'],device=Config['device'])
    elif Config['model'] == 'TransformerFusion':
        model = TransformerFusionModel([Config['input_dim'], Config['input_dim'], Config['input_dim']], Config['input_dim']*3, Config['num_heads'], dropout=Config['dropout'],device=Config['device'])
    else:
        raise NotImplementedError('Model not implemented')
    
    model = WrapModel(model, cnn_form=Config['cnn_form']).to(Config['device'])
    
    model.load_state_dict(torch.load(os.path.join(exp_path, 'best_model.pth')))
    model.to(device)
    y_pred_list = test(model, test_loader, device)
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_test = test_dataset.y.reshape(-1,1)

    print("y_pred: ", y_pred)
    print("y_test: ", y_test)

    evaluate(y_pred, y_test)

if __name__ == '__main__':
    main()

