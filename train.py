import torch
from utils.loss import regression_loss
from dataset.data import HCPDataset
from tqdm import tqdm
from fusion_model.AttentionFusion import AttentionFusionModel
from fusion_model.Naive_Cat import NaiveCatModel
from fusion_model.TransformerFusion import TransformerFusionModel
#logger
import logging
import os
import sys
import time



def train_step(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss=0.
    for batch_idx, sample in enumerate(tqdm(train_loader)):
        X0 = sample['X0'].to(device).to(torch.float32)
        X1 = sample['X1'].to(device).to(torch.float32)
        X2 = sample['X2'].to(device).to(torch.float32)
        y = sample['y'].to(device).to(torch.float32)
        
        optimizer.zero_grad()
        y_pred = model([X0, X1, X2])
        loss = criterion(y, y_pred)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    train_loss /= len(train_loader)
    return train_loss

def eval_step(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(val_loader)):
            X0 = sample['X0'].to(device).to(torch.float32)
            X1 = sample['X1'].to(device).to(torch.float32)
            X2 = sample['X2'].to(device).to(torch.float32)
            y = sample['y'].to(device).to(torch.float32)
            
            y_pred = model([X0, X1, X2])
            loss = criterion(y, y_pred)
            
            val_loss += loss.item()
            
    val_loss /= len(val_loader)
    return val_loss

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs, save_model_path):
    train_loss_list = []
    val_loss_list = []
    min_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_step(model, train_loader, optimizer, criterion, device)
        val_loss = eval_step(model, val_loader, criterion, device)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        print('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch, train_loss, val_loss))
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_model_path, 'best_model.pth'))
    return train_loss_list, val_loss_list


def config():
    Config=dict()
    Config['data_root'] = 'dataset/'
    Config['dim_reduce'] = 'ivis'
    Config['task_id'] = 0
    Config['mode'] = 'train'
    Config['batch_size'] = 32
    Config['epochs'] = 500
    Config['lr'] = 1e-4
    Config['weight_decay'] = 1e-4
    Config['dropout'] = 0.2
    Config['num_heads'] = 4
    Config['device'] = 'cuda:0'
    Config['model'] = 'TransformerFusion'
    Config['input_dim'] = 128
    Config['alpha'] = 0.
    Config['output_dir'] = 'exps'
    return Config

def experiment(Config):

    train_dataset = HCPDataset(Config['data_root'], Config['task_id'],Config['dim_reduce'], mode='train')
    val_dataset = HCPDataset(Config['data_root'], Config['task_id'],Config['dim_reduce'], mode='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Config['batch_size'], shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=Config['batch_size'], shuffle=False, num_workers=4)
    if Config['model'] == 'AttentionFusion':
        model = AttentionFusionModel([Config['input_dim'], Config['input_dim'], Config['input_dim']], Config['input_dim']*3, dropout=Config['dropout'], num_heads=Config['num_heads'],device=Config['device'])
    elif Config['model'] == 'NaiveCat':
        model = NaiveCatModel([Config['input_dim'], Config['input_dim'], Config['input_dim']], Config['input_dim']*3, dropout=Config['dropout'],device=Config['device'])
    elif Config['model'] == 'TransformerFusion':
        model = TransformerFusionModel([Config['input_dim'], Config['input_dim'], Config['input_dim']], Config['input_dim']*3, Config['num_heads'], dropout=Config['dropout'],device=Config['device'])
    else:
        raise NotImplementedError('Model not implemented')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=Config['lr'], weight_decay=Config['weight_decay'])
    
    def criterion(y_true, y_pred):
        return regression_loss(y_true, y_pred, alpha=Config['alpha'])
    
    # save dirname= timestamp
    dirname = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # create dir
    os.makedirs(os.path.join(Config['output_dir'], dirname))
    
    train_loss_list, val_loss_list = train(model, train_loader, val_loader, optimizer, criterion, Config['device'], epochs=Config['epochs'], save_model_path=os.path.join(Config['output_dir'], dirname))
    
    
    # save model
    torch.save(model.state_dict(), os.path.join(Config['output_dir'], dirname, 'final_model.pth'))
    # save config
    with open(os.path.join(Config['output_dir'], dirname, 'config.txt'), 'w') as f:
        f.write(str(Config))
    # save log
    logging.basicConfig(filename=os.path.join(Config['output_dir'], dirname, 'log.txt'), level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Train Loss: {}'.format(train_loss_list))
    logger.info('Val Loss: {}'.format(val_loss_list))



    return train_loss_list, val_loss_list

def main():
    Config = config()
    train_loss_list, val_loss_list = experiment(Config)
    print('Train Loss: ', train_loss_list)
    print('Val Loss: ', val_loss_list)

if __name__ == '__main__':
    main()
