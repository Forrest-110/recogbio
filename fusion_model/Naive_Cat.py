import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(MLPBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(Encoder, self).__init__()
        self.mlp = MLPBlock(in_features, out_features, dropout=dropout)
        self.linear = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden_state = self.mlp(x)
        x = self.linear(hidden_state)
        x = self.dropout(x)
        x = self.relu(x)
        return x, hidden_state
    

class NaiveCatFusion(nn.Module):
    def __init__(self, in_modal_list, out_features, dropout=0.2):
        super(NaiveCatFusion, self).__init__()
        self.encoder_list = nn.ModuleList()
        for in_features in in_modal_list:
            self.encoder_list.append(Encoder(in_features, out_features//len(in_modal_list), dropout=dropout))
        self.out_features = out_features

    def forward(self, x_list):
        assert len(x_list) == len(self.encoder_list)
        out_feature_list = []
        for x, encoder in zip(x_list, self.encoder_list):
            out_feature, _ = encoder(x)
            out_feature_list.append(out_feature)
        out_feature = torch.cat(out_feature_list, dim=1)
        return out_feature
    

class RegressionHead(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(RegressionHead, self).__init__()
        self.mlps = nn.Sequential(
            MLPBlock(in_features, in_features//2),
            MLPBlock(in_features//2, in_features//4),
            MLPBlock(in_features//4, in_features//8),
            nn.Linear(in_features//8, out_features)
        )

    def forward(self, x):
        return self.mlps(x)
    
class NaiveCatModel(nn.Module):
    def __init__(self, in_modal_list, out_features, dropout=0.2,device='cuda:0'):
        super(NaiveCatModel, self).__init__()
        self.fusion = NaiveCatFusion(in_modal_list, out_features, dropout=dropout).to(device)
        self.regression = RegressionHead(out_features, out_features=1).to(device)
    def forward(self, x_list):
        out_feature = self.fusion(x_list)
        
        y_pred = self.regression(out_feature)

        return y_pred
    
if __name__ == '__main__':
    device = 'cuda'
    x_list = [torch.randn(32, 10).to(device), torch.randn(32, 20).to(device)]
    model = NaiveCatModel([10, 20], 6, dropout=0.2,device=device).cuda()
    y_pred = model(x_list)
    print(y_pred.shape)