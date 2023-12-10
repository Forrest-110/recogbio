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
    

class TransformerFusion(nn.Module):
    def __init__(self, in_modal_list, out_features, nhead, dropout=0.2):
        super(TransformerFusion, self).__init__()
        self.encoder_list = nn.ModuleList()
        for in_features in in_modal_list:
            self.encoder_list.append(Encoder(in_features, out_features//len(in_modal_list), dropout=dropout))
        self.out_features = out_features

        self.attention = nn.TransformerEncoderLayer(
            d_model=out_features,
            nhead=nhead,
            dropout=dropout
        )

    def forward(self, x_list):
        assert len(x_list) == len(self.encoder_list)
        out_feature_list = []
        for x, encoder in zip(x_list, self.encoder_list):
            out_feature, hidden_state = encoder(x)
            out_feature_list.append(out_feature)
        attention_out= self.attention(torch.cat(
            out_feature_list,
        dim=1).unsqueeze(1))

        return attention_out.squeeze(1)
    

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
    
class TransformerFusionModel(nn.Module):
    def __init__(self, in_modal_list, out_features, nhead, dropout=0.2,device='cuda:0'):
        super(TransformerFusionModel, self).__init__()
        self.fusion = TransformerFusion(in_modal_list, out_features, nhead, dropout).to(device)
        self.regression_head = RegressionHead(out_features, 1).to(device)

    def forward(self, x_list):
        out_feature = self.fusion(x_list)
        out = self.regression_head(out_feature)
        return out
    

if __name__ == "__main__":
    device = "cuda"
    model = TransformerFusionModel([10, 10, 10], 12, 4).cuda()
    x_list = [torch.randn(5, 10).cuda(), torch.randn(5, 10).cuda(), torch.randn(5, 10).cuda()]
    out = model(x_list)
    print(out.shape)
    print(model)