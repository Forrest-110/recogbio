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
    

class AttentionFusion(nn.Module):
    def __init__(self, in_modal_list, out_features, dropout=0.2, num_heads=4):
        super(AttentionFusion, self).__init__()
        self.encoder_list = nn.ModuleList()
        for in_features in in_modal_list:
            self.encoder_list.append(Encoder(in_features, out_features//len(in_modal_list), dropout=dropout))
        self.out_features = out_features

        # multi-head attention
        self.num_heads = num_heads
        self.attention_list =nn.ModuleList()
        for i in range(len(in_modal_list)):
            self.attention_list.append(nn.MultiheadAttention(out_features//len(in_modal_list), num_heads=num_heads))

    def forward(self, x_list):
        assert len(x_list) == len(self.encoder_list)
        out_feature_list = []
        hidden_state_list = []
        for x, encoder in zip(x_list, self.encoder_list):
            out_feature, hidden_state = encoder(x)
            out_feature_list.append(out_feature)
            hidden_state_list.append(hidden_state)
        
        # multi-head attention
        attention_out_list = []
        for i in range(len(x_list)):
            # every modal attends to all other modals
            attention_list=[]
            for j in range(len(x_list)):
                if i != j:
                    attention_out, _ = self.attention_list[i](hidden_state_list[i], hidden_state_list[j], hidden_state_list[j])
                    attention_list.append(attention_out)
            attention_out = torch.cat(attention_list, dim=1)
            attention_out_list.append(attention_out)
        
        # concat
        out_list=[]
        for i in range(len(x_list)):
            out = torch.cat([out_feature_list[i], attention_out_list[i]], dim=1)
            out_list.append(out)

        out_feature = torch.cat(out_list, dim=1)

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
    

class AttentionFusionModel(nn.Module):
    def __init__(self, in_modal_list, out_features, dropout=0.2, num_heads=4,device='cuda:0'):
        super(AttentionFusionModel, self).__init__()
        self.fusion = AttentionFusion(in_modal_list, out_features, dropout=dropout, num_heads=num_heads).to(device)
        self.regression_head = RegressionHead(out_features*len(in_modal_list),1).to(device)
    def forward(self, x_list):
        out_feature = self.fusion(x_list)
        out = self.regression_head(out_feature)
        return out

if __name__ == "__main__":
    model = AttentionFusionModel([2048, 2048, 2048], 96, dropout=0.2, num_heads=4)
    print(model)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    x_list = [torch.randn(5, 2048).to("cuda"), torch.randn(5, 2048).to("cuda"), torch.randn(5, 2048).to("cuda")]
    out = model(x_list)
    print(out.shape)
    print(out)


    

