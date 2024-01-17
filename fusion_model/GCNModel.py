import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseGCNConv


class RegGNN(nn.Module):
    '''Regression using a DenseGCNConv layer from pytorch geometric.

       Layers in this model are identical to GCNConv.
    '''

    def __init__(self, nfeat, nhid, nout, dropout):
        super(RegGNN, self).__init__()

        self.gc1 = DenseGCNConv(nfeat, nhid)
        self.gc2 = DenseGCNConv(nhid, nout)
        self.LinearLayer = nn.Linear(400, 1)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj,add_loop=False))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj,add_loop=False)
        x = self.LinearLayer(torch.transpose(x, 2, 1))
        return x.squeeze(-1)
    
class GCNModel(torch.nn.Module):
    def __init__(self, fusion_model):
        super(GCNModel, self).__init__()
        self.backbone = RegGNN(400, 400, 400, 0.5)
        self.fusion_model = fusion_model
    def forward(self, x_list):
        X0 = x_list[0]
        X1 = x_list[1]
        X2 = x_list[2]
        feat= torch.ones(X0.shape[0], X0.shape[1], 400).to(X0.device)
        X0=self.backbone(feat,X0)
        X1=self.backbone(feat,X1)
        X2=self.backbone(feat,X2)
        # 32x400
        y_pred = self.fusion_model([X0, X1, X2])
        
        return y_pred


if __name__ == '__main__':
    x = torch.randn(2,400,400).cuda()
    model= GCNModel(None).cuda()
    out=model([x,x,x])
    # print(out.shape)