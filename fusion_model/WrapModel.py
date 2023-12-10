import torch
from torchvision import models

class WrapModel(torch.nn.Module):
    def __init__(self, fusion_model, cnn_form):
        super(WrapModel, self).__init__()
        self.cnn_form=cnn_form
        if cnn_form:
            self.backbone = models.resnet18(pretrained=True)
        self.fusion_model = fusion_model
    def forward(self, x_list):
        X0 = x_list[0]
        X1 = x_list[1]
        X2 = x_list[2]
        if self.cnn_form:
            X0=self.backbone(X0.reshape(-1,3,152,175))
            X1=self.backbone(X1.reshape(-1,3,152,175))
            X2=self.backbone(X2.reshape(-1,3,152,175))
        y_pred = self.fusion_model([X0, X1, X2])
        
        return y_pred
    

if __name__ == "__main__":
    X=torch.randn(2,3,152,175)
    model=models.resnet18(pretrained=True)
    print(model(X).shape)