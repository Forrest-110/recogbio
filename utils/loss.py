import torch


def mae_loss(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def pearson_corr_loss(y_true, y_pred):
    y_true = y_true - torch.mean(y_true)
    y_pred = y_pred - torch.mean(y_pred)
    return -torch.sum(y_true * y_pred) / (torch.sqrt(torch.sum(y_true ** 2)) * torch.sqrt(torch.sum(y_pred ** 2)))

def regression_loss(y_true, y_pred, alpha=0.5):
    return  mae_loss(y_true, y_pred) +alpha * pearson_corr_loss(y_true, y_pred)

if __name__ == '__main__':
    y_true = torch.randn(100)
    y_pred = torch.randn(100)
    print(mae_loss(y_true, y_pred))
    print(pearson_corr_loss(y_true, y_pred))
    print(regression_loss(y_true, y_pred))