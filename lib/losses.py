import torch


def bce_loss(out: torch.Tensor, labels: torch.Tensor):
    return torch.mean(out - out * labels + torch.log(1 + torch.exp(-out)))

def focal_loss(y_pred : torch.Tensor, y_true : torch.Tensor, gamma = 2, eps = 8e-10):
    return -(((1 - y_pred)**gamma)*y_true*torch.log(y_pred) + (1-y_true)*torch.log(1-y_pred)).sum()

def dice_loss(y_pred : torch.Tensor,y_true : torch.Tensor):
    ins = 2 * y_pred * y_true
    union = y_pred + y_true + 8e-10
    return 1 - (ins / union).mean()

def get_tversky(alpha):
    def tversky_loss(y_pred : torch.Tensor, y_true : torch.Tensor, alpha=alpha):
        A = (y_pred*y_true).sum()
        B = (y_true*(1-y_pred)).sum()
        C = ((1-y_true)*y_pred).sum()
        return -A / (A + alpha*B + (1-alpha)*C + 1e-8)
    return tversky_loss

def iou_pytorch(output: torch.Tensor, labels : torch.Tensor):
    output = (output.squeeze(1) > 0.5).byte()
    labels = labels.squeeze(1).byte()
    SMOOTH = 1e-8
    intersection = (output & labels).float().sum((1,2))
    union = (output | labels).float().sum((1,2))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou