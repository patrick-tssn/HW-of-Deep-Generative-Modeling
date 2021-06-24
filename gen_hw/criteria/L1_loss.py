import torch
from torch import nn

class L1_Loss(nn.Module):
    def __init__(self, opts):
        super(L1_Loss, self).__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.opts = opts

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        return x

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            loss += torch.norm(y_hat_feats[i]-y_feats[i], p=1)
            count += 1

        return loss / count
