from torch import nn
from torch import cat
import torch.nn.functional as F

from src.accent_classifiers.layers.linear import LinearLayer
from src.accent_classifiers.layers.reverse_grad import grad_reverse
from src.accent_classifiers.layers.tdnn import TDNNLayer

class AC(nn.Module):
    """Fully-connected neural network to classify accents."""

    def __init__(self, input_size, n_accents, dropout, mode, standard, alpha):
        """Dropout can be null. `standard` is the standard accent for OneWayDAT. Other
        accents are reversed; `standard` is not. DAT mode ignores this `standard` arg
        and reverse all gradients."""
        super().__init__()
        self.mode = mode
        self.standard = standard
        self.alpha = alpha
        self.fc1 = LinearLayer(input_size, 1024, F.relu, dropout)
        self.fc2 = LinearLayer(1024, 256, F.relu, dropout)
        self.fc3 = LinearLayer(256, n_accents, None, dropout)

    def forward(self, x, y):
        if self.mode == "DAT":
            x = grad_reverse(x, self.alpha)
        elif self.mode == "OneWayDAT":
            if y != self.standard:
                x = grad_reverse(x, self.alpha)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


class X_Vector(nn.Module):
    """x_vector from https://ieeexplore.ieee.org/document/8461375 with changed input/output parameters and embeddings can be extracted after seg6"""
    def __init__(self, input_size, n_accents, dropout, mode, standard, alpha):
        """Dropout can be null. `standard` is the standard accent for OneWayDAT. Other
        accents are reversed; `standard` is not. DAT mode ignores this `standard` arg
        and reverse all gradients."""
        super().__init__()
        self.mode = mode
        self.standard = standard
        self.alpha = alpha
        self.frame1 = TDNNLayer(input_size, 512, F.relu, dropout,5,1)
        self.frame2 = TDNNLayer(512, 512, F.relu, dropout,3,2)
        self.frame3 = TDNNLayer(512, 512, F.relu, dropout,3,3)
        self.frame4 = TDNNLayer(512, 512, F.relu, dropout,1,1)
        self.frame5 = TDNNLayer(512, 1500, F.relu, dropout,1,1,)
        """Stats pooling compute mean and stadard deviation"""
        self.seg6 = LinearLayer(3000,512,F.relu,dropout)
        self.seg7 = LinearLayer(512,512,F.relu,dropout)
        self.out = LinearLayer(512,n_accents,None,dropout)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x, y):
        if self.mode == "DAT":
            x = grad_reverse(x, self.alpha)
        elif self.mode == "OneWayDAT":
            if y != self.standard:
                x = grad_reverse(x, self.alpha)

        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)
        """Stats pooling compute mean, stadard deviation and concat them"""
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        x = cat((mean,std),1)
        embeddings = self.seg6(x)
        x = self.seg7(embeddings)
        x = self.out(x)
        return self.softmax(x)