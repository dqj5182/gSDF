import torch
import torch.nn as nn

class FCHead(nn.Module):
    def __init__(self, base_neurons=[512, 256, 128], out_dim=3):
        super().__init__()
        layers = []
        for (inp_neurons, out_neurons) in zip(base_neurons[:-1], base_neurons[1:]):
            layers.append(nn.Linear(inp_neurons, out_neurons))
            layers.append(nn.ReLU())
        self.final_layer = nn.Linear(out_neurons, out_dim)
        self.decoder = nn.Sequential(*layers)

    def forward(self, inp):
        decoded = self.decoder(inp)
        out = self.final_layer(decoded)

        return out


class ConvHead(nn.Module):
    def __init__(self, feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
        super().__init__()
        layers = []
        for i in range(len(feat_dims)-1):
            layers.append(nn.Conv2d(in_channels=feat_dims[i], out_channels=feat_dims[i + 1], kernel_size=kernel, stride=stride, padding=padding))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims) - 2 or (i == len(feat_dims) - 2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers) 

    def forward(self, inp):
        return self.layers(inp)