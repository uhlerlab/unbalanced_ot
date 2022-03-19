import torch
from torch import nn
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, dim=500, nz=500):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.nz = nz
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.nz),
            nn.ReLU(),
            nn.Linear(self.nz, self.nz),
            nn.ReLU(),
            nn.Linear(self.nz, self.nz),
            nn.ReLU(),
            nn.Linear(self.nz, 1),
            )

    def forward(self, x):
        x = x.view(-1, self.nz)
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, nz, hidden):
        super(Generator, self).__init__()
        self.nz = nz
        self.net = nn.Sequential(
            nn.Linear(self.nz, self.nz),
            nn.ReLU(),
            nn.Linear(self.nz, self.nz),
            nn.ReLU(),
            nn.Linear(self.nz, self.nz),
            nn.ReLU(),
            nn.Linear(self.nz, self.nz),
            )
        self.scale = nn.Sequential(
            nn.Linear(self.nz, self.nz),
            nn.ReLU(),
            nn.Linear(self.nz, 1),
            nn.Softplus(),
            )

    def forward(self, x):
        #z = self.net(x)
        #gates, dz, scale = self.gates(z), self.dz(z), self.scale(z)
        #gates = self.sigmoid(gates)
        return self.net(x), self.scale(x)


