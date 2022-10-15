import torch
import torch.nn as nn
import torch.nn.functional as F

class myTransformer1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(myTransformer1D, self).__init__()
        self.relu = nn.ReLU()
        self.li1_a = nn.Linear(input_dim, output_dim)
        self.li1_b = nn.Linear(input_dim, output_dim)
        self.li1_c = nn.Linear(input_dim, output_dim)
        self.li2_a = nn.Linear(output_dim, output_dim)
        self.li2_b = nn.Linear(output_dim, output_dim)
        self.li2_c = nn.Linear(output_dim, output_dim)
        self.li3 = nn.Linear(output_dim, output_dim)
        self.li4 = nn.Linear(output_dim, output_dim)
        
    def forward(self, x):
        a = self.relu(self.li1_a(x))
        a = self.relu(self.li2_a(a))
        b = self.relu(self.li1_b(x))
        b = self.relu(self.li2_b(b))
        c = self.relu(self.li1_c(x))
        c = self.relu(self.li2_c(c))
        x = a * b + c
        x = self.relu(self.li3(x))
        x = self.relu(self.li4(x))
        return x

class MBEPredictor(nn.Module):
    def __init__(self):
        super(MBEPredictor, self).__init__()
        self.Conv3D_G = nn.Sequential(
            nn.Conv3d(1, 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 1, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU()
        )
        self.myTransformer1D_G = nn.Sequential(
            myTransformer1D(686, 512),
            myTransformer1D(512, 512),
            myTransformer1D(512, 512)
        )
        self.linear = nn.Linear(512, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x1, x2):
        x1 = self.Conv3D_G(x1)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.Conv3D_G(x2)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.myTransformer1D_G(x)
        x = self.linear(x)
        x = self.relu(x)
        return x

def loadCO2Model():
    return torch.randn(1, 1, 111, 111, 111)
    
def demo():
    net = MBEPredictor()
    x1 = torch.randn(1, 1, 111, 111, 111)
    x2 = torch.randn(1, 1, 111, 111, 111)
    y = net(x1, x2)
    print(y)

if __name__ == '__main__':
    demo()