import torch
import torch.nn as nn
import torch.nn.functional as F

class wordEmbedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super(wordEmbedder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.embedding(x)

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

class ProteinFold3D(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super(ProteinFold3D, self).__init__()
        self.wordEmbedder = wordEmbedder(vocab_size, embedding_dim, padding_idx)
        self.Conv1D_G = nn.Sequential(
            nn.Conv1d(embedding_dim, int(embedding_dim/2), 5, 2),
            nn.BatchNorm1d(int(embedding_dim/2)),
            nn.ReLU(),
            nn.Conv1d(int(embedding_dim/2), int(embedding_dim/2), 4, 3),
            nn.BatchNorm1d(int(embedding_dim/2)),
            nn.ReLU(),
            nn.Conv1d(int(embedding_dim/2), int(embedding_dim/2), 3, 4),
            nn.BatchNorm1d(int(embedding_dim/2)),
            nn.ReLU()
        )
        self.myTransformer1D_G = nn.Sequential(
            myTransformer1D(768, 512),
            myTransformer1D(512, 512),
            myTransformer1D(512, 512)
        )
        self.TranConv3D_G = nn.Sequential(
            nn.ConvTranspose3d(1, 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 8, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.Conv3D = nn.Conv3d(8, 1, (3, 3, 3), stride=(1, 1, 1))

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x)
            x = x.view(1, -1)
        x = self.wordEmbedder(x)
        x = x.permute(0, 2, 1) #swap data to make conv1d work
        x = self.Conv1D_G(x)
        x = x.view(x.size(0), -1)
        x = self.myTransformer1D_G(x)
        x = x.view(x.size(0), 1, 8, 8, 8)
        x = self.TranConv3D_G(x)
        x = self.Conv3D(x)
        return x

def demo():
    model = ProteinFold3D(20, 64, 0)
    x = torch.randint(0, 20, (1, 576))
    y = model(x)
    print(y.shape)

if __name__ == '__main__':
    demo()