import torch
from torch import nn
import torch.nn.functional as F



class DQN(nn.Module):
    def __init__(self, input, output_dim):
        super().__init__()
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=input[0], out_channels=32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, input):
        return self.online(input)


class DuelingDDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDDQN, self).__init__()
        # Build a CNN layer to extract information
        self.CNN_layer = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=(8, 8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
        )

        # Build ANN layer
        self.fc1 = nn.Linear(3136, 512)
        # DUELING
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, action_dim)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def forward(self, state):
        t = self.CNN_layer(state)
        t = F.relu(self.fc1(t.reshape(t.shape[0], -1)))
        A = self.A(t)
        V = self.V(t).expand_as(A)
        Q = V + A - A.mean(1, keepdim=True).expand_as(A)
        return Q



