import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

RANGE = 255

class DQNbn(torch.jit.ScriptModule):
    __constants__ = ['RANGE']

    def __init__(self, in_channels=4, n_actions=14):


        super(DQNbn, self).__init__()

        self.RANGE = RANGE

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            #nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    @torch.jit.script_method
    def forward(self, x):
        x = x.float() / self.RANGE
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DDQNbn(torch.jit.ScriptModule):
    def __init__(self, in_channels=4, n_actions=14):
        __constants__ = ['n_actions', 'r']

        super(DDQNbn, self).__init__()

        self.n_actions = n_actions
        self.r = 255.

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            #nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.fc_val = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    @torch.jit.script_method
    def forward(self, x):
        x = x.float() / self.r
        x = self.convs(x)
        x = x.view(x.size(0), -1)

        adv = self.fc_adv(x)
        val = self.fc_adv(x)

        return val + adv - adv.mean(1).unsqueeze(1)

    @torch.jit.script_method
    def value(self, x):
        x = x.float() / 255
        x = self.convs(x)
        x = x.view(x.size(0), -1)

        val = self.fc_adv(x)

        return val


class LanderDQN(torch.jit.ScriptModule):
    def __init__(self, n_state, n_actions, nhid=64):
        super(RamDQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_state, nhid),
            #nn.BatchNorm1d(nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid),
            #nn.BatchNorm1d(nhid),
            nn.ReLU(),
            nn.Linear(nhid, n_actions)
        )

        self.apply(init_weights)

    @torch.jit.script_method
    def forward(self, x):
        return self.layers(x)

class RamDQN(nn.Module):
    def __init__(self, n_state, n_actions):
        super(RamDQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_state, 256),
            #nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            #nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            #nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )


    @torch.jit.script_method
    def forward(self, x):
        return self.layers(x)
