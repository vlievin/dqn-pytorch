import math

import torch
import torch.nn as nn

SQRT2 = math.sqrt(2.0)

ACT = nn.ReLU


class DQN(torch.jit.ScriptModule):
    def __init__(self, in_channels=4, n_actions=14):
        super(DQN, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            ACT(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            ACT(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            ACT()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            ACT(),
            nn.Linear(512, n_actions)
        )

    @torch.jit.script_method
    def forward(self, x):
        x = x.float() / 255
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DDQN(torch.jit.ScriptModule):
    def __init__(self, in_channels=4, n_actions=14):
        __constants__ = ['n_actions']

        super(DDQN, self).__init__()

        self.n_actions = n_actions

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            ACT(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            ACT(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            ACT()
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            ACT(),
            nn.Linear(512, n_actions)
        )

        self.fc_val = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            ACT(),
            nn.Linear(512, 1)
        )

        def scale_grads_hook(module, grad_out, grad_in):
            """scale gradient by 1/sqrt(2) as in the original paper"""
            grad_out = tuple(map(lambda g: g / SQRT2, grad_out))
            return grad_out

        self.fc_adv.register_backward_hook(scale_grads_hook)
        self.fc_val.register_backward_hook(scale_grads_hook)

    @torch.jit.script_method
    def forward(self, x):
        x = x.float() / 255
        x = self.convs(x)
        x = x.view(x.size(0), -1)

        adv = self.fc_adv(x)
        val = self.fc_val(x)

        return val + adv - adv.mean(1).unsqueeze(1)

    @torch.jit.script_method
    def value(self, x):
        x = x.float() / 255
        x = self.convs(x)
        x = x.view(x.size(0), -1)

        return self.fc_val(x)


class LanderDQN(torch.jit.ScriptModule):
    def __init__(self, n_state, n_actions, nhid=64):
        super(LanderDQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_state, nhid),
            ACT(),
            nn.Linear(nhid, nhid),
            ACT(),
            nn.Linear(nhid, n_actions)
        )

    @torch.jit.script_method
    def forward(self, x):
        x = self.layers(x)
        return x


class RamDQN(torch.jit.ScriptModule):
    def __init__(self, n_state, n_actions):
        super(RamDQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_state, 256),
            ACT(),
            nn.Linear(256, 128),
            ACT(),
            nn.Linear(128, 64),
            ACT(),
            nn.Linear(64, n_actions)
        )

    @torch.jit.script_method
    def forward(self, x):
        return self.layers(x)
