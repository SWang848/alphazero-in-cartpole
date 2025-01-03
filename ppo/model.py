import torch
import torch.nn as nn
import torch.nn.functional as F

def init(module, weight_init, bias_init, gain=1):
    for name, param in module.named_parameters():
        if "weight" in name:
            weight_init(param, gain=gain)
        if "bias" in name:
            bias_init(param)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(CNN, self).__init__()

        # Define the CNN structure
        self.main = nn.Sequential(
            nn.Conv2d(num_inputs, 32, 3, stride=1),  # Output size: [32, 9, 9]
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1),  # Output size: [64, 7, 7]
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),  # Output size: [32, 5, 5]
            nn.ReLU(),
            Flatten(),  # Flatten the output for the Linear layer
            nn.Linear(32 * 5 * 5, hidden_size),  # Adjusted to match the flattened size
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_shape, num_blocks=10, out_channels=64, hidden_size=512):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], out_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels) for _ in range(num_blocks)]
        )
        self._hidden_size = hidden_size
        
        self.conv2 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1)
        self.obs_out = nn.Linear(input_shape[1] * input_shape[2] * out_channels // 4, hidden_size)

        self.train()

    def forward(self, x):
        x = self.conv1(x)

        for block in self.res_blocks:
            x = block(x)

        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten
        obs_out = F.relu(self.obs_out(x))

        return obs_out

    @property
    def output_size(self):
        return self._hidden_size


class Actor(nn.Module):
    def __init__(
        self,
        obs_shape,
        num_act,
        device,
        hidden_size=512,
    ):
        super(Actor, self).__init__()
        self.resnet = ResNet(obs_shape, num_blocks=10, out_channels=64, hidden_size=hidden_size)

        # Linear layer for value estimation
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )
        self.actor = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, num_act))
        )

        self._hidden_size = hidden_size

        self.train()
        self.to(device)

    def forward(
        self,
        obs: torch.tensor,
    ) -> torch.tensor:
        obs_out = self.resnet(obs)
        action_out = self.actor(obs_out)

        return action_out


class Critic(nn.Module):
    def __init__(
        self,
        obs_shape,
        device,
        hidden_size=512,
    ):
        super(Critic, self).__init__()
        self.resnet = ResNet(obs_shape, num_blocks=10, out_channels=64, hidden_size=hidden_size)
        # Linear layer for value estimation
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )
        self.critic = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, 1))
        )

        self.train()
        self.to(device)

    def forward(
        self,
        obs: torch.tensor,
    ) -> torch.tensor:
        obs_out = self.resnet(obs)
        x_out = self.critic(obs_out)
        
        return x_out
