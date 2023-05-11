# Write policy gradient model for poker game
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x1, x2):
        out = self.conv(x1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, x2), 1)
        out = self.fc(out)
        return out
    

class PokerNet(ResNet):
    def __init__(self, input_size, hidden_size, output_size):
        super(PokerNet, self).__init__(ResidualBlock, [2, 2, 2], num_classes=2)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = F.elu(self.bn3(self.fc3(x)))
        x = F.elu(self.bn4(self.fc4(x)))
        x = F.elu(self.bn5(self.fc5(x)))
        pi = self.pi(x)
        v = self.v(x)
        return torch.softmax(pi, dim=1), v
    
    def loss(self, pi, v, action, reward):
        m = torch.distributions.Categorical(pi)
        advantage = reward - v
        loss = -m.log_prob(action) * advantage.detach() \
            + F.smooth_l1_loss(v.flatten(), reward.detach()) - 0.01 * m.entropy()
        loss = loss.mean()
        return loss
    
    def get_action(self, state):
        self.eval()
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(self.device)
        pi, v = self.forward(state)
        # print(pi.detach().cpu().numpy()[0], v.detach().cpu().numpy()[0])
        m = torch.distributions.Categorical(pi)
        action = m.sample()
        return action.item()
    
    def predict(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(self.device)
        pi, v = self.forward(state)
        action = torch.argmax(pi)
        # print(pi.detach().cpu().numpy())
        return action.item()
        
    def _training(self, data):
        self.train() 
        states = Variable(torch.from_numpy(np.array(data['state'])).float()).to(self.device)
        actions = Variable(torch.from_numpy(np.array(data['action'])).long()).to(self.device)
        rewards = Variable(torch.from_numpy(np.array(data['reward'])).float()).to(self.device)
        data = list(zip(states, actions, rewards))
        data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
        total_v = 0
        
        for data in data_loader:
            self.optimizer.zero_grad()
            loss = 0
            state, action, reward = data
            pi, v = self.forward(state)
            loss = self.loss(pi, v, action, reward)
            loss.backward()
            self.optimizer.step()
            total_v += v.mean().item()
        return total_v / len(data_loader)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    def save(self, path):
        torch.save(self.state_dict(), path)