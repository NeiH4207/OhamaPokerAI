# Write policy gradient model for poker game
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

class PokerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PokerNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size,  hidden_size >> 1)
        self.bn2 = nn.BatchNorm1d(hidden_size >> 1)
        self.fc3 = nn.Linear(hidden_size >> 1,  hidden_size >> 2)
        self.bn3 = nn.BatchNorm1d(hidden_size >> 2)
        self.fc4 = nn.Linear(hidden_size >> 2,  hidden_size >> 3)
        self.bn4 = nn.BatchNorm1d(hidden_size >> 3)
        self.fc5 = nn.Linear(hidden_size >> 3,  hidden_size >> 4)
        self.bn5 = nn.BatchNorm1d(hidden_size >> 4)
        self.pi = nn.Linear(hidden_size >> 4, output_size)
        self.v = nn.Linear(hidden_size >> 4, 1)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.002)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        pi = self.pi(x)
        v = self.v(x)
        return torch.softmax(pi, dim=1), v
    
    def loss(self, pi, v, action, reward):
        m = torch.distributions.Categorical(pi)
        advantage = reward - v
        loss = -m.log_prob(action) * advantage
        loss = loss.mean()
        return loss
    
    def get_action(self, state):
        self.eval()
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        pi, v = self.forward(state)
        m = torch.distributions.Categorical(pi)
        action = m.sample()
        return action.item()
    
    def predict(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        pi, v = self.forward(state)
        action = torch.argmax(pi)
        print(pi.detach().numpy())
        return action.item()
        
    def _training(self, data):
        self.train() 
        states = Variable(torch.from_numpy(np.array(data['state'])).float())
        actions = Variable(torch.from_numpy(np.array(data['action'])).long())
        rewards = Variable(torch.from_numpy(np.array(data['reward'])).float())
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
        return total_v
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    def save(self, path):
        torch.save(self.state_dict(), path)