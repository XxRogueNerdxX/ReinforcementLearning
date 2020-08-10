import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np
import torch as T 

class Network(nn.Module):
    def __init__(self, lr, n_actions, input_dims,name):
        super(Network, self) .__init__()
        self.input_dims = input_dims
        self.name = name
        self.conv1 = nn.Conv2d(self.input_dims[0],32,8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        dims = self.calculate_dims()
        self.fc1 = nn.Linear(dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if  T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_dims(self):
        dim = T.zeros(self.input_dims)
        dim = dim.reshape(1,*self.input_dims)
        layer1 = self.conv1(dim)
        layer1 = self.conv2(layer1)
        layer1 = self.conv3(layer1)

        return np.prod(layer1.size())

    def forward(self,state):
        layer1 = F.relu(self.conv1(state))
        layer2 = F.relu(self.conv2(layer1))
        layer3 = F.relu(self.conv3(layer2))

        layer4 = T.flatten(layer3, start_dim=1)
        layer5 = F.relu(self.fc1(layer4))
        action = self.fc2(layer5)
        
        return action
    
    def save(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(),self.name)
    
    def load(self):
        print('...Loading checkpoint....')
        self.load_state_dict(T.load(self.name))

class Replay():
    def __init__(self, mem_size, input_dims):
        self.mem_size = mem_size
        self.mem_cntr = 0
    
        self.states = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.actions = np.zeros((self.mem_size), dtype = np.int64)
        self.rewards = np.zeros((self.mem_size), dtype = np.float64)
        self.next_states = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.dones = np.zeros((self.mem_size), dtype = np.uint8)
        
    def stor_mem(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done
        self.mem_cntr+=1
    
    def get_mem(self, batch_size):
        batch = min(self.mem_size, self.mem_cntr)
        index = np.random.choice(batch, batch_size, replace = False)

        state = self.states[index]
        action = self.actions[index]
        reward = self.rewards[index]
        next_state = self.next_states[index]
        done = self.dones[index]
    
        return state , action , reward, next_state, done
        
        

        