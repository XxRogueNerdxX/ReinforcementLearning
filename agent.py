import numpy as np 
from network_and_replay import Network, Replay
from Untitle import env_manager
import torch as T 

class Agent():
    def __init__ (self , lr, mem_size, batch_size, replace, 
                gamma = 0.99,epsilon = 0.9, eps_min = 0.01,
                eps_dec = 1e-4):
        
        self.env = env_manager('CartPole-v0')
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon 
        self.eps_min = eps_min
        self.eps_dec = eps_dec 
        self.n_actions = self.env.action_space()
        self.action_space = np.arange(self.n_actions)
        self.input_dims = self.env.observation_space()
        
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.replace = replace
        self.replace_cntr = 0 

        self.q_eval = Network(self.lr, self.n_actions, self.input_dims,'Model.pth')
        self.q_pred = Network(self.lr, self.n_actions, self.input_dims, 'Model_2.pth')
        self.mem = Replay(self.mem_size, self.input_dims)
        
    def decay(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
    
    def choose_action(self, state):
        j = 0 
        e = np.random.uniform(0,1)
        if e < self.epsilon:
            action = np.random.choice(self.action_space)
        else : 
            state = T.tensor([state], dtype = T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
            j = 1
        
        return action 

    def store_memory(self, state, action, reward, next_state, done):
        self.mem.stor_mem(state, action, reward, next_state, done)

    def sample_mem(self):
        states, actions , rewards, next_states, dones = self.mem.get_mem(self.batch_size)

        state = T.tensor(states, dtype = T.float).to(self.q_eval.device)
        action = T.tensor(actions).to(self.q_eval.device)
        reward = T.tensor(rewards).to(self.q_eval.device)
        next_state = T.tensor(next_states, dtype = T.float).to(self.q_eval.device)
        done = T.tensor(dones).to(self.q_eval.device)

        return state, action, reward, next_state, done
    
    def save_net(self):
        self.q_eval.save()
        self.q_pred.save()
    
    def load_net(self):
        self.q_eval.load()
        self.q_pred.load()

    def replace_network(self):
        if self.replace_cntr % self.replace == 0:
            self.q_pred.load_state_dict(self.q_eval.state_dict())

    def learn(self):
        if self.mem.mem_cntr < self.batch_size:
            return 
        
       
        self.replace_network()

        states, actions, rewards, states_ , dones = self.sample_mem()

        indices = np.arange(self.batch_size)
       # indices = T.arange(end = self.batch_size,dtype= T.int64).to(self.q_eval.device)


        elf.q_eval.optimizer.zero_grad()
        q_out = self.q_eval.forward(states)
        q_out = q_out[indices, actions]
        q_next = self.q_pred(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_out, q_target).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.replace_cntr += 1 

        self.decay()



      
        
 


