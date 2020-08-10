import numpy as np 
import matplotlib.pyplot as plt 
from Untitle import env_manager
from agent import Agent

env = env_manager('CartPole-v0')
n_games = 300
agent = Agent(lr = 0.01, mem_size = 20000, batch_size =32, replace = 100)

n_steps = 0 
score_list = []
eps_history = []
for i in range(n_games):
    done = False
    observation = env.reset()

    score = 0 
    while not done: 
        action = agent.choose_action(observation)
        observation_ , reward, done , info = env.step(action)
        score += reward    
        agent.mem.stor_mem(observation, action, reward, observation_, done)

        agent.learn()
        observation = observation_ 
    score_list.append(score)
    eps_history.append(agent.epsilon)
   
    if i%10 == 0:
        j = 0
        score_avg = sum(score_list[j:])/len(score_list[j:])
        #score_avg = sum(score_list)/len(score_list)
        print('score', score_avg,'epsilon',agent.epsilon)
        agent.save_net()
        agent.load_net()
        j+=10   

N = len(score_list)
running_avg = np.empty(N)
for t in range(N):
	running_avg[t] = np.mean(score_list[max(0, t-20):(t+1)])
    
x = np.arange(n_games)
plt.plot(x, score_list)
plt.plot(x, running_avg)

plt.show()

