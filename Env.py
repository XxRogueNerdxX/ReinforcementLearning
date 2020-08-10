import gym 
import collections 
import numpy as np 
import cv2




class env_manager():
    def __init__(self, name = 'CartPole-v0', repeat = 2 , shape = (1,84,84)):
        self.env = gym.make(name)
        self.stack =   collections.deque(maxlen= repeat)
        self.shape = shape
        self.output_dims = [2,84,84]
        self.mode = 'rgb_array'
        self.env._max_episode_steps = 1000

    def _preprocess(self, img):
        img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.shape[1:])
        img = np.expand_dims(img, axis=0)
        img = img/255
        return img
    
    def observation_space(self):
        return self.output_dims

    def action_space(self):
        return self.env.action_space.n

    def reset(self):
        self.stack.clear()
        _ = self.env.reset()
        observation = self.env.render(self.mode)
        observation = self._preprocess(observation)

        for _ in range(self.stack.maxlen):
            self.stack.append(observation)
        
        return np.array(self.stack).reshape(-1,*self.shape[1:])

    def step(self,action):
        _,reward,done,info = self.env.step(action)
        observation = self.env.render(self.mode)
        observation = self._preprocess(observation)
        self.stack.append(observation)
        state = np.array(self.stack).reshape(-1,*self.shape[1:])

        return state, reward, done, info
    
    def render(self):
        self.env.render()

       


