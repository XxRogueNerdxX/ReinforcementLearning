# ReinforcementLearning

This is a DQN code, which consists of four files. The Env file contains the gym wrapper class. It was custom wrapped hence I did'nt use gym.Wrapper. The second file houses the replay memory and the network. The agent file has the agent class which access the replay and memory class to create q_pred(Target) and q_eval(Prediction) networks
