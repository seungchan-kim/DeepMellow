import gym
import random
import numpy as np
import sys
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

class DQN_Agent:
    def __init__(self, env, lr, num_hidden_layers, num_neurons, target_net_update_freq, mellowBool, w):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.001
        self.learning_rate = lr #parameterized
        self.model = self.build_model(num_hidden_layers, num_neurons) #parameterized
        #num_hidden_layers = 1, 2, 3 / num_neurons = 50, 100, 300, 500
        print(self.model.summary())
        self.target_model = self.build_model(num_hidden_layers, num_neurons) #parameterized
        self.target_network_update_freq = target_net_update_freq #parametrized
        self.update_target_model()
        self.minibatch_size = 64
        self.mellowBool = mellowBool
        self.w = w

    def build_model(self, num_hidden_layers, num_neurons):
        model = Sequential()
        for i in range(num_hidden_layers): #hidden layers
            if i == 0:
                model.add(Dense(num_neurons,input_dim=self.state_size,activation='relu'))
            else:
                model.add(Dense(num_neurons,activation='relu'))
        model.add(Dense(self.action_size, activation='linear')) # output layer
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    def loss_func(self, target, prediction):
        td_error = target - prediction
        return K.mean(K.sqrt(1+K.square(td_error))-1, axis=-1)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            q_val = self.model.predict(state)
            action = np.argmax(q_val[0])
        return action

    def run(self):
        minibatch = random.sample(self.memory,self.minibatch_size)
        S = np.zeros((self.minibatch_size, self.state_size))
        y = np.zeros((self.minibatch_size, self.action_size))
        for i in range(self.minibatch_size):
            state, action, reward, next_state, done = minibatch[i]
            S[i] = state
            t = self.model.predict(state) #1xna            
            if done:
                t[:,action] = reward
            else:
                Q_target_estimate = self.target_model.predict(next_state)
                if mellowBool:
                    t[:,action] = reward + self.gamma * self.mellowmax(Q_target_estimate[0], w)
                else:
                    t[:,action] = reward + self.gamma * np.max(Q_target_estimate[0])
            y[i] = t
        self.model.train_on_batch(S, y)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def mellowmax(self, q, w):
        n = self.action_size
        c = np.multiply(w,q)
        a = np.max(c)
        y = np.log(np.sum(np.exp(c-a))) + a
        return 1/w * (np.log(1/n) + y)
        
        
        
        

if __name__ == '__main__':
    env=gym.make('Acrobot-v1')
    num_episodes = 1000
    filepath = sys.argv[1]
    f = open(filepath,'r')
    lrstr = f.readline().strip('\n')
    lr = float(lrstr)
    num_hidden_layers = int(f.readline().strip('\n'))
    num_neurons = int(f.readline().strip('\n'))
    target_net_update_freq = int(f.readline().strip('\n'))
    max_or_mm = sys.argv[2]
    mellowBool = False
    if max_or_mm == 'max':
        mellowBool = False
        w = 0
        iterNum = sys.argv[3]
    elif max_or_mm == 'mellow':
        mellowBool = True
        w = int(sys.argv[3])
        iterNum = sys.argv[4]
    else:
        raise Exception('should be either max or mellowmax!')
        
    
    print(lr)
    print(num_hidden_layers)
    print(num_neurons)
    print(target_net_update_freq)

    filename = filepath[6:]

    if mellowBool:
        csvname = 'csv_acrobot/reward_' + filename +'_' + max_or_mm + '_w' + str(w) + '_' + str(iterNum) + '.csv'
    else:
        csvname = 'csv_acrobot/reward_' + filename +'_' + max_or_mm + str(iterNum) + '.csv'

    print(csvname)
    file1 = open(csvname, 'w')
    
    max_timesteps_ep = 500
    agent = DQN_Agent(env, lr, num_hidden_layers, num_neurons, target_net_update_freq, mellowBool, w)
    state_size = agent.state_size
    action_size = agent.action_size
    batch_size = agent.minibatch_size
    total_steps = 0
    done = False

    for e in range(num_episodes):
        total_reward = 0
        episode_step = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        for t in range(max_timesteps_ep):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1,state_size])
            total_reward += reward
            total_steps += 1
            episode_step += 1
            agent.store_memory(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.run()

            if total_steps > 0 and total_steps % target_net_update_freq == 0:
                agent.update_target_model()

            if done:
                print("episode %2i, reward: %7.3f, steps: %i, next eps: %7.3f"%(e, total_reward, episode_step, agent.epsilon))
                file1.write(str(total_reward)+','+str(e)+',\n')
                break
    

    
    
