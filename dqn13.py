import gym
import numpy as np
import tensorflow as tf
import random
from collections import deque

gamma = 0.9
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
REPLAY_SIZE = 10000
BATCH_SIZE = 32

class DQN():
    # DQN agent
    def __init__(self, env):
        self.replay_buffer = deque()
        # store transition
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n


        self.create_Q_network()
        # initialize action-value function Q
        self.create_training_method()
        # initialize replay momory

        # inti session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # define Q(
        # network weights
        # use 20 layer MLP
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder('float', [None, self.state_dim])
        # hidden layer
        hidden_layer = tf.nn.relu(tf.matmul(self.state_input, W1)+b1)
        # Q value layer
        self.Q_value = tf.matmul(hidden_layer,W2)+b2

    def weight_variable(self, shape):
            initial = tf.truncated_normal(shape)
            return tf.Variable(initial)

    def bias_variable(self, shape):
            initial = tf.constant(0.01,shape=shape)
            return tf.Variable(initial)

    def perceive(self, state, action, reward, next_state, done):
        # store trasition in D
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append(
            (state,one_hot_action,reward,next_state,done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        # pop from left
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()
        ## when replay>batch,begin the training

    def egreedy_action(self,state):
        Q_value  =  self.Q_value.eval(feed_dict={
            self.state_input:[state]})[0]
        if random.random() <= self.epsilon:
            return random.randint(0,self.action_dim - 1)
        else :
            return np.argmax(Q_value)
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000


    def action(self,state):
        return np.argmax(
            self.Q_value.eval(
                feed_dict={self.state_input:[state]})[0])
        # nn action


    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action_dim])
        self.y_input = tf.placeholder('float',[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),
                                  reduction_indices =[1]
                                  )
        # Q-value for each action and use reduce_sum into one dimension
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action ))
        # a gradient descent step
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
        ## y_input = targetQ

    def train_Q_network(self):
        # use minibatch to train
        self.time_step += 1
        # step1:obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # step2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict=
                                          {self.state_input:next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            # for terminal
            if done:
                y_batch.append(reward_batch[i])
            # for non-terminal
            else:
                y_batch.append(reward_batch[i] + gamma * np.argmax(Q_value_batch[i]))

        self.optimizer.run(feed_dict={self.y_input:y_batch,
                                      self.action_input:action_batch,
                                      self.state_input:state_batch})

# Hyper parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000  # Episode limitation
STEP = 300  # step limitation in an episode
TEST = 10

def main():
    # initialize OpenAI gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = DQN(env)

    for episode in range(EPISODE):
        # ignore preprocessing and using observation as state
        # initialize state
        state = env.reset()
        # Train t=1,T
        for step in range(STEP):

            # select action through epsilon-greedy policy for training
            action = agent.egreedy_action(state)
            # execute action in the emulator and observe reward and image
            next_state, reward, done, _ = env.step(action)
            # next_state=observation (object)
            # reward (float)
            # done (boolean)
            # whether it’s time to reset the environment again.
            # True indicates the episode has terminated.
            # info (dict)seldom

            # define reward for agent

            ## reward_agent = -1 if done else 0.1

            ## input
            agent.perceive(state, action,  reward, next_state, done)
            state = next_state
            if done :
                break

    # test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(state) ##random action
                    state, reward, done, _ = env.step(action)
                    total_reward  += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print('episode:', episode, 'Evaluation average reward :',ave_reward )
            if ave_reward >= 200:
                break


if __name__ == '__main__':
    main()












