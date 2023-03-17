import torch
import random
import numpy as np
from collections import deque
from simulation_q import Simulation 
from model import Linear_QNet, QTrainer
from graph import plot
import helper

"""
https://github.com/patrickloeber/snake-ai-pytorch
git@github.com:patrickloeber/snake-ai-pytorch.git
"""

"""
TODOs In Order of Priorities: 
1) Double check validity of forward pass of Linear_QNet forward pass function
2) Add Double Deep Q Learning 
2) One Hot Encoding instead of string --> integer mapping
    My hope is that OHE could eliminate bias towards higher integers.
3) Hyperparameter tunings
4) Restructure: it's a mess right now
"""

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

with open("all_mc_blocks.txt", 'r') as f:
    string = f.read().strip()
    a_list = string.split(",")

BLOCK_MAP = {}
ACTION_MAP = {}

"""
Make mapping from string actions and blocks to integer. Because pytorch demands it this way.
"""

actions = ["N","S","W","E","U","M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]
for action in actions:
    if action not in ACTION_MAP:
        ACTION_MAP[action] = len(ACTION_MAP)


for string in a_list:
    if string not in BLOCK_MAP:
        BLOCK_MAP[string] = len(BLOCK_MAP)
BLOCK_MAP["air"] = 1151

# Account for air + [block]
for key, _val in list(BLOCK_MAP.items()):
    if key != "air":
        BLOCK_MAP[f"air+{key}"] = len(BLOCK_MAP)

def str_to_int(state):
    state = list(state)
    state[-1] = str(state[-1])
    state = list(map(lambda x: BLOCK_MAP[x], state[:-1])) + [int(state[-1])]
    return state

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.15 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 15)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        state = str_to_int(state)
        next_state = str_to_int(next_state)
        # print("ACTION", action)
        action = list(map(lambda x: ACTION_MAP[x], [action]))
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, env):
        # random moves: tradeoff exploration / exploitation
        # self.epsilon = 80 - self.n_games
        moves = env.agent.get_possible_moves(state)
        state = list(state)
        state[-1] = str(state[-1])
        if random.random() < self.epsilon:
            action = random.choice(moves)
        else:
            state = list(map(lambda x: BLOCK_MAP[x], state[:-1])) + [int(state[-1])]
            # print(state)
            state0 = torch.tensor(state, dtype=torch.float32)
            # print(moves)
            prediction = self.model(state0, len(moves))
            ind = torch.argmax(prediction).item()
            # print(ind)
            action = moves[ind]

        return action


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    
    agent = Agent()

    terrain_data, terrain_height = helper.create_custom_world(50, 50, [(3, "air"), (5, "stone") ,(2,"diamond_ore")])
    starting_height = terrain_height - 1

    STEPS = 200

    for _ in range(1000):
        s = Simulation(terrain_data,starting_height)
        steps = 0
        while steps < STEPS:
            # get old state
            state_old = s.get_current_state()

            # get move
            final_move = agent.get_action(state_old, s)

            # perform move and get new state
            # reward, done, score = game.play_step(final_move)
            # state_new = agent.get_state(game)

            reward, done = s.get_reward(state_old, final_move)
            state_new = s.get_current_state()

            done = not (done == False)

            score = s.agent.inventory["diamond_ore"]# diamonds

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember

            # change from string to int
            state_old = str_to_int(state_old)
            final_move = ACTION_MAP[final_move]
            state_new = str_to_int(state_new)
            agent.remember(state_old, final_move, reward, state_new, done)

            # if done:
            #     # train long memory, plot result
                
            #     # game.reset()

            #     agent.n_games += 1
            #     agent.train_long_memory()

            #     if score > record:
            #         record = score
            #         agent.model.save()

            #     print('Game', agent.n_games, 'Score', score, 'Record:', record)

            #     plot_scores.append(score)
            #     total_score += score
            #     mean_score = total_score / agent.n_games
            #     plot_mean_scores.append(mean_score)
            #     plot(plot_scores, plot_mean_scores)

            #     break
                
            steps += 1


        agent.n_games += 1
        agent.train_long_memory()

        if score > record:
            record = score
            agent.model.save()

        print('Game', agent.n_games, 'Score', score, 'Record:', record)

        plot_scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        plot_mean_scores.append(mean_score)
        plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()