# # imports are always needed
# import torch


# # get index of currently selected device
# print(torch.cuda.current_device()) # returns 0 in my case


# # get number of GPUs available
# torch.cuda.device_count() # returns 1 in my case


# # get the name of the device
# torch.cuda.get_device_name(0) # good old Tesla K80

import torch
import random
import numpy as np
from collections import deque
from simulation_q import Simulation 
from NN.model import Linear_QNet, QTrainer
from NN.graph import plot
import helper
from sklearn.preprocessing import OneHotEncoder

"""
https://www.youtube.com/watch?v=L8ypSXwyBds
https://github.com/patrickloeber/snake-ai-pytorch
git@github.com:patrickloeber/snake-ai-pytorch.git
"""

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

with open("all_blocks.txt", 'r') as f:
    string = f.read().strip()
    a_list = string.split(",")

BLOCK_MAP = {}
ACTION_MAP = {}

actions = ["N","S","W","E","U","M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]

"""
Make mapping from string actions and blocks to integer. Because pytorch demands it this way.
"""
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


OHE_STATE = OneHotEncoder()
block_keys = np.array(list(BLOCK_MAP.keys())).reshape(-1, 1)
OHE_STATE.fit(block_keys)

OHE_ACTION = OneHotEncoder()
action_keys = np.array(list(ACTION_MAP.keys())).reshape(-1, 1)
OHE_ACTION.fit(action_keys)

print(action_keys)

print(OHE_STATE.transform([["stone"],
                           ["diamond_ore"],
                           ["grass"]
                           ]).toarray())