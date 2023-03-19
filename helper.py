"""
Helper functions:
http://microsoft.github.io/malmo/0.30.0/Documentation/annotated.html
"""

import pickle as pck
import numpy as np
import json, math
from keras.utils import to_categorical


REWARD_TABLE = {
    "diamond_ore": 1000,
    "emerald_ore": 500,
    "redstone_ore": 100,
    "lapis_ore": 100, 
    "gold_ore": 100,
    "iron_ore": 10,
    "coal_ore": 5,
}
EXCLUSION = {"air", "lava", "flowing_lava", "water", "flowing_water", "bedrock"}.union(set(REWARD_TABLE.keys()))
DEATH_VALUE = -1000
ALL_MOVES = ["N","S","W","E","U","M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]


def pickilizer(obj, filename):
    file = open(filename, 'wb')
    pck.dump(obj, file)
    file.close()

def unpickle(filename):
    file = open(filename, 'rb')
    obj = pck.load(file)
    file.close()
    return obj

def create_custom_world(width, length, layers):
    """
    Width: (int) world width
    Length: (int) world length
    Layers: Layers to create world ie) [(3, "air"), (5, "stone), etc.]

    Example: create_custom_world(3, 3, [(4, "air"), (2, "stone")])
    
    """

    world = np.array([])
    layers.reverse()    
    total_y_length = 0
    for layer in layers:
        n = np.full(shape=(layer[0], width, length), fill_value=layer[1])
        total_y_length += layer[0]
        world = np.append(world, n)

    world = np.reshape(world, (total_y_length, width, length))

    print("WORLD SHAPE!!:", world.shape)
    #print(world)

    return world, total_y_length

def get_grid_observation(world_state, name):

    if (world_state.number_of_observations_since_last_state):
        state = world_state.observations[-1].text
        state = json.loads(state)
        return state[name], math.floor(state["YPos"])
        
    return None, None


def onehotencode():
    # TODO: Change this implementation to One Hot Encoding
    '''
    with open("all_mc_blocks.txt", 'r') as f:
        string = f.read().strip()
        block_list = string.split(",")
        
    '''
    
    block_list = ["stone", "air", "bedrock"]
        
    for block in REWARD_TABLE.keys():
        block_list += block
        block_list += "air+" + block
    
    BLOCK_MAP = {}
    ACTION_MAP = {}

    actions = ["N","S","W","E","U","M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]

    #ONE HOT
    block_code = to_categorical([i for i in range(len(block_list))])
    action_code = to_categorical([i for i in range(len(actions))])

    # print(block_code)
    # print(action_code)


    for i in range(len(block_list)):
        BLOCK_MAP[block_list[i]] = block_code[i]
    
    for i in range(len(actions)):
        ACTION_MAP[actions[i]] = action_code[i]

    return ACTION_MAP, BLOCK_MAP

def enumerate_one_hot():
    block_list = ["stone", "air", "bedrock"]
        
    for block in REWARD_TABLE.keys():
        block_list.append(block)
        block_list.append("air+" + block)
    
    BLOCK_MAP = {}
    ACTION_MAP = {}

    actions = ["N","S","W","E","U","M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]

    #ONE HOT
    block_code = np.array([i for i in range(len(block_list))], dtype=np.float32)
    action_code = np.array([i for i in range(len(actions))], dtype=np.float32)

    for i in range(len(block_list)):
        BLOCK_MAP[block_list[i]] = block_code[i]
    
    for i in range(len(actions)):
        ACTION_MAP[actions[i]] = action_code[i]

    # print(BLOCK_MAP)
    return BLOCK_MAP, ACTION_MAP


# def convert_bits(a):
#     l = []
#     for i in a:
#         num = 0
#         for j in range(len(i)):
#             num += i[j]
#             num << 1
#         l.append(num)

#     return np.array(l, dtype=np.float32)

if __name__ == "__main__":
    #onehotencode()
    pass