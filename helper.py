"""
Helper functions:
http://microsoft.github.io/malmo/0.30.0/Documentation/annotated.html
"""

import pickle as pck
import numpy as np

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
