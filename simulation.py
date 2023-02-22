
import numpy as np
import world_data_extractor
import pickle as pck


def pickilizer(obj, filename):
    file = open(filename, 'wb')
    pck.dump(obj, file)
    file.close()

def unpickle(filename):
    file = open(filename, 'rb')
    obj = pck.load(file)
    file.close()
    return obj

# terrain_data, starting_height = world_data_extractor.run()

terrain_data = unpickle("terrain_data.pck")
starting_height = 70

reward_table = {
    "diamond_ore": 1000,
    "emerald_ore": 500,
    "redstone_ore": 100,
    "lapis_ore": 100, 
    "gold_ore": 100,
    "iron_ore": 10,
    "coal_ore": 5,
}



# Pre-process unimportant blocks to stone
all_blocks = set(np.unique(terrain_data))
exclusion = {"air", "lava", "flowing_lava", "water", "flowing_water", "bedrock"}.union(set(reward_table.keys()))
for i in all_blocks - exclusion:
    terrain_data[terrain_data==i] = "stone"

    
class Simulation:
    class Agent:
        def __init__(self, x, y, z) -> None:
            self.x, self.y, self.z = x, y, z



    def __init__(self, terrain_data, starting_height) -> None:
        self.terrain_data = terrain_data
        self.starting_height = starting_height

        self.agent = Simulation.Agent(int(terrain_data).shape[1]/2,starting_height,int(terrain_data).shape[2]/2)
    

    def at(self, x,y,z):
        return self.terrain_data[y, x, z]

    
print(np.unique(terrain_data))
# obj = tbd(terrain_data, starting_height)
# print(obj.at(150, 9, 150))

# for i in range(terrain_data.shape[0]): 
#     print(obj.at(150, i, 150))