
import numpy as np
import world_data_extractor
import pickle as pck
import random


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
starting_height = 70 #CHANGE LATER

reward_table = {
    "diamond_ore": 1000,
    "emerald_ore": 500,
    "redstone_ore": 100,
    "lapis_ore": 100, 
    "gold_ore": 100,
    "iron_ore": 10,
    "coal_ore": 5,
}

class Agent:
        """
        Consider health if time permits.
        """
        def __init__(self, x, y, z) -> None:
            self.x, self.height, self.z = x, y, z
            self.health = 20
            self.inventory = []
        
        def move(self,direction):
            
            if direction == "N":
                self.z -= 1
                #consider height change
            elif direction == "S":
                self.z += 1
            elif direction == "E":
                self.x += 1
            elif direction == "W":
                self.z -= 1
            elif direction == "U":
                self.height += 1
        
            
            
        def get_possible_moves(self, state):
            """
            The x-axis indicates the player's distance east (positive) or west (negative) of the origin point—i.e., the longitude,
            The z-axis indicates the player's distance south (positive) or north (negative) of the origin point—i.e., the latitude 
            """
            moves = []
            #MOVING
            # N
            if state[0].startswith('air') and state[1].startswith('air'):
                moves.append("N")
                
            # E
            if state[2].startswith('air') and state[3].startswith('air'):
                moves.append("E")

            # S
            if state[4].startswith('air') and state[5].startswith('air'):
                moves.append("S")

            # W
            if state[6].startswith('air') and state[7].startswith('air'):
                moves.append("W")

            #U
            if state[8].startswith('air'):
                moves.append("U")

            #MINING
            #(NL, NU, EL, EU, SL, SU, WL, WU, U, D, height)

            M = ["M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]

            for i in range(len(state) - 1):
                if not state[i].startswith('air'):
                    moves.append(M[i])

            return moves


class Simulation:
    def __init__(self, terrain_data, starting_height) -> None:
        self.terrain_data = terrain_data
        self.starting_height = starting_height

        self.agent_mined = set()

        self.agent = Agent(int(terrain_data.shape[1]/2),starting_height,int(terrain_data.shape[2]/2))

    def run(self):
        self.fall() #make agent touch the ground
        
        #do the simulation

        #q learning?
        
        return #?
        

    def at(self, x,y,z):
        if self.is_mined(x,y,z):
            return "air"
        
        return self.terrain_data[y, x, z]
    
    def agent_xyz(self):
        return self.agent.x, starting_height - self.agent.height, self.agent.z
    
    def choose_move(self, epsilon):
        possible_moves = self.agent.get_possible_moves(self.get_current_state())
        if (random.random < epsilon):
            return #random move from 
        
        # best move given q function

        # add reward?
        
        # mine out blocks if needed    self.mine_out(x,y,z)
        
        return #best_move(get_current_state)

    def get_current_state(self) -> "state":
        """
        The x-axis indicates the player's distance east (positive) or west (negative) of the origin point—i.e., the longitude,
        The z-axis indicates the player's distance south (positive) or north (negative) of the origin point—i.e., the latitude 
        """
        x, y, z = self.agent_xyz()

        '''
        (NL, NU, EL, EU, SL, SU, WL, WU, U, D, height)
        '''
        #X, Z
        dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        state_space = []
        for d in dir:
            state_space.append(self.at(x+d[0], y, z+d[1]))   # Lower
            state_space.append(self.at(x+d[0], y+1, z+d[1])) # Upper

        state_space.append(self.at(x, y - 1, z))   # below feet
        state_space.append(self.at(x, y + 2, z))   # above head
        state_space.append(self.agent.height)

        #TODO: AIR BLOCK THING
        #if block air, make it "air+diamond_ore", etc
        

        
        return tuple(state_space)

    def mine(self, x, y, z):
        self.agent_mined.add((x, y, z))
    
    def is_mined(self, x, y, z):
        return (x, y, z) in self.agent_mined

    
    def fall(self):
        #1 point (half a heart) for each block of fall distance after the third
        x, y, z = self.agent_xyz()

        fall_through = ["air", "lava", "flowing_lava", "water", "flowing_water"]

        while(True): 
            if (any(self.at(x, y - 1, z) == i for i in fall_through)):
                self.agent.height -= 1
                y = starting_height - self.agent.height
            else: #or hits bottom of the world
                break
            
    def agent_death(self):
        x, y, z = self.agent_xyz()

        danger = ["lava", "flowing_lava"]

        return (any(self.at(x, y, z) == i for i in danger))


if __name__ == "__main__":
    terrain_data = unpickle("terrain_data.pck")
    starting_height = 70 #CHANGE LATER
    
    # Pre-process: convert unimportant blocks to stone
    all_blocks = set(np.unique(terrain_data))
    exclusion = {"air", "lava", "flowing_lava", "water", "flowing_water", "bedrock"}.union(set(reward_table.keys()))
    for i in all_blocks - exclusion:
        terrain_data[terrain_data==i] = "stone"

    # Testing Simulation.get_current_state
    
    test = Simulation(terrain_data, starting_height=50)
    print(test.get_current_state())
    # Returned ('stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 50)

    # Replace two blocks north of agent w/ Air:
    # Expect: ('air', 'air', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 50)
    # self.terrain_data[y, x, z]

    test.mine(150, 50, 150-1)
    test.mine(150, 51, 150-1)
    
    # test.terrain_data[50, 150, 150-1] = "air"
    # test.terrain_data[51, 150, 150-1] = "air"

    print(test.terrain_data[50, 150, 150-1])
    print(test.terrain_data[51, 150, 150-1])

    print(test.get_current_state())
