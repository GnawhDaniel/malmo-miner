import numpy as np
#import world_data_extractor
import random, copy
from helper import pickilizer, unpickle
from operator import add
import helper
from collections import defaultdict


# CONSTANTS
REWARD_TABLE = helper.REWARD_TABLE

EXCLUSION = {"air", "lava", "flowing_lava", "water", "flowing_water", "bedrock"}.union(set(REWARD_TABLE.keys()))
DEATH_VALUE = -1000
MOVE_PENALTY = 0


class Agent:
    """
    Consider health if time permits.
    """

    def __init__(self, x, y, z) -> None:
        self.x, self.height, self.z = x, y, z

        self.inventory = defaultdict(int)

    def move(self, direction):

        if direction == "N":
            self.z -= 1
            # consider height change
        elif direction == "S":
            self.z += 1
        elif direction == "E":
            self.x += 1
        elif direction == "W":
            self.x -= 1
        elif direction == "U":
            self.height += 1

    def get_possible_moves(self, state):
        """
        The x-axis indicates the player's distance east (positive) or west (negative) of the origin point—i.e., the longitude,
        The z-axis indicates the player's distance south (positive) or north (negative) of the origin point—i.e., the latitude
        """
        moves = []

        # MOVING
        # North
        if state[0].startswith('air') and state[1].startswith('air'):
            moves.append("N")
            # East
        if state[2].startswith('air') and state[3].startswith('air'):
            moves.append("E")
        # South
        if state[4].startswith('air') and state[5].startswith('air'):
            moves.append("S")
        # West
        if state[6].startswith('air') and state[7].startswith('air'):
            moves.append("W")
        # Upper
        if state[8].startswith('air'):
            moves.append("U")

        # MINING
        # (NL, NU, EL, EU, SL, SU, WL, WU, U, D, height)
        M = ["M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]
        for i in range(len(state) - 1):
            if not state[i].startswith('air') and state[i] != "bedrock":
                moves.append(M[i])
        #print(moves)
        return moves


class Simulation:
    def __init__(self, terrain_data, starting_height) -> None:
        self.terrain_data = copy.deepcopy(terrain_data)
        self.starting_height = starting_height

        self.agent_mined = set()
        self.agent_placed = set()
        self.diamonds_mined = 0

        self.agent = Agent(int(terrain_data.shape[1] / 2), starting_height - 2, int(terrain_data.shape[2] / 2))

    def boundary_check(self, x, y, z) -> bool:
        """
        Returns: True if in bounds
        """
        world_shape = self.terrain_data.shape  # (y, x, z)
        if x < 0 or x > world_shape[1] - 1:
            return False
        if z < 0 or z > world_shape[2] - 1:
            return False
        if y < 0 or y > starting_height:
            return False
        
        return True
  
    def at(self, x, y, z):
        if self.is_placed(x, y, z):
            return "stone"

        if self.is_mined(x, y, z):
            return "air"
        #print("testing",[y, x, z])
        return self.terrain_data[y, x, z]

    def agent_xyz(self):
        # starting_height - self.agent.heightcle
        return self.agent.x, self.agent.height, self.agent.z
    
    #change4
    def choose_move(self, epsilon, state, q_table):
        possible_moves = self.agent.get_possible_moves(state)
        #EPSILON
        if (random.random() < epsilon or state not in q_table.keys()):
            return possible_moves[random.randint(0, len(possible_moves) - 1)]
        else:
            """
            careful when add new move as the index will be wrong
            """
            move = ["N", "S", "W", "E", "U", "M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U",
                    "M_D"]  
            
            
            table_of_possible = [(move[i], x) for i, x in enumerate(q_table[state]) if move[i] in possible_moves]
            max_q_value = max([i[1] for i in table_of_possible])
            best_move = [i[0] for i in table_of_possible if i[1] == max_q_value]
            
            return random.choice(best_move)
        # best move given q function
        # add reward?
        # mine out blocks if needed    self.mine_out(x,y,z)


    def get_current_state(self) -> "state":
        """
        The x-axis indicates the player's distance east (positive)  or west (negative) of the origin point—i.e., the longitude,
        The z-axis indicates the player's distance south (positive) or north (negative) of the origin point—i.e., the latitude
        """
        x, y, z = self.agent_xyz()

        '''
        (NL, NU, EL, EU, SL, SU, WL, WU, U, D, height)
        '''
        # X, Z
        # dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dir = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        state_space = []

        # Searching Range
        r_distance = 5

        #NESW
        for d in dir:
            lower_coord = (x + d[0], y , z + d[1])
            if not self.boundary_check(*lower_coord):
                state_space.append("bedrock")  # Lower
            else:
                lower = self.at(*lower_coord)
                search_direction = (d[0], 0, d[1])

                if lower == "air":  # lower
                    # coord = (x + d[0], y, z + d[1])
                    max_block = self.recursive_search(lower_coord, search_direction, r_distance)

                    if max_block in REWARD_TABLE:
                        lower += "+" + max_block
                
                state_space.append(lower)  # Lower


            upper_coord = (x + d[0], y + 1 , z + d[1])
            #print(upper_coord)
            if not self.boundary_check(*upper_coord):
                state_space.append("bedrock")
            else:
                upper = self.at(x + d[0], y + 1, z + d[1])

                if upper == "air":  # upper
                    # coord = (x + d[0], y + 1, z + d[1])
                    max_block = self.recursive_search(upper_coord, search_direction, r_distance)

                    if max_block in REWARD_TABLE:
                        upper += "+" + max_block

                state_space.append(upper)  # Upper

        
        #ABOVE
        if not self.boundary_check(x, y + 2, z):
            state_space.append("bedrock")
        else:  
            above = self.at(x, y + 2, z)  # above head
            if above == "air":
                max_block = self.recursive_search((x, y + 2, z), (0, 1, 0), r_distance)

                if max_block in REWARD_TABLE:
                    lower += "+" + max_block

            state_space.append(above)   


        #BELOW
        if not self.boundary_check(x, y - 1, z):
            state_space.append("bedrock")
        else:
            state_space.append(self.at(x, y - 1, z))   # below feet
        
        #HEIGHT
        state_space.append(self.agent.height)

        return tuple(state_space)

    def recursive_search(self, coordinate, direction, search_depth):
        '''
        recursively finds max value block
        returns str: 'max_value_block'
        '''

        coordinate = tuple(coordinate)
        # base cases:
        # Search depth exceeded
        # print(coordinate)
        if search_depth <= 0:
            return "air"
        # coordinate to search out of bounds
        elif not self.boundary_check(coordinate[0], coordinate[1], coordinate[2]):
            return "air"
        # block to search is not air
        elif self.at(coordinate[0], coordinate[1], coordinate[2]) != 'air':
            return "air"

        # possible search directions
        directions = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (-1, 0, 0),
            (0, -1, 0),
            (0, 0, -1)
        ]

        # blocks found in the local search
        blocks = []

        # if a direction has a 0 where the direction is non-zero, we want to look in that direction (perpendicular)
        index_of_non_zero = 0
        for i in range(len(coordinate)):
            if coordinate[i] != 0:
                index_of_non_zero = i
                break

        for dir in directions:
            if dir[index_of_non_zero] == 0 or dir == direction:
                c = tuple(map(add, coordinate, dir))  # Adding tuples into each other
                if self.boundary_check(*c):
                    blocks.append(self.at(*c))

        max_local = blocks[0]
        for i in blocks:
            if i != max_local:
                if (max_local not in REWARD_TABLE):
                    max_local = i
                elif (i in REWARD_TABLE):
                    if REWARD_TABLE[i] > REWARD_TABLE[max_local]:
                        max_local = i

        # find max of other blocks recursively
        max_recurred = self.recursive_search(map(add, coordinate, direction), direction, search_depth - 1)

        if (max_local not in REWARD_TABLE):
            max_local = max_recurred
        elif (max_recurred in REWARD_TABLE):
            if REWARD_TABLE[max_recurred] > REWARD_TABLE[max_local]:
                max_local = max_recurred

        return max_local

    def get_reward(self, state, action):
        """
        Returns: int: 'reward', bool: 'death'
        """
        # MINING
        if action.startswith('M_'):
            # mining actions in the same index as the effected block in the state space
            M = ["M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]

            # get the index in the state space for the mining action
            block = M.index(action)

            # relative coordinates for each block
            coords = [(0, 0, -1), (0, 1, -1),  # N
                      (1, 0, 0), (1, 1, 0),  # E
                      (0, 0, 1), (0, 1, 1),  # S
                      (-1, 0, 0), (-1, 1, 0),  # W
                      (0, 2, 0),  # U
                      (0, -1, 0)  # D
                      ]

            # get the real-world coordinate
            coord = map(add, self.agent_xyz(), coords[block])
            #change3
            temp = [i for i in coord]
            #print(action,temp)
            # Unpack tuple and mine the block out
            
            block_mined = self.at(*temp)
            self.agent.inventory[block_mined] += 1
            self.mine(*temp)

            reward = 0
            dead = False

            if block in REWARD_TABLE:
                reward = REWARD_TABLE[block_mined]

            if block == "diamond_ore":
                self.diamonds_mined += 1

            # IF lava in the state space and doesn't move
            if (any(map(lambda x: x == "lava" or x == "flowing_lava", state))):
                dead = True

            #or falling
            died_falling = self.fall()
            dead = dead or died_falling

            if dead:
                reward += DEATH_VALUE

            # return the reward for mining the block
            return reward, dead

        # MOVING
        else:
            # relative coordinates for each move
            coords = [(0, 0, -1),  # N
                      (1, 0, 0),  # E
                      (0, 0, 1),  # S
                      (-1, 0, 0),  # W
                      (0, 1, 0),  # U
                      ]

            M = ["N", "E", "S", "W", "U"]
            relative_coord = coords[M.index(action)]

            # If up, place block under
            if action == "U":
                #change2
                c = self.agent_xyz()
                if self.at(*c) == 'air' or self.is_mined(*c):
                    self.place_block(*c)

            # Move the agent, return if died or not
            #print(coords[M.index(action)])
            # change1
            #print("test1",relative_coord)
            if self.agent_move(*relative_coord):  # if died
                #print("test2")
                return DEATH_VALUE, True

            # otherwise return
            return MOVE_PENALTY, False

    def mine(self, x, y, z):
        if (self.is_placed(x, y, z)):
            self.agent_placed.remove((x, y, z))

        self.agent_mined.add((x, y, z))

    def is_mined(self, x, y, z):
        return (x, y, z) in self.agent_mined

    def place_block(self, x, y, z):
        if (self.is_mined(x, y, z)):
            self.agent_mined.remove((x, y, z))

        self.agent_placed.add((x, y, z))

    def is_placed(self, x, y, z):
        return (x, y, z) in self.agent_placed

    def agent_move(self, x, y, z):
        #print("cord1", self.agent.x, self.agent.height, self.agent.z)
        #print("d",x,y,z)
        self.agent.x += x
        self.agent.height += y
        self.agent.x += z
        #print("cord2", self.agent.x, self.agent.height, self.agent.z)
        #print(self.get_current_state())

        #change
        return self.fall() #falls, whether it died

    def fall(self):
        # 1 point (half a heart) for each block of fall distance after the third
        x, y, z = self.agent_xyz()
        #print(x,y,z)
        fall_through = ["air", "lava", "flowing_lava", "water", "flowing_water"]
        #
        
        while (True):
            x, y, z = self.agent_xyz()

            #change to y+1
            #print("testing",self.at(x, y +2, z))

            #hit bottom of world
            if not self.boundary_check(x, y - 1, z):
                break
            
            
            if (self.at(x, y - 1, z) in fall_through):
                # import time
                # time.sleep(0.5)
                #print("fall: block under: ", self.at(x, y - 1, z))
                #change to height +=1
                self.agent.height -= 1
                #CHANGE
                #y = starting_height - self.agent.height
                #if (self.agent_death()):
                    #return True
            else:  # or hits bottom of the world
                break

        return False

    def agent_death(self):
        x, y, z = self.agent_xyz()

        danger = ["lava", "flowing_lava"]

        return (any(self.at(x, y, z) == i for i in danger))


if __name__ == "__main__":
    # terrain_data = unpickle("terrain_data.pck")
    # starting_height = 70 # TODO: Change later to proper starting height using terrain_data.pck

    # # Pre-process: convert unimportant blocks to stone
    # all_blocks = set(np.unique(terrain_data))
    # for i in all_blocks - EXCLUSION:
    #     terrain_data[terrain_data==i] = "stone"

    # Smaller Scale World
    # 3 layers air, 5 layers stone, 1 layer gold, 5 layers stone, 1 layer diamond, and 5 layers stones
    num_epi = 100000
    max_step = 200
    moving_penatly = 1
    mining_penatly = 2

    MINING_REWARD = 0
    MOVING_REWARD = -10

    lr = 0.1                #learning rate
    discount_rate = 0.99 
    epsilon = 0.2


    reward_track = []        #store reward for each epsiode

    #
    terrain_data, terrain_height = helper.create_custom_world(50, 50, [(3, "air"), (5, "stone") ,(2,"diamond_ore")])
    
    starting_height = terrain_height - 1

    file = open("results.txt", 'w')
    
    move= ["N","S","W","E","U","M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]
    q_table = {}


    diamond_sum  = 0
    for i in range(num_epi):
        s = Simulation(terrain_data,starting_height)

        steps = max_step

        state = s.get_current_state()

        #Every 100th, show best policy
        ep = epsilon if i % 100 else 0

        d = 0                         #track if death in epsidoe
        while steps > 0:
           
            # time.sleep(1.5)
            #print("state", state,"step",steps)
            """
            consider pass state as a parameter to the choose move to reduce run time
            """

            action = s.choose_move(ep, state, q_table)
            #print("action is",action)

            #action
            """
            I assume getReward call the function to "change the map" or move the agent
            """
            reward, d = s.get_reward(state, action)

            if not action.startswith('M_'):
                reward += MOVING_REWARD

            steps = steps - mining_penatly if action.startswith('M_') else steps - moving_penatly

            #get new state
            new_state = s.get_current_state()
            #update q table
            index = move.index(action)

            if new_state not in q_table.keys():
                q_table[new_state] = [0 for i in range(len(move))]

            if state in q_table.keys():
                """
                check equation right or not
                """
                q_table[state][index] = (1-lr) * q_table[state][index] + lr * (reward + discount_rate*max(q_table[new_state]))
            else:
                q_table[state] = [0 for _ in range(len(move))]
                q_table[state][index] = lr * (reward + discount_rate * max(q_table[new_state]))


            state = new_state

            #break if meet lava
            if d:
                break

        total_reward = 0

        #print(s.agent.inventory)
        diamonds_mined = s.agent.inventory["diamond_ore"]

        diamond_sum += diamonds_mined

        while s.agent.inventory:
            k,v = s.agent.inventory.popitem()
            if k in REWARD_TABLE.keys():
                total_reward += REWARD_TABLE[k]*v

        reward_track.append(total_reward)

        if i % 100 == 0 and i != 0:
            print("Episode:", i)
            print("\tQ_TABLE:", len(q_table))
            print("\tAverage diamonds mined:", diamond_sum / 100)
            print("\tBest policy: ", diamonds_mined) 

            # episode, q_table, avg_diamonds
            file.write(f"{i}, {len(q_table)}, {diamond_sum / 100}, {diamonds_mined}\n")

            #print("\tQ_TABLE:", len(q_table))
            diamond_sum = 0


    file.close()