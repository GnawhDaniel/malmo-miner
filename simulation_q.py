import numpy as np
import world_data_extractor
import random, copy
from helper import pickilizer, unpickle
from operator import add
import helper


#
starting_height = 70  # CHANGE LATER

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

        from collections import defaultdict
        self.q_table = {}

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
            self.z -= 1
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
            if not state[i].startswith('air'):
                moves.append(M[i])

        return moves




class Simulation:
    def __init__(self, terrain_data, starting_height) -> None:
        self.terrain_data = copy.deepcopy(terrain_data)
        self.starting_height = starting_height

        self.agent_mined = set()
        self.agent_placed = set()

        self.agent = Agent(int(terrain_data.shape[1] / 2), starting_height, int(terrain_data.shape[2] / 2))

    def run(self):
        self.fall()  # make agent touch the ground
        # do the simulation
        # q learning?
        return  # ?

    def boundary_check(self, x, y, z) -> bool:
        """
        Returns: True if in bounds
        """
        world_shape = self.terrain_data.shape  # (y, x, z)
        if x < 0 or x > world_shape[1]:
            return False
        if z < 0 or z > world_shape[2]:
            return False
        if y < 5 or y > starting_height:
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
        # starting_height - self.agent.height
        return self.agent.x, self.agent.height, self.agent.z
    #change4
    def choose_move(self, epsilon,state):
        possible_moves = self.agent.get_possible_moves(state)
        if (random.random() < epsilon or state not in self.agent.q_table.keys()):
            return  possible_moves[random.randint(0, len(possible_moves) - 1)]
        else:
            """
            careful when add new move as the index will be wrong
            """
            move = ["N", "S", "W", "E", "U", "M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U",
                    "M_D"]
            max_q_value = max(self.agent.q_table[state])
            best_move = [move[i] for i, x in enumerate(self.agent.q_table[state]) if x == max_q_value]
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
        # print(x,y,z)
        '''
        (NL, NU, EL, EU, SL, SU, WL, WU, U, D, height)
        '''
        # X, Z
        # dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dir = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        state_space = []

        # Searching Range
        r_distance = 5

        for d in dir:
            lower = self.at(x + d[0], y, z + d[1])
            upper = self.at(x + d[0], y + 1, z + d[1])
            # ('air+diamond_ore', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 50))

            search_direction = (d[0], 0, d[1])

            if lower == "air":  # lower
                coord = (x + d[0], y, z + d[1])
                max_block = self.recursive_search(coord, search_direction, r_distance)

                if max_block in REWARD_TABLE:
                    lower += "+" + max_block

            if upper == "air":  # upper
                coord = (x + d[0], y + 1, z + d[1])
                max_block = self.recursive_search(coord, search_direction, r_distance)

                if max_block in REWARD_TABLE:
                    upper += "+" + max_block

            state_space.append(lower)  # Lower
            state_space.append(upper)  # Upper

        state_space.append(self.at(x, y - 1, z))  # below feet

        above = self.at(x, y + 2, z)  # above head
        if above == "air":
            max_block = self.recursive_search(coord, (0, 1, 0), r_distance)

            if max_block in REWARD_TABLE:
                lower += "+" + max_block

        state_space.append(above)
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

    def getReward(self, state, action):
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
            # Unpack tuple and mine the block out
            self.mine(*temp)
            block_mined = self.at(*temp)
            self.agent.inventory[block_mined] += 1

            reward = 0
            dead = False

            if block in REWARD_TABLE:
                reward = REWARD_TABLE[block_mined]

            # IF lava in the state space and doesn't move
            if (any(map(lambda x: x == "lava" or x == "flowing_lava", state))):
                reward += DEATH_VALUE
                dead = True

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
                t1 = self.agent_xyz()
                t2 = relative_coord
                new_coord = tuple(t1[i]-t2[i] for i in range(len(t1)))
                if self.at(*new_coord) == 'air' or self.is_mined(*new_coord):
                    self.place_block(*new_coord)

            # Move the agent, return if died or not
            #print(coords[M.index(action)])
            # change1
            #print("test1",relative_coord)
            if self.agent_move(*relative_coord):  # if died
                print("test2")
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
        self.agent.x += x
        self.agent.height += y
        self.agent.x += z
        #print("cord2", self.agent.x, self.agent.height, self.agent.z)

        #change
        return self.fall()

    def fall(self):
        # 1 point (half a heart) for each block of fall distance after the third
        x, y, z = self.agent_xyz()
        print(x,y,z)
        fall_through = ["air", "lava", "flowing_lava", "water", "flowing_water"]

        while (True):
            #change to y+1
            if (any(self.at(x, y +1, z) == i for i in fall_through)):
                #change to height +=1
                self.agent.height += 1
                print("fall",self.get_current_state())
                y = starting_height - self.agent.height
                if (self.agent_death()):
                    return True
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
    num_epi = 1000
    max_step = 200
    moving_penatly = 1
    mining_penatly = 2

    lr = 0.1                #learning rate
    discount_rate = 0.99
    epsilon = 0.2


    reward_track =[]        #store reward for each epsiode

    #
    terrain_data = helper.create_custom_world(50, 50, [(3, "air"), (30, "stone"),(30,"diamond_ore")])
    starting_height = 1
    s = Simulation(terrain_data,starting_height)
    move= ["N","S","W","E","U","M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]
    for i in range(num_epi):
        steps = max_step

        state = s.get_current_state()

        d = 0                         #track if death in epsidoe
        while steps > 0:
            print("state", state,"step",steps)
            """
            consider pass state as a parameter to the choose move to reduce run time
            """
            action = "N"#s.choose_move(epsilon,state)
            print(action)

            #action
            """
            I assume getReward call the function to "change the map" or move the agent
            """
            reward, d = s.getReward(state, action)

            steps = steps - mining_penatly if action.startswith('M_') else steps - moving_penatly

            #get new state
            new_state = s.get_current_state()
            print("new_state",new_state)
            #update q table
            index = move.index(action)

            if new_state not in s.agent.q_table.keys():
                s.agent.q_table[new_state] = [0 for i in range(len(move))]

            if state in s.agent.q_table.keys():
                """
                check equation right or not
                """
                s.agent.q_table[state][index] = (1-lr) * s.agent.q_table[state][index] + lr * (reward + discount_rate*max(s.agent.q_table[new_state]))
            else:
                s.agent.q_table[state] = [0 for i in range(len(move))]
                s.agent.q_table[state][index] = lr * (reward + discount_rate * max(s.agent.q_table[new_state]))


            state = new_state

            #break if meet lava
            if d:
                break

        total_reward = 0

        while s.agent.inventory:
            k,v = s.agent.inventory.popitem()
            if k in REWARD_TABLE.keys():
                total_reward += REWARD_TABLE[i]*v

        reward_track.append(total_reward)

        """
        
        """
        s.agent_mined = set()
        s.agent_placed = set()
        s.agent.x = int(terrain_data.shape[1] / 2)
        s.agent.height = starting_height
        s.agent.z= int(terrain_data.shape[2] / 2)




