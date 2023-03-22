import numpy as np
#import world_data_extractor
import random, copy
from helper import pickilizer, unpickle
from operator import add
import helper
from collections import defaultdict



class Agent:
    # Constants      
    REWARD_TABLE = helper.REWARD_TABLE
    NOT_MINEABLE = {"air", "lava", "flowing_lava", "water", "flowing_water", "bedrock"}
    EXCLUSION = NOT_MINEABLE.union(set(REWARD_TABLE.keys()))
    DEATH_VALUE = -1000
    MOVE_PENALTY = -5

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

    
        # (NL, NU, EL, EU, SL, SU, WL, WU, U, D, prev_move, height)

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
        for i in range(len(state) - 2):
            if not state[i].startswith('air') and state[i] not in Agent.NOT_MINEABLE:
                moves.append(M[i])
        return moves


class Simulation:
    def __init__(self, terrain_data, starting_height) -> None:
        self.terrain_data = copy.deepcopy(terrain_data)
        self.starting_height = starting_height

        self.closest_diamond = None

        self.agent_mined = set()
        self.agent_placed = set()
        
        self.last_move = "N"
        # self.diamonds_mined = 0

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
        if y < 0 or y >= self.starting_height:
            return False
        
        return True
  
    def at(self, x, y, z):
        if self.boundary_check(x, y, z):
            if self.is_placed(x, y, z):
                return "stone"

            if self.is_mined(x, y, z):
                return "air"
            return self.terrain_data[y, x, z]
        else:
            return "bedrock"

    def agent_xyz(self):
        # starting_height - self.agent.heightcle
        return self.agent.x, self.agent.height, self.agent.z
    
    #change4
    def choose_move(self, epsilon, state, q_table):
        possible_moves = self.agent.get_possible_moves(state)
        #EPSILON
        if (random.random() < epsilon or state not in q_table.keys()):
            return [random.randint(0, len(possible_moves) - 1)]
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
        The x-axis indicates the player's distance east (positive)  or west (negative) of the origin point-i.e., the longitude,
        The z-axis indicates the player's distance south (positive) or north (negative) of the origin point-i.e., the latitude
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

                    if max_block in Agent.REWARD_TABLE:
                        lower += "+" + "ore"# max_block
                
                state_space.append(lower)  # Lower


            upper_coord = (x + d[0], y + 1 , z + d[1])
            if not self.boundary_check(*upper_coord):
                state_space.append("bedrock")
            else:
                upper = self.at(x + d[0], y + 1, z + d[1])

                if upper == "air":  # upper
                    # coord = (x + d[0], y + 1, z + d[1])
                    max_block = self.recursive_search(upper_coord, search_direction, r_distance)

                    if max_block in Agent.REWARD_TABLE:
                        upper += "+" + "ore"# max_block

                state_space.append(upper)  # Upper

        
        #ABOVE
        if not self.boundary_check(x, y + 2, z):
            state_space.append("bedrock")
        else:  
            above = self.at(x, y + 2, z)  # above head
            if above == "air":
                max_block = self.recursive_search((x, y + 2, z), (0, 1, 0), r_distance)

                if max_block in Agent.REWARD_TABLE:
                    lower += "+" + "ore"# max_block

            state_space.append(above)   


        #BELOW
        if not self.boundary_check(x, y - 1, z):
            state_space.append("bedrock")
        else:
            state_space.append(self.at(x, y - 1, z))   # below feet
        
        #last_move
        state_space.append(self.last_move)

        #HEIGHT
        state_space.append(self.agent.height)

        return state_space

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
                if (max_local not in Agent.REWARD_TABLE):
                    max_local = i
                elif (i in Agent.REWARD_TABLE):
                    if Agent.REWARD_TABLE[i] > Agent.REWARD_TABLE[max_local]:
                        max_local = i

        # find max of other blocks recursively
        max_recurred = self.recursive_search(map(add, coordinate, direction), direction, search_depth - 1)

        if (max_local not in Agent.REWARD_TABLE):
            max_local = max_recurred
        elif (max_recurred in Agent.REWARD_TABLE):
            if Agent.REWARD_TABLE[max_recurred] > Agent.REWARD_TABLE[max_local]:
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
            # Unpack tuple and mine the block out
            
            block_mined = self.at(*temp)
            self.agent.inventory[block_mined] += 1
            self.mine(*temp)

            reward = 0
            dead = False
            
            if block_mined in Agent.REWARD_TABLE: # TODO: changed block to block_mined
                reward = Agent.REWARD_TABLE[block_mined]
            
            """
            FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
                            if type(self.closest_diamond) != type(None) and block_mined == self.closest_diamond:
            """
            if type(self.closest_diamond) != type(None) and not np.any(temp - self.closest_diamond):
                self.closest_diamond = None

            # if block == "diamond_ore":
            #     self.diamonds_mined += 1

            # IF lava in the state space and doesn't move
            if (any(map(lambda x: x == "lava" or x == "flowing_lava", state))):
                dead = True

            #or falling
            died_falling = self.fall()
            dead = dead or died_falling

            if dead:
                reward += Agent.DEATH_VALUE

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
            # change1
            if self.agent_move(*relative_coord):  # if died
                return Agent.DEATH_VALUE, True

            # otherwise return
            return Agent.MOVE_PENALTY, False

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
        self.agent.x += x
        self.agent.height += y
        self.agent.z += z

        #change
        return self.fall() #falls, whether it died

    def fall(self):
        # 1 point (half a heart) for each block of fall distance after the third
        x, y, z = self.agent_xyz()
        fall_through = ["air", "lava", "flowing_lava", "water", "flowing_water"]
        #
        
        while (True):
            x, y, z = self.agent_xyz()

            #hit bottom of world
            if not self.boundary_check(x, y - 1, z):
                break
            
            
            if (self.at(x, y - 1, z) in fall_through):
                self.agent.height -= 1
            else:  
                break

        return False

    def agent_death(self):
        x, y, z = self.agent_xyz()

        danger = ["lava", "flowing_lava"]

        return (any(self.at(x, y, z) == i for i in danger))
