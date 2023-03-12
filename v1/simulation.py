from agent import Agent
import random, copy
from operator import add
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
MOVE_PENALTY = -2


class Simulation:
    def __init__(self, terrain_data, starting_height) -> None:
        self.terrain_data = copy.deepcopy(terrain_data)
        self.starting_height = starting_height

        self.agent_mined = set()
        self.agent_placed = set()
        self.diamonds_mined = 0

        self.agent = Agent(int(terrain_data.shape[1] / 2), starting_height - 2, int(terrain_data.shape[2] / 2))

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
        if x < 0 or x > world_shape[1] - 1:
            return False
        if z < 0 or z > world_shape[2] - 1:
            return False
        if y < 0 or y > self.starting_height:
            return False

        return True

    def at(self, x, y, z):
        if self.is_placed(x, y, z):
            return "stone"

        if self.is_mined(x, y, z):
            return "air"
        # print("testing",[y, x, z])
        return self.terrain_data[y, x, z]

    def agent_xyz(self):
        # starting_height - self.agent.heightcle
        return self.agent.x, self.agent.height, self.agent.z

    # change4
    def choose_move(self, epsilon, state, clf,mapping):
        possible_moves = self.agent.get_possible_moves(state)
        # EPSILON
        if (random.random() < epsilon):
            return possible_moves[random.randint(0, len(possible_moves) - 1)]
        else:

            """
            careful when add new move as the index will be wrong
            """
            t = list(state)
            X = [t + [i] for i in possible_moves]
            X = self.convert(X,mapping)



            q_table = clf.predict(X)
            best_move = [AR[0] for AR in list(zip(possible_moves, q_table)) if AR[1] == max(q_table)]

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

        # NESW
        for d in dir:
            lower_coord = (x + d[0], y, z + d[1])
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

            upper_coord = (x + d[0], y + 1, z + d[1])
            # print(upper_coord)
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

        # ABOVE
        if not self.boundary_check(x, y + 2, z):
            state_space.append("bedrock")
        else:
            above = self.at(x, y + 2, z)  # above head
            if above == "air":
                max_block = self.recursive_search((x, y + 2, z), (0, 1, 0), r_distance)

                if max_block in REWARD_TABLE:
                    lower += "+" + max_block

            state_space.append(above)

            # BELOW
        if not self.boundary_check(x, y - 1, z):
            state_space.append("bedrock")
        else:
            state_space.append(self.at(x, y - 1, z))  # below feet

        # HEIGHT
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
    def convert(self,SA,mapping):
        #print("called")
        for sample in SA:
            for i in range(len(sample)):
                temp = sample[i]
                if type(temp) == int:
                    pass
                else:
                    if temp not in mapping:
                        mapping.append(temp)
                    sample[i] = mapping.index(temp) + 1000
        SA = np.asarray(SA)
        return SA

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
            # change3
            temp = [i for i in coord]
            # print(action,temp)
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

            # or falling
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
                # change2
                c = self.agent_xyz()
                if self.at(*c) == 'air' or self.is_mined(*c):
                    self.place_block(*c)

            # Move the agent, return if died or not
            # print(coords[M.index(action)])
            # change1
            # print("test1",relative_coord)
            if self.agent_move(*relative_coord):  # if died
                # print("test2")
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
        # print("cord1", self.agent.x, self.agent.height, self.agent.z)
        # print("d",x,y,z)
        self.agent.x += x
        self.agent.height += y
        self.agent.x += z
        # print("cord2", self.agent.x, self.agent.height, self.agent.z)
        # print(self.get_current_state())

        # change
        return self.fall()  # falls, whether it died

    def fall(self):
        # 1 point (half a heart) for each block of fall distance after the third
        x, y, z = self.agent_xyz()
        # print(x,y,z)
        fall_through = ["air", "lava", "flowing_lava", "water", "flowing_water"]
        #

        while (True):
            x, y, z = self.agent_xyz()

            # change to y+1
            # print("testing",self.at(x, y +2, z))

            # hit bottom of world
            if not self.boundary_check(x, y - 1, z):
                break

            if (self.at(x, y - 1, z) in fall_through):
                # import time
                # time.sleep(0.5)
                # print("fall: block under: ", self.at(x, y - 1, z))
                # change to height +=1
                self.agent.height -= 1
                # CHANGE
                # y = starting_height - self.agent.height
                # if (self.agent_death()):
                # return True
            else:  # or hits bottom of the world
                break

        return False

    def agent_death(self):
        x, y, z = self.agent_xyz()

        danger = ["lava", "flowing_lava"]

        return (any(self.at(x, y, z) == i for i in danger))
