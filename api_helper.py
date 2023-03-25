from helper import REWARD_TABLE, EXCLUSION
import helper
import numpy as np
import random
from operator import add

HEIGHT_MOD = 1

#ONE HOT ENCODING
BLOCK_MAP, ACTION_MAP = helper.enumerate_one_hot()
ALL_MOVES = ["N","S","W","E","U","M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]

class BestPolicy:
    def __init__(self, ddqn) -> None:
        self.ddqn = ddqn

    def choose_random_move(self, state):
        possible_moves = self.get_possible_moves(state)

        return random.choice(possible_moves)
    

    def choose_move(self, state):
        possible_moves = self.get_possible_moves(state)

        height = state[-1] // HEIGHT_MOD
        prev_move = ACTION_MAP[state[-2]]
        state = [BLOCK_MAP[s] for s in state[:-2]]
        state.append(prev_move)
        state.append(height)
        state = np.array(state, dtype=np.float32)
        state = state[np.newaxis, :]

        actions = self.ddqn.q_eval.predict(state, verbose=0)[0]
        
        sorted_actions = np.flip(np.argsort(actions))

        #choose best move only if move is possible in current state
        for action in sorted_actions:
            if ALL_MOVES[action] in possible_moves:
                last_move = ALL_MOVES[action]
                return last_move
            
        raise

    def get_possible_moves(self, state):
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
            if not state[i].startswith('air') and state[i] not in helper.NOT_MINEABLE:
                moves.append(M[i])
        return moves

def get_current_state(y,x,z, height, last_move, terrain_data) -> "state":
    """
    The x-axis indicates the player's distance east (positive)  or west (negative) of the origin point—i.e., the longitude,
    The z-axis indicates the player's distance south (positive) or north (negative) of the origin point—i.e., the latitude
    """

    '''
    (NL, NU, EL, EU, SL, SU, WL, WU, U, D, height)
    '''
    # Searching Range
    r_distance = 5
    state_space = []
    #N E S W
    dir = [(-1, 0), (0, 1), (1, 0), (0, -1)] # (x, z)

    
    def recursive_search(coordinate, direction, search_depth, terrain_data):
        '''
        recursively finds max value block
        returns str: 'max_value_block'
        '''

        coordinate = tuple(coordinate)
        # base cases:
        # Search depth exceeded
        if search_depth <= 0:
            return "air"
        
        if terrain_data[coordinate] != "air":
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
                blocks.append(terrain_data[c])

        for i in blocks:
            if (i in REWARD_TABLE):
                return "diamond_ore"

        # find max of other blocks recursively
        max_recurred = recursive_search(map(add, coordinate, direction), direction, search_depth - 1, terrain_data)

        return max_recurred


    for direction in dir:
        for y_ in [y, y+1]:
            if terrain_data[y_,x + direction[0],z + direction[1]] != "air":
                state_space.append(terrain_data[y_,x + direction[0],z + direction[1]])
            else:
                block_found = recursive_search((y_,x + direction[0],z + direction[1]), (0, direction[0], direction[1]), r_distance, terrain_data)
                        
                block = "air"
                #print(temp_see)
                if block_found in REWARD_TABLE:
                    block += "+ore"
            
                state_space.append(block)

    #UP
    if terrain_data[y+2, x, z] != "air":
        state_space.append(terrain_data[y+2, x, z])
    else:
        # coordinate, direction, search_depth, terrain_data
        block_found = recursive_search((y+2, x, z), (1, 0, 0), r_distance, terrain_data)
                        
        block = "air"
        #print(temp_see)
        if block_found in REWARD_TABLE:
            block += "+ore"
    
        state_space.append(block)

    # BELOW

    state_space.append(terrain_data[y - 1,x, z])  # below feet

    state_space.append(last_move)

    # HEIGHT
    state_space.append(height)

    return state_space

