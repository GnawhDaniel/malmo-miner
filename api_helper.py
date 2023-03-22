from helper import REWARD_TABLE, EXCLUSION
import helper
import numpy as np

HEIGHT_MOD = 1

#ONE HOT ENCODING
BLOCK_MAP, ACTION_MAP = helper.enumerate_one_hot()
ALL_MOVES = ["N","S","W","E","U","M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]

class BestPolicy:
    def __init__(self, ddqn) -> None:
        self.ddqn = ddqn

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

def get_current_state(y,x,z,height, last_move, terrain_data) -> "state":
    """
    The x-axis indicates the player's distance east (positive)  or west (negative) of the origin point—i.e., the longitude,
    The z-axis indicates the player's distance south (positive) or north (negative) of the origin point—i.e., the latitude
    """

    '''
    (NL, NU, EL, EU, SL, SU, WL, WU, U, D, height)
    '''
    # X, Z
    # dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    dir = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    state_space = []
    # Searching Range
    r_distance = 6
    #  S
    # W  E
    #  N
    # X=5, Y = 1 , Z = 1

    #North lower
    if terrain_data[y,x-1,z] != "air":
        state_space.append(terrain_data[y,x-1,z])
    else:
        temp_x = x
        temp_see = []
        for i in range(r_distance):
            temp_x -= 1

            if terrain_data[y,temp_x,z] != "air":
                temp_see.append(terrain_data[y,temp_x,z])
                break
            else:
                #print(terrain_data[y + 1,temp_x, z])
                temp_see.append(terrain_data[y + 1,temp_x, z])
                #print(terrain_data[y - 1, temp_x, z])
                temp_see.append(terrain_data[y - 1, temp_x, z])
                temp_see.append(terrain_data[y, temp_x, z+1])
                temp_see.append(terrain_data[y, temp_x, z-1])
        block = "air"
        #print(temp_see)
        for i in REWARD_TABLE:
            if i in temp_see:
                block += "+ore"
                break
 

        state_space.append(block)

    # North upper
    y+=1
    if terrain_data[y, x - 1, z] != "air":
        state_space.append(terrain_data[y, x - 1, z])
    else:
        temp_x = x
        temp_see = []
        for i in range(r_distance):
            temp_x -= 1
            if terrain_data[y, temp_x, z] != "air":
                temp_see.append(terrain_data[y, temp_x, z])
                break
            else:
                temp_see.append(terrain_data[y + 1, temp_x, z])
                temp_see.append(terrain_data[y - 1, temp_x, z])
                temp_see.append(terrain_data[y, temp_x, z + 1])
                temp_see.append(terrain_data[y, temp_x, z - 1])
        block = "air"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += "+ore"
                break


        state_space.append(block)
    y-=1

    # East lower
    if terrain_data[y, x, z + 1] != "air":
        state_space.append(terrain_data[y, x, z + 1])
    else:
        temp_z = z
        temp_see = []
        for i in range(r_distance):
            temp_z += 1
            if terrain_data[y, x, temp_z] != "air":
                temp_see.append(terrain_data[y, x, temp_z])
                break
            else:
                temp_see.append(terrain_data[y + 1, x, temp_z])
                temp_see.append(terrain_data[y - 1, x, temp_z])
                temp_see.append(terrain_data[y, x + 1, temp_z])
                temp_see.append(terrain_data[y, x - 1, temp_z])
        block = "air"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += "+ore"
                break


        state_space.append(block)

    # East upper
    y += 1
    if terrain_data[y, x, z + 1] != "air":
        state_space.append(terrain_data[y, x, z + 1])
    else:
        temp_z = z
        temp_see = []
        for i in range(r_distance):
            temp_z += 1
            if terrain_data[y, x, temp_z] != "air":
                temp_see.append(terrain_data[y, x, temp_z])
                break
            else:
                temp_see.append(terrain_data[y + 1, x, temp_z])
                temp_see.append(terrain_data[y - 1, x, temp_z])
                temp_see.append(terrain_data[y, x + 1, temp_z])
                temp_see.append(terrain_data[y, x - 1, temp_z])
        block = "air"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += "+ore"
                break

        state_space.append(block)
    y -= 1

    # South lower
    if terrain_data[y, x + 1, z] != "air":
        state_space.append(terrain_data[y, x + 1, z])
    else:
        temp_x = x
        temp_see = []
        for i in range(r_distance):
            temp_x += 1
            if terrain_data[y, temp_x, z] != "air":
                temp_see.append(terrain_data[y, temp_x, z])
                break
            else:
                temp_see.append(terrain_data[y + 1, temp_x, z])
                temp_see.append(terrain_data[y - 1, temp_x, z])
                temp_see.append(terrain_data[y, temp_x, z + 1])
                temp_see.append(terrain_data[y, temp_x, z - 1])
        block = "air"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += "+ore"
                break


        state_space.append(block)

    # South upper
    y += 1
    if terrain_data[y, x + 1, z] != "air":
        state_space.append(terrain_data[y, x + 1, z])
    else:
        temp_x = x
        temp_see = []
        for i in range(r_distance):
            temp_x += 1
            if terrain_data[y, temp_x, z] != "air":
                temp_see.append(terrain_data[y, temp_x, z])
                break
            else:
                temp_see.append(terrain_data[y + 1, temp_x, z])
                temp_see.append(terrain_data[y - 1, temp_x, z])
                temp_see.append(terrain_data[y, temp_x, z + 1])
                temp_see.append(terrain_data[y, temp_x, z - 1])
        block = "air"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += "+ore"
                break

        state_space.append(block)
    y -= 1




    # West lower
    if terrain_data[y, x, z - 1] != "air":
        state_space.append(terrain_data[y, x, z - 1])
    else:
        temp_z = z
        temp_see = []
        for i in range(r_distance):
            temp_z -= 1
            if terrain_data[y, x, temp_z] != "air":
                temp_see.append(terrain_data[y, x, temp_z])
                break
            else:
                temp_see.append(terrain_data[y + 1, x, temp_z])
                temp_see.append(terrain_data[y - 1, x, temp_z])
                temp_see.append(terrain_data[y, x + 1, temp_z])
                temp_see.append(terrain_data[y, x - 1, temp_z])
        block = "air"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += "+ore"
                break


        state_space.append(block)

    # West upper
    y += 1
    if terrain_data[y, x, z - 1] != "air":
        state_space.append(terrain_data[y, x, z - 1])
    else:
        temp_z = z
        temp_see = []
        for i in range(r_distance):
            temp_z -= 1
            if terrain_data[y, x, temp_z] != "air":
                temp_see.append(terrain_data[y, x, temp_z])
                break
            else:
                temp_see.append(terrain_data[y + 1, x, temp_z])
                temp_see.append(terrain_data[y - 1, x, temp_z])
                temp_see.append(terrain_data[y, x + 1, temp_z])
                temp_see.append(terrain_data[y, x - 1, temp_z])
        block = "air"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += "+ore"
                break
 

        state_space.append(block)
    y -= 1

    #UPPER
    if terrain_data[y+2, x, z] != "air":
        state_space.append(terrain_data[y+2, x, z])
    else:
        temp_y = y
        temp_see = []
        for i in range(r_distance):
            temp_y += 1
            if terrain_data[temp_y, x, z] != "air":
                temp_see.append(terrain_data[temp_y, x, z])
                break
            else:
                temp_see.append(terrain_data[temp_y, x, z+1])
                temp_see.append(terrain_data[temp_y , x, z-1])
                temp_see.append(terrain_data[temp_y, x + 1, z])
                temp_see.append(terrain_data[temp_y, x - 1, z])
        block = "air"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += "+ore"
                break


        state_space.append(block)



    # BELOW

    state_space.append(terrain_data[y - 1,x, z])  # below feet

    state_space.append(last_move)

    # HEIGHT
    state_space.append(height)

    return state_space

