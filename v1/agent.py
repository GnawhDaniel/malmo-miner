from collections import defaultdict


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
