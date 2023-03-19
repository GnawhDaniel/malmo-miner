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

def get_current_state(x,y,z,height,terrain_data) -> "state":
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
    r_distance = 5

    # NESW
    for d in dir:
        lower_coord = (x + d[0], y, z + d[1])
        lower = terrain_data[y, x, z]
        search_direction = (d[0], 0, d[1])

        if lower == "air":  # lower
            # coord = (x + d[0], y, z + d[1])
            max_block = recursive_search(lower_coord, search_direction, r_distance)

            if max_block in REWARD_TABLE:
                lower += "+" + max_block

        state_space.append(lower)  # Lower

        upper_coord = (x + d[0], y + 1, z + d[1])
        # print(upper_coord)

        upper = terrain_data[x + d[0], y + 1, z + d[1]]

        if upper == "air":  # upper
            # coord = (x + d[0], y + 1, z + d[1])
            max_block = recursive_search(upper_coord, search_direction, r_distance)

            if max_block in REWARD_TABLE:
                upper += "+" + max_block

        state_space.append(upper)  # Upper

    # ABOVE
    above = terrain_data[x, y + 2, z]  # above head
    if above == "air":
        max_block = recursive_search((x, y + 2, z), (0, 1, 0), r_distance)

        if max_block in REWARD_TABLE:
            lower += "+" + max_block

    state_space.append(above)

    # BELOW
    state_space.append(terrain_data[x, y - 1, z])  # below feet

    # HEIGHT
    state_space.append(height)

    return state_space


def recursive_search(coordinate, direction, search_depth):
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