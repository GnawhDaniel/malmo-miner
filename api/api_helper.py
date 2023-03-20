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

def get_current_state(y,x,z,height,terrain_data) -> "state":
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
        for i in range(5):
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
        block = "air+"
        #print(temp_see)
        for i in REWARD_TABLE:
            if i in temp_see:
                block += i
                break
        if len(block) == 4:
            block += temp_see[0]

        state_space.append(block)

    # North upper
    y+=1
    if terrain_data[y, x - 1, z] != "air":
        state_space.append(terrain_data[y, x - 1, z])
    else:
        temp_x = x
        temp_see = []
        for i in range(5):
            temp_x -= 1
            if terrain_data[y, temp_x, z] != "air":
                temp_see.append(terrain_data[y, temp_x, z])
                break
            else:
                temp_see.append(terrain_data[y + 1, temp_x, z])
                temp_see.append(terrain_data[y - 1, temp_x, z])
                temp_see.append(terrain_data[y, temp_x, z + 1])
                temp_see.append(terrain_data[y, temp_x, z - 1])
        block = "air+"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += i
                break
        if len(block) == 4:
            block += temp_see[0]

        state_space.append(block)
    y-=1

    # East lower
    if terrain_data[y, x, z + 1] != "air":
        state_space.append(terrain_data[y, x, z + 1])
    else:
        temp_z = z
        temp_see = []
        for i in range(5):
            temp_z += 1
            if terrain_data[y, x, temp_z] != "air":
                temp_see.append(terrain_data[y, x, temp_z])
                break
            else:
                temp_see.append(terrain_data[y + 1, x, temp_z])
                temp_see.append(terrain_data[y - 1, x, temp_z])
                temp_see.append(terrain_data[y, x + 1, temp_z])
                temp_see.append(terrain_data[y, x - 1, temp_z])
        block = "air+"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += i
                break
        if len(block) == 4:
            block += temp_see[0]

        state_space.append(block)

    # East upper
    y += 1
    if terrain_data[y, x, z + 1] != "air":
        state_space.append(terrain_data[y, x, z + 1])
    else:
        temp_z = z
        temp_see = []
        for i in range(5):
            temp_z += 1
            if terrain_data[y, x, temp_z] != "air":
                temp_see.append(terrain_data[y, x, temp_z])
                break
            else:
                temp_see.append(terrain_data[y + 1, x, temp_z])
                temp_see.append(terrain_data[y - 1, x, temp_z])
                temp_see.append(terrain_data[y, x + 1, temp_z])
                temp_see.append(terrain_data[y, x - 1, temp_z])
        block = "air+"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += i
                break
        if len(block) == 4:
            block += temp_see[0]

        state_space.append(block)
    y -= 1

    # South lower
    if terrain_data[y, x + 1, z] != "air":
        state_space.append(terrain_data[y, x + 1, z])
    else:
        temp_x = x
        temp_see = []
        for i in range(5):
            temp_x += 1
            if terrain_data[y, temp_x, z] != "air":
                temp_see.append(terrain_data[y, temp_x, z])
                break
            else:
                temp_see.append(terrain_data[y + 1, temp_x, z])
                temp_see.append(terrain_data[y - 1, temp_x, z])
                temp_see.append(terrain_data[y, temp_x, z + 1])
                temp_see.append(terrain_data[y, temp_x, z - 1])
        block = "air+"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += i
                break
        if len(block) == 4:
            block += temp_see[0]

        state_space.append(block)

    # South upper
    y += 1
    if terrain_data[y, x + 1, z] != "air":
        state_space.append(terrain_data[y, x + 1, z])
    else:
        temp_x = x
        temp_see = []
        for i in range(5):
            temp_x += 1
            if terrain_data[y, temp_x, z] != "air":
                temp_see.append(terrain_data[y, temp_x, z])
                break
            else:
                temp_see.append(terrain_data[y + 1, temp_x, z])
                temp_see.append(terrain_data[y - 1, temp_x, z])
                temp_see.append(terrain_data[y, temp_x, z + 1])
                temp_see.append(terrain_data[y, temp_x, z - 1])
        block = "air+"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += i
                break
        if len(block) == 4:
            block += temp_see[0]
    y -= 1

    state_space.append(block)



    # West lower
    if terrain_data[y, x, z - 1] != "air":
        state_space.append(terrain_data[y, x, z - 1])
    else:
        temp_z = z
        temp_see = []
        for i in range(5):
            temp_z -= 1
            if terrain_data[y, x, temp_z] != "air":
                temp_see.append(terrain_data[y, x, temp_z])
                break
            else:
                temp_see.append(terrain_data[y + 1, x, temp_z])
                temp_see.append(terrain_data[y - 1, x, temp_z])
                temp_see.append(terrain_data[y, x + 1, temp_z])
                temp_see.append(terrain_data[y, x - 1, temp_z])
        block = "air+"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += i
                break
        if len(block) == 4:
            block += temp_see[0]

        state_space.append(block)

    # West upper
    y += 1
    if terrain_data[y, x, z - 1] != "air":
        state_space.append(terrain_data[y, x, z - 1])
    else:
        temp_z = z
        temp_see = []
        for i in range(5):
            temp_z -= 1
            if terrain_data[y, x, temp_z] != "air":
                temp_see.append(terrain_data[y, x, temp_z])
                break
            else:
                temp_see.append(terrain_data[y + 1, x, temp_z])
                temp_see.append(terrain_data[y - 1, x, temp_z])
                temp_see.append(terrain_data[y, x + 1, temp_z])
                temp_see.append(terrain_data[y, x - 1, temp_z])
        block = "air+"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += i
                break
        if len(block) == 4:
            block += temp_see[0]

        state_space.append(block)
    y -= 1

    #UPPER
    if terrain_data[y+2, x, z] != "air":
        state_space.append(terrain_data[y+2, x, z])
    else:
        temp_y = y
        temp_see = []
        for i in range(5):
            temp_y += 1
            if terrain_data[temp_y, x, z] != "air":
                temp_see.append(terrain_data[temp_y, x, z])
                break
            else:
                temp_see.append(terrain_data[temp_y, x, z+1])
                temp_see.append(terrain_data[temp_y , x, z-1])
                temp_see.append(terrain_data[temp_y, x + 1, z])
                temp_see.append(terrain_data[temp_y, x - 1, z])
        block = "air+"
        for i in REWARD_TABLE:
            if i in temp_see:
                block += i
                break
        if len(block) == 4:
            block += temp_see[0]

        state_space.append(block)



    # BELOW

    state_space.append(terrain_data[y - 1,x, z])  # below feet

    # HEIGHT
    state_space.append(height)

    return state_space

