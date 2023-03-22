import MalmoPython
import os
import sys
import time
import random
import json
from api_helper import BestPolicy
import numpy as np
from api_helper import get_current_state,EXCLUSION
from model import DDQN

#change N/S, check wheter E/w is correct
def move(r_move,agent_host):
    sleep_time = 1
    if r_move == "N":
        agent_host.sendCommand("move -1")
        time.sleep(sleep_time)
    if r_move == "S":
        agent_host.sendCommand("move 1")
        time.sleep(sleep_time)
    if r_move == "W":
        agent_host.sendCommand("strafe 1")
        time.sleep(sleep_time)
    if r_move == "E":
        agent_host.sendCommand("strafe -1")
        time.sleep(sleep_time)
    if r_move == "U":
        agent_host.sendCommand("jump")
        agent_host.sendCommand("jump")
        time.sleep(1)
        # this is not working
        agent_host.sendCommand("look 1")
        agent_host.sendCommand("look 1")
        agent_host.sendCommand("hotbar.2 1")
        #check height
        world_state = agent_host.getWorldState()
        msg = world_state.observations[-1].text
        observations = json.loads(msg)
        agent_y = observations.get("YPos", None)
        new_height = agent_y
        count = 0
        while new_height == agent_y and count<=20:
            agent_host.sendCommand("jump")
            agent_host.sendCommand("jump")
            agent_host.sendCommand("use 1")
            time.sleep(0.5)
            world_state = agent_host.getWorldState()
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            new_height = observations.get("YPos", None)
            count+=1

        agent_host.sendCommand("look -1")
        agent_host.sendCommand("look -1")

def random_move():
    ALL_MOVES = ["N", "S", "W", "E", "U"]
    return ALL_MOVES[random.randint(0,len(ALL_MOVES)-1)]
def random_min():
    ALL_MOVES = ["M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]
    return ALL_MOVES[random.randint(0, len(ALL_MOVES) - 1)]




def turn(direction, agent_host):
    #neg = int(degrees / abs(degrees))
    thresh = 0.5
    
    coord = {
        "South": 0,
        "East": 270,
        "West": 90,
        "North": 180,
    }

    world_state = agent_host.peekWorldState()
    pp_text = world_state.observations[-1].text
    pp_dict = json.loads(pp_text)
    
    if pp_dict['Yaw'] - coord[direction] < 0:
        agent_host.sendCommand("turn 0.25")
    else:
        agent_host.sendCommand("turn -0.25")

    while True:
        world_state = agent_host.peekWorldState()
          # most horrible api in the galaxy award
        obs_dict = None
        if world_state.is_mission_running and world_state.number_of_observations_since_last_state > 0:
            obs_text = world_state.observations[-1].text
            obs_dict = json.loads(obs_text)
        
        yaw = obs_dict['Yaw'] if obs_dict is not None else None
        print(yaw)
        if yaw != None and abs(coord[direction] - yaw) < thresh:
            break
    
    agent_host.sendCommand("turn 0")

def pitch(degrees, agent_host):
    #neg = int(degrees / abs(degrees))
    thresh = 2

    world_state = agent_host.peekWorldState()
    pp_text = world_state.observations[-1].text
    pp_dict = json.loads(pp_text)
    
    if pp_dict['Pitch'] - degrees < 0:
        agent_host.sendCommand("pitch 0.3")
    else:
        agent_host.sendCommand("pitch -0.3")

    while True:
        world_state = agent_host.peekWorldState()
          # most horrible api in the galaxy award
        obs_dict = None
        if world_state.is_mission_running and world_state.number_of_observations_since_last_state > 0:
            obs_text = world_state.observations[-1].text
            obs_dict = json.loads(obs_text)
        
        pitch = obs_dict['Pitch'] if obs_dict is not None else None
        print(pitch)
        # 90 is straight down
        # -90 up
        if pitch != None and abs(degrees - pitch) < thresh:
            break
    
    agent_host.sendCommand("pitch 0")

def pitch(degrees, agent_host):
    #neg = int(degrees / abs(degrees))
    thresh = 2

    world_state = agent_host.peekWorldState()
    pp_text = world_state.observations[-1].text
    pp_dict = json.loads(pp_text)
    
    if pp_dict['Pitch'] - degrees < 0:
        agent_host.sendCommand("pitch 0.3")
    else:
        agent_host.sendCommand("pitch -0.3")

    while True:
        world_state = agent_host.peekWorldState()
          # most horrible api in the galaxy award
        obs_dict = None
        if world_state.is_mission_running and world_state.number_of_observations_since_last_state > 0:
            obs_text = world_state.observations[-1].text
            obs_dict = json.loads(obs_text)
        
        pitch = obs_dict['Pitch'] if obs_dict is not None else None
        print(pitch)
        # 90 is straight down
        # -90 up
        if pitch != None and abs(degrees - pitch) < thresh:
            break
    
    agent_host.sendCommand("pitch 0")



def mine(r_move,agent_host):
    print("HERE")
    sleep_time = 0.5
    agent_host.sendCommand("hotbar.1 0")
    #mine down
    if r_move == 'M_D':
        agent_host.sendCommand("setPitch 90")
        time.sleep(sleep_time)
        agent_host.sendCommand("attack 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("setPitch 0")
        time.sleep(sleep_time)
    #mine south
    if r_move == "M_SU":
        agent_host.sendCommand("attack 1")
        time.sleep(sleep_time)
    if r_move == "M_SL":
        # agent_host.sendCommand("look 1")
        agent_host.sendCommand("setPitch 60")
        time.sleep(sleep_time)
        agent_host.sendCommand("attack 1")
        time.sleep(sleep_time)
        # agent_host.sendCommand("look -1")
        agent_host.sendCommand("setPitch 0")
    #mine north
    if r_move == "M_NU":
        agent_host.sendCommand("turn 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("turn 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("attack 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("turn 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("turn 1")
        time.sleep(sleep_time)
    if r_move == "M_NL":
        agent_host.sendCommand("turn 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("turn 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("setPitch 60")
        time.sleep(sleep_time)
        agent_host.sendCommand("attack 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("setPitch 0")
        time.sleep(sleep_time)
        agent_host.sendCommand("turn 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("turn 1")
        time.sleep(sleep_time)
    #mine west The x-axis indicates the player's distance east (positive)  or west (negative) of the origin point—i.e., the longitude,
    if r_move == "M_WU":
        agent_host.sendCommand("turn 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("attack 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("turn -1")
        time.sleep(sleep_time)
    if r_move == "M_WL":
        agent_host.sendCommand("turn 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("setPitch 60")
        time.sleep(sleep_time)
        agent_host.sendCommand("attack 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("setPitch 0")
        time.sleep(sleep_time)
        agent_host.sendCommand("turn -1")
        time.sleep(sleep_time)
    #mine east
    if r_move == "M_EU":
        agent_host.sendCommand("turn -1")
        time.sleep(sleep_time)
        agent_host.sendCommand("attack 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("turn 1")
        time.sleep(sleep_time)
    if r_move == "M_EL":
        agent_host.sendCommand("turn -1")
        time.sleep(sleep_time)
        agent_host.sendCommand("setPitch 60")
        time.sleep(sleep_time)
        agent_host.sendCommand("attack 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("setPitch 0")
        time.sleep(sleep_time)
        agent_host.sendCommand("turn 1")
        time.sleep(sleep_time)
    #mine up
    if r_move == "M_U":
        agent_host.sendCommand("setPitch -90")
        time.sleep(sleep_time)
        agent_host.sendCommand("attack 1")
        time.sleep(sleep_time)
        agent_host.sendCommand("setPitch 0")
        time.sleep(sleep_time)

def make_move(r_move,agent_host):
    if r_move[0] == "M":
        mine(r_move,agent_host)
    else:
        move(r_move, agent_host)


def testing(agent_host):
    while 1:
        c = input("command")
        agent_host.sendCommand(c)
        time.sleep(1)


#SUPER FLAT VERSION
#<FlatWorldGenerator generatorString="3;7,5*1,2*56,3*1,3,2;1;"/>
#<DefaultWorldGenerator/>

# Helper function to get the XML string for the mission
def get_mission_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                    <Summary>World data extractor</Summary>
                    </About>

                    <ServerSection>
                    <ServerHandlers>
                        <FlatWorldGenerator generatorString="3;7,5*1,2*56,3*1,3,2;1;"/>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                    </ServerSection>
                    <AgentSection mode="Creative">
                    <Name>MalmoTutorialBot</Name>
                    <AgentStart>
                        <Inventory>
                            <InventoryItem slot="0" type="diamond_pickaxe"/>
                            <InventoryItem slot="1" type="stone" quantity="64"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <DiscreteMovementCommands/>
                        <ChatCommands />
                        <InventoryCommands />
                        <ObservationFromFullStats/>
                        <ObservationFromGrid>
                            <Grid name="state_space_box">
                                <min x="-6" y="-1" z="-6"/>
                                <max x="6" y="6" z="6"/>
                            </Grid>
                        </ObservationFromGrid>
                    </AgentHandlers>
                    </AgentSection>
                </Mission>'''

#                         <DiscreteMovementCommands />

def print_inventory(agent_host):
    agent_host.sendCommand("chat /clear @p[scores={inventory_min=1..}]")



def single_world():
    # Create the agent host
    agent_host = MalmoPython.AgentHost()
    # Parse the command-line arguments
    agent_host.parse(sys.argv)

    # Initialize GetPolicy class
    ddqn = DDQN(layers=(64,256,128,128,64))        # TODO: Load ddqn network  
    ddqn.load_model(filename="weights_save_superflat\super_flat NN at 14 trainings.h5")
    best_policy = BestPolicy(ddqn)

    # Get the mission XML and create a mission
    mission_xml = get_mission_xml()
    mission = MalmoPython.MissionSpec(mission_xml, True)
    mission.allowAllAbsoluteMovementCommands()
    mission_record = MalmoPython.MissionRecordSpec()

    # Start the mission
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission(mission, mission_record)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    # Wait for the mission to start
    print("Waiting for the mission to start")
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()

    #agent_host.sendCommand("chat /give @p minecraft:diamond_pickaxe{Enchantments:[{id:efficiency,lvl:5},{id:unbreaking,lvl:3},{id:mending,lvl:1},{id:fortune,lvl:3}]} 1")

    #move("N", agent_host)


    x = y = z = 0
    min_x = min_z = -6
    max_x = max_z = 6
    min_y = -1
    max_y = 6
    # S
    #W  E
    # N
    agent_host.sendCommand("jump 1")
    time.sleep(0.3)
    agent_host.sendCommand("jump 0")

    # agent_host.sendCommand("setPitch 30")

    # pitch(30, agent_host)
    # turn("West", agent_host)
    last_move = "N"

    max_step = 200
    while max_step >= 0:
        world_state = agent_host.getWorldState()
        
        if world_state.number_of_observations_since_last_state > 0:
            print(max_step)
            msg = world_state.observations[-1].text
            observations = json.loads(msg)

            # Access block information from the grid
            blocks_around_agent = observations.get('state_space_box', [])
            #tranfer to np array
            terrain_data = np.array(blocks_around_agent)
            terrain_data = terrain_data.reshape((max_y - min_y + 1, max_z - min_z + 1, max_x - min_x + 1))
            terrain_data[~np.isin(terrain_data, list(EXCLUSION))] = "stone"
            agent_y = observations.get("YPos", None)
            #print(terrain_data)
            # print(get_current_state(1,5,5,agent_y,terrain_data))

            state = get_current_state(y=1, x=6, z=6, last_move=last_move, height=agent_y, terrain_data=terrain_data)
            #update last move
            last_move = state[10]

            # Choose move
            chosen_move = best_policy.choose_move(state=state)
            agent_host.sendCommand(f'chat {max_step} {chosen_move}')
            # Move agent

            make_move(chosen_move, agent_host)

            max_step -= 1


    # Wait for the mission to end
    print("Waiting for the mission to end")
    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()

    print("Mission ended")

single_world()