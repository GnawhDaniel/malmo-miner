import MalmoPython
import os
import sys
import time
import random
import json
import numpy as np
#from api_helper import get_current_state

#change N/S, check wheter E/w is correct
def move(r_move):
    if r_move == "N":
        agent_host.sendCommand("move -1")
        time.sleep(1)
    if r_move == "S":
        agent_host.sendCommand("move 1")
        time.sleep(1)
    if r_move == "W":
        agent_host.sendCommand("strafe 1")
        time.sleep(1)
    if r_move == "E":
        agent_host.sendCommand("strafe -1")
        time.sleep(1)
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
        while new_height == agent_y:
            agent_host.sendCommand("jump")
            agent_host.sendCommand("jump")
            agent_host.sendCommand("use 1")
            time.sleep(1)
            world_state = agent_host.getWorldState()
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            new_height = observations.get("YPos", None)

        agent_host.sendCommand("look -1")
        agent_host.sendCommand("look -1")








def mine(r_move):
    agent_host.sendCommand("hotbar.1 0")
    #mine down
    if r_move == 'M_D':
        agent_host.sendCommand("look 1")
        time.sleep(0.5)
        agent_host.sendCommand("look 1")
        time.sleep(0.5)
        agent_host.sendCommand("attack 1")
        time.sleep(0.5)
        agent_host.sendCommand("look -1")
        time.sleep(0.5)
        agent_host.sendCommand("look -1")
        time.sleep(0.5)
    #mine south
    if r_move == "M_SU":
        agent_host.sendCommand("attack 1")
    if r_move == "M_SL":
        agent_host.sendCommand("look 1")
        time.sleep(0.5)
        agent_host.sendCommand("attack 1")
        time.sleep(0.5)
        agent_host.sendCommand("look -1")
    #mine north
    if r_move == "M_NU":
        agent_host.sendCommand("turn 1")
        time.sleep(1)
        agent_host.sendCommand("turn 1")
        time.sleep(1)
        agent_host.sendCommand("attack 1")
        time.sleep(1)
        agent_host.sendCommand("turn 1")
        time.sleep(1)
        agent_host.sendCommand("turn 1")
        time.sleep(1)
    if r_move == "M_NL":
        agent_host.sendCommand("turn 1")
        time.sleep(1)
        agent_host.sendCommand("turn 1")
        time.sleep(1)
        agent_host.sendCommand("look 1")
        time.sleep(0.5)
        agent_host.sendCommand("attack 1")
        time.sleep(0.5)
        agent_host.sendCommand("look -1")
        time.sleep(0.5)
        agent_host.sendCommand("turn 1")
        time.sleep(1)
        agent_host.sendCommand("turn 1")
        time.sleep(1)
    #mine west The x-axis indicates the player's distance east (positive)  or west (negative) of the origin point—i.e., the longitude,
    if r_move == "M_WU":
        agent_host.sendCommand("turn 1")
        time.sleep(1)
        agent_host.sendCommand("attack 1")
        time.sleep(1)
        agent_host.sendCommand("turn -1")
        time.sleep(1)
    if r_move == "M_WL":
        agent_host.sendCommand("turn 1")
        time.sleep(1)
        agent_host.sendCommand("look 1")
        time.sleep(0.5)
        agent_host.sendCommand("attack 1")
        time.sleep(0.5)
        agent_host.sendCommand("look -1")
        time.sleep(1)
        agent_host.sendCommand("turn -1")
        time.sleep(1)
    #mine east
    if r_move == "M_EU":
        agent_host.sendCommand("turn -1")
        time.sleep(1)
        agent_host.sendCommand("attack 1")
        time.sleep(1)
        agent_host.sendCommand("turn 1")
        time.sleep(1)
    if r_move == "M_EL":
        agent_host.sendCommand("turn -1")
        time.sleep(1)
        agent_host.sendCommand("look 1")
        time.sleep(0.5)
        agent_host.sendCommand("attack 1")
        time.sleep(0.5)
        agent_host.sendCommand("look -1")
        time.sleep(1)
        agent_host.sendCommand("turn 1")
        time.sleep(1)
    #mine up
    if r_move == "M_U":
        agent_host.sendCommand("look -1")
        time.sleep(0.5)
        agent_host.sendCommand("look -1")
        time.sleep(0.5)
        agent_host.sendCommand("attack 1")
        time.sleep(0.5)
        agent_host.sendCommand("look 1")
        time.sleep(0.5)
        agent_host.sendCommand("look 1")
        time.sleep(0.5)

def testing():
    while 1:
        c = input("command")
        agent_host.sendCommand(c)
        time.sleep(1)

# Helper function to get the XML string for the mission
def get_mission_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

                    <About>
                    <Summary>World data extractor</Summary>
                    </About>

                    <ServerSection>
                    <ServerHandlers>
                        <DefaultWorldGenerator/>
                        <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                    </ServerSection>
                    <AgentSection mode="Creative">
                    <Name>MalmoTutorialBot</Name>
                    <AgentStart>
                        <Inventory>
                            <InventoryItem slot="1" type="stone" quantity="64"/>
                            <InventoryItem slot="0" type="diamond_pickaxe"/>
                        </Inventory>
                    </AgentStart>
                    <AgentHandlers>
                        <InventoryCommands />
                        <ObservationFromFullStats/>
                        <DiscreteMovementCommands />
                        <ObservationFromGrid>
                            <Grid name="state_space_box">
                                <min x="-5" y="-1" z="-5"/>
                                <max x="5" y="6" z="5"/>
                            </Grid>
                        </ObservationFromGrid>
                    </AgentHandlers>
                    </AgentSection>
                </Mission>'''


# Create the agent host
agent_host = MalmoPython.AgentHost()

# Parse the command-line arguments
agent_host.parse(sys.argv)

# Get the mission XML and create a mission
mission_xml = get_mission_xml()
mission = MalmoPython.MissionSpec(mission_xml, True)
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


import json

x = y = z = 0
min_x = min_z = -5
max_x = max_z = 5
min_y = -1
max_y = 6
# S
#W  E
# N

max_step = 200
moving_cause = 2
mining_cause = 1
while max_step >= 0:
    max_step = 200
    moving_cause = 2
    mining_cause = 1
    while max_step >= 0:
        world_state = agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)

            # Access block information from the grid
            blocks_around_agent = observations.get('state_space_box', [])
            #tranfer to np array
            terrain_data = np.array(blocks_around_agent)
            terrain_data = terrain_data.reshape((max_y - min_y + 1, max_z - min_z + 1, max_x - min_x + 1))

            agent_y = observations.get("YPos", None)

            #print(get_current_state(1,5,5,agent_y,terrain_data))

        max_step -= 1
        time.sleep(10)

    # send_move to NN

# Wait for the mission to end
print("Waiting for the mission to end")
while world_state.is_mission_running:
    time.sleep(0.1)
    world_state = agent_host.getWorldState()

print("Mission ended")