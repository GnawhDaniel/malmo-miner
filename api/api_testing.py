import MalmoPython
import os
import sys
import time
import random
import json
import numpy as np


def move(r_move):
    if r_move == "N":
        agent_host.sendCommand("move 1")
        time.sleep(1)
    if r_move == "S":
        agent_host.sendCommand("move -1")
        time.sleep(1)
    if r_move == "W":
        agent_host.sendCommand("strafe -1")
        time.sleep(1)
    if r_move == "E":
        agent_host.sendCommand("strafe 1")
        time.sleep(1)
    if r_move == "U":
        # this is not working
        agent_host.sendCommand("jump 1")
        time.sleep(0.25)
        agent_host.sendCommand("hotbar.0 1")
        agent_host.sendCommand("jump 0")
        time.sleep(0.15)
        agent_host.sendCommand("place down")
        time.sleep(0.25)


def mine(r_move):
    agent_host.sendCommand("hotbar.1 0")
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
    if r_move == "M_NU":
        agent_host.sendCommand("attack 1")
    if r_move == "M_NL":
        agent_host.sendCommand("look 1")
        time.sleep(0.5)
        agent_host.sendCommand("attack 1")
        time.sleep(0.5)
        agent_host.sendCommand("look -1")
    if r_move == "M_EU":
        agent_host.sendCommand("turn 1")
        time.sleep(1)
        agent_host.sendCommand("attack 1")
        time.sleep(1)
        agent_host.sendCommand("turn -1")
        time.sleep(1)
    if r_move == "M_EL":
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
    if r_move == "M_SU":
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
    if r_move == "M_SL":
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
    if r_move == "M_WU":
        agent_host.sendCommand("turn -1")
        time.sleep(1)
        agent_host.sendCommand("attack 1")
        time.sleep(1)
        agent_host.sendCommand("turn 1")
        time.sleep(1)
    if r_move == "M_WL":
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
    if r_move == "M_WL":
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
                                <min x="-2" y="-1" z="-2"/>
                                <max x="2" y="2" z="2"/>
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
min_x = min_z = -2
max_x = max_z = 2
min_y = -1
max_y = 2


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
            print(agent_y)
            for i in terrain_data:
                print(i)
        max_step -= 1
        time.sleep(10)

    # send_move to NN

# Wait for the mission to end
print("Waiting for the mission to end")
while world_state.is_mission_running:
    time.sleep(0.1)
    world_state = agent_host.getWorldState()

print("Mission ended")