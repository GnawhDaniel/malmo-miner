from __future__ import print_function
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

from builtins import range
import sys

sys.path.append('../') # Add this to import MalmoPython
import MalmoPython

import time
import helper
import numpy as np
import pickle as pck
from helper import pickilizer


def run():
    # if sys.version_info[0] == 2:
    #     sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
    # else:
    #     import functools
    #     print = functools.partial(print, flush=True)

    # More interesting generator string: "3;7,44*49,73,35:1,159:4,95:13,35:13,159:11,95:10,159:14,159:6,35:6,95:6;12;"

    missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
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
                    </AgentStart>
                    <AgentHandlers>
                        <ObservationFromFullStats/>
                        <ContinuousMovementCommands turnSpeedDegs="180"/>
                        <ObservationFromGrid>
                        <Grid name="state_space_box">
                            <min x="-150" y="-1" z="-150"/>
                            <max x="150" y="-1" z="150"/>
                        </Grid>
                        </ObservationFromGrid>
                    </AgentHandlers>
                    </AgentSection>
                </Mission>'''
            
    #<ServerQuitFromTimeUp timeLimitMs="30000"/>

    #<Placement x="0.5" y="120.0" z="0.5"/>

    # Create default Malmo objects:

    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    my_mission = MalmoPython.MissionSpec(missionXML, True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    # Attempt to start a mission:
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')

    world_state = agent_host.getWorldState()

    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission running ", end=' ')

    agent_host.sendCommand("pitch 1")
    time.sleep(3)
    agent_host.sendCommand("pitch 0")

    agent_host.sendCommand("attack 1")

    height_seen = set()
    terrain_data = []
    
    AIR_BUFFER = 3 
    for _i in range(AIR_BUFFER):
        terrain_data.append(np.full((301,301), "air"))

    starting_height = None

    # Loop until mission ends:
    while world_state.is_mission_running:
        #print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()

        for error in world_state.errors:
            print("Error:",error.text)

        world_data, height = helper.get_grid_observation(world_state, "state_space_box")

        if height != None:

            if (starting_height == None):
                starting_height = height

            if height not in height_seen:
                world_data = np.array(world_data)
                world_data = world_data.reshape((301,301))
                terrain_data.append(world_data)

                height_seen.add(height)
    
            #print(terrain_data, height)

            #end the mission when world extracted
            if (height <= 5):
                # world_state.is_mission_running = False
                break


    BEDROCK_BUFFER = 5
    for _i in range(BEDROCK_BUFFER):
        terrain_data.append(np.full((301,301), "bedrock"))

    terrain_data = np.array(terrain_data)
    print()
    print("Mission ended")
    # Mission has ended.

    print("Starting height: ", starting_height)
    print("Layers: ", terrain_data.shape)
    # print("Layer size: ", terrain_data[0].shape)

    #PREPROCESS BLOCKS
    all_blocks = set(np.unique(terrain_data))
    for i in all_blocks - helper.EXCLUSION:
        terrain_data[terrain_data==i] = "stone"
        
    terrain_data = np.flipud(terrain_data)
    save_dict = {"terrain_data": terrain_data, "starting_height": starting_height+AIR_BUFFER}
    pickilizer(save_dict, "terrain_data.pck")

    return terrain_data, starting_height + AIR_BUFFER