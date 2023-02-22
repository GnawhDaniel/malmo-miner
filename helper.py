"""
Helper functions:
http://microsoft.github.io/malmo/0.30.0/Documentation/annotated.html
"""
import MalmoPython as Malmo
import os
import sys
import time
import json
import math

def get_grid_observation(world_state, name):

    if (world_state.number_of_observations_since_last_state):
        state = world_state.observations[-1].text
        state = json.loads(state)

        a_dict = {
            "north": [],
            "east": [],
            "south": [],
            "west": [],
            "top": [],
            "bottom": []
        }

        return state[name], math.floor(state["YPos"])
        
    return None, None


def get_state_space(mission_spec, agent_host):
    # Find the doc for getting blocks around player
    
    return

def rotate_agent(degrees, agent_host):
    return

def pitch_agent(degrees, agent_host):
    return

def teleport(direction, agent_host):    
    return

def mine_forward(direction, agent_host, up = False):
    return

