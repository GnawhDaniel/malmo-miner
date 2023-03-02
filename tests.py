from simulation import Simulation
import pickle as pck
import numpy as np
from helper import pickilizer, unpickle, REWARD_TABLE, EXCLUSION

"""
!!!
Use "pytest ./tests.py" in command line.
!!!
"""

terrain_data = unpickle("terrain_data.pck")
starting_height = 70 # TODO: Change later to proper starting height using terrain_data.pck

# Pre-process: convert unimportant blocks to stone
all_blocks = set(np.unique(terrain_data))
for i in all_blocks - EXCLUSION:
    terrain_data[terrain_data==i] = "stone"

class TestStateSpace:
    def test_case_state_space1(self):
        # Testing Simulation.get_current_state
        test = Simulation(terrain_data, starting_height=50) # Returned ('stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 50)
    
        # Replace two blocks north of agent w/ Air:
        # Expect: ('air', 'air', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 50)
        test.terrain_data[50, 150, 150-1] = "diamond_ore"
        test.terrain_data[51, 150, 150-1] = "diamond_ore"
        assert (test.get_current_state()==("diamond_ore", "diamond_ore", 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 50)), test.get_current_state()

    def test_case_state_space2(self):
        # Replace two blocks East of agent w/ Air:
        test = Simulation(terrain_data, starting_height=50)

        test.terrain_data[50, 150+1, 150] = "diamond_ore"
        test.terrain_data[51, 150+1, 150] = "diamond_ore"
        assert (test.get_current_state()==('stone', 'stone', "diamond_ore", "diamond_ore", 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 50))
    
    def test_case_state_space3(self):
        # Replace two blocks South of agent w/ Air:
        test = Simulation(terrain_data, starting_height=50)

        test.terrain_data[50, 150, 150+1] = "diamond_ore"
        test.terrain_data[51, 150, 150+1] = "diamond_ore"
        assert (test.get_current_state()==('stone', 'stone', 'stone', 'stone', 'diamond_ore', 'diamond_ore', 'stone', 'stone', 'stone', 'stone', 50))

    def test_case_state_space4(self):
        # Replace two blocks West of agent w/ Air:
        test = Simulation(terrain_data, starting_height=50)

        test.terrain_data[50, 150-1, 150] = "diamond_ore"
        test.terrain_data[51, 150-1, 150] = "diamond_ore"
        assert (test.get_current_state()==('stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'diamond_ore', 'diamond_ore', 'stone', 'stone', 50))


class TestAirBlock:
    def test_air_block1(self):
        # Search Length = 1
        test = Simulation(terrain_data, starting_height=50) 
        test.terrain_data[50, 150, 150-1] = 'air'
        test.terrain_data[50, 150, 150-2] = 'diamond_ore'
        assert (test.get_current_state()==('air+diamond_ore', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 50)), test.get_current_state()

    def test_air_block2(self):
        # Search Length = 2
        test = Simulation(terrain_data, starting_height=50) 
        test.terrain_data[50, 150, 150-1] = 'air'
        test.terrain_data[50, 150, 150-2] = 'air'
        test.terrain_data[50, 150, 150-3] = 'diamond_ore' # Diamond at the end (lower level)
        assert (test.get_current_state()==('air+diamond_ore', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 50)), test.get_current_state()
    
    def test_air_block3(self):
        """
        The x-axis indicates the player's distance east (positive)  or west (negative) of the origin point—i.e., the longitude,
        The z-axis indicates the player's distance south (positive) or north (negative) of the origin point—i.e., the latitude 
        """
        # Search Length = 5
        test = Simulation(terrain_data, starting_height=50) 
        # y, x , z
        test.terrain_data[51, 150+1, 150] = 'air'
        test.terrain_data[51, 150+2, 150] = 'air'
        test.terrain_data[51, 150+3, 150] = 'air'
        test.terrain_data[51, 150+4, 150] = 'air'
        test.terrain_data[51, 150+5, 150] = 'diamond_ore' # Diamond at the end of search length (eye level)
        print(test.get_current_state())
        assert (test.get_current_state()==('stone', 'stone', 'stone', 'air+diamond_ore', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 50)), test.get_current_state()

    def test_air_block4(self):
        test = Simulation(terrain_data, starting_height=50)
        test.terrain_data[51, 150+1, 150] = 'air'
        test.terrain_data[52, 150+2, 150] = "diamond_ore" # Diamond ore on ceiling
        test.terrain_data[51, 150+2, 150] = 'air'
        test.terrain_data[51, 150+3, 150] = 'air'
        test.terrain_data[51, 150+4, 150] = 'air'
        test.terrain_data[51, 150+5, 150] = 'stone'
        assert (test.get_current_state()==('stone', 'stone', 'stone', 'air+diamond_ore', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 50)), test.get_current_state()


    def test_air_block5(self):
        test = Simulation(terrain_data, starting_height=50)
        test.terrain_data[51, 150+1, 150] = 'air'
        test.terrain_data[52, 150+2, 150] = "diamond_ore" # Diamond ore on ceiling
        test.terrain_data[50, 150+2, 150] = "coal_ore"
        test.terrain_data[51, 150+2, 150] = 'air'
        test.terrain_data[51, 150+3, 150] = 'air'
        test.terrain_data[51, 150+4, 150] = 'air'
        test.terrain_data[51, 150+5, 150] = 'stone'
        assert (test.get_current_state()==('stone', 'stone', 'stone', 'air+diamond_ore', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 50)), test.get_current_state()

        
    def test_air_block6(self):
        test = Simulation(terrain_data, starting_height=50)
        test.terrain_data[51, 150+1, 150] = 'air'
        test.terrain_data[52, 150+2, 150] = "emerald_ore"
        test.terrain_data[50, 150+2, 150] = "coal_ore"
        test.terrain_data[50, 150+4, 150] = "diamond_ore"
        test.terrain_data[51, 150+2, 150] = 'air'
        test.terrain_data[51, 150+3, 150] = 'air'
        test.terrain_data[51, 150+4, 150] = 'air'
        test.terrain_data[51, 150+5, 150] = 'stone'
        assert (test.get_current_state()==('stone', 'stone', 'stone', 'air+diamond_ore', 'stone', 'stone', 'stone', 'stone', 'stone', 'stone', 50)), test.get_current_state()




    
