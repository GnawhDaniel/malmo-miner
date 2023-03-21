import tensorflow as tf
import world_data_extractor as extract
import numpy as np
import helper
from simulation_deep_q import Simulation_deep_q

# terrain_data, starting_height = extract.run()

obj = helper.unpickle("terrain_data.pck")
terrain_data = obj["terrain_data"]
starting_height = obj["starting_height"]
print(starting_height)

sim = Simulation_deep_q(terrain_data, starting_height-2)
print(sim.get_current_state())




