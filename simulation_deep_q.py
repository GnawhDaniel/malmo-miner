from simulation_q import Simulation, Agent
import world_data_extractor
from helper import pickilizer, unpickle
import helper
import random, numpy as np
from model import DDQN, Memory
import tensorflow as tf
import copy, math, os, glob
from collections import deque
'''
░░░░▄▄▄▄▀▀▀▀▀▀▀▀▄▄▄▄▄▄
░░░░█░░░░▒▒▒▒▒▒▒▒▒▒▒▒░░▀▀▄
░░░█░░░▒▒▒▒▒▒░░░░░░░░▒▒▒░░█
░░█░░░░░░▄██▀▄▄░░░░░▄▄▄░░░█
░▀▒▄▄▄▒░█▀▀▀▀▄▄█░░░██▄▄█░░░█
█▒█▒▄░▀▄▄▄▀░░░░░░░░█░░░▒▒▒▒▒█
█▒█░█▀▄▄░░░░░█▀░░░░▀▄░░▄▀▀▀▄▒█
░█▀▄░█▄░█▀▄▄░▀░▀▀░▄▄▀░░░░█░░█
░░█░░▀▄▀█▄▄░█▀▀▀▄▄▄▄▀▀█▀██░█
░░░█░░██░░▀█▄▄▄█▄▄█▄████░█
░░░░█░░░▀▀▄░█░░░█░███████░█
░░░░░▀▄░░░▀▀▄▄▄█▄█▄█▄█▄▀░░█
░░░░░░░▀▄▄░▒▒▒▒░░░░░░░░░░█
░░░░░░░░░░▀▀▄▄░▒▒▒▒▒▒▒▒▒▒░█
░░░░░░░░░░░░░░▀▄▄▄▄▄░░░░░█
Problem?
'''

HEIGHT_MOD = 1

#ONE HOT ENCODING
BLOCK_MAP, ACTION_MAP = helper.enumerate_one_hot()
ALL_MOVES = ["N","S","W","E","U","M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]


class Simulation_deep_q(Simulation):
    def choose_move(self, epsilon, state, dqn):
        possible_moves = self.agent.get_possible_moves(state)
        
        #EPSILON
        save_state = copy.deepcopy(state) # TODO: uncomment
        if (random.random() < epsilon):
            self.last_move = possible_moves[random.randint(0, len(possible_moves)-1)]
            return self.last_move
        #GREEDY
        else:
            height = state[-1] // HEIGHT_MOD
            prev_move = ACTION_MAP[state[-2]]
            state = [BLOCK_MAP[s] for s in state[:-2]]
            state.append(prev_move)
            state.append(height)
            state = np.array(state, dtype=np.float32)
            state = state[np.newaxis, :]

            actions = dqn.q_eval.predict(state, verbose=0)[0]

            #SORTED INDECES
            sorted_actions = np.flip(np.argsort(actions))

            #choose best move only if move is possible in current state
            for action in sorted_actions:
                if ALL_MOVES[action] in possible_moves:
                    if epsilon == 0:
                        """Debug"""
                        # print(dqn.q_eval.layers[1].weights)
                        print(ALL_MOVES[action], save_state[-1], end= ", ")
                        #  save_state, self.closest_diamond

                        # myfunc_vec = np.vectorize(lambda x: ACTION_MAP[x].astype(int))
                        # possible_moves_ind = myfunc_vec(possible_moves)
                        # print(possible_moves_ind)

                        # print(possible_moves)
                        # print(np.array(actions))
                        # print(np.array(actions)[possible_moves_ind])
                        # print(action)
                        pass

                    self.last_move = ALL_MOVES[action]
                    return self.last_move
            
            raise 
    
    def calculate_diamond_locations(self): 
        diamonds = []

        #find diamond coordinates in the terrain data
        ys, xs, zs = np.where(self.terrain_data == "diamond_ore")

        #add to diamonds list
        for i in range(len(ys)):
            ore = self.at(xs[i], ys[i], zs[i])
            assert ore == "diamond_ore", f"Not Diamond ore got, {ore}"
            diamonds.append((xs[i], ys[i], zs[i]))

        self.diamond_locations = np.array(diamonds)

    def diamond_heuristic(self, move, multiplier=1.5):
        loc = np.array([*self.agent_xyz()])
        loc_eyes = loc
        loc_eyes[1] += 1
        
        if type(self.closest_diamond) == type(None):
            best = self.diamond_locations[0]
            best_dist = np.Inf
        
            self.diamond_locations -= loc
            distances = np.linalg.norm(self.diamond_locations, axis=1)
            self.diamond_locations += loc

            for i in range(len(self.diamond_locations)):
                if (self.at(*self.diamond_locations[i]) == "diamond_ore"):
                    #compare distances
                    
                    if (distances[i]) < best_dist:
                        best = self.diamond_locations[i]
                        best_dist = distances[i]
            
            self.closest_diamond = best

        dist_eyes = np.linalg.norm(self.closest_diamond - loc_eyes)
        dist_feet = np.linalg.norm(self.closest_diamond - loc)

        relative_coords = {"N": (0, 0, -1),"S":(0, 0, 1),"W":(-1, 0, 0),"E":(1, 0, 0),"U":(0, 1, 0), "D": (0, -1, 0)}
        
        best_dist = min(dist_feet, dist_eyes)

        updated_move = loc + relative_coords[move[0]] if move[0] != "M" else loc + relative_coords[move[2]]
        updated_move = None
        if best_dist == dist_feet:
            updated_move = loc + relative_coords[move[0]] if move[0] != "M" else loc + relative_coords[move[2]]
        else:
            updated_move = loc_eyes + relative_coords[move[0]] if move[0] != "M" else loc_eyes + relative_coords[move[2]]
        #CALCULATE HEURISTIC BASED ON THE BEST DIAMOND LOCATION


        new_distance = np.linalg.norm(updated_move - self.closest_diamond)

        value = (best_dist - new_distance) * multiplier

        return value


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    plt.ion()
    fig, ax = plt.subplots()
    # ax.set_ylim(0, 20)
    line, = ax.plot([], [], lw=2)

    lr = 0.007
    gamma = 0.95 # 0.95
    layers = (64, 256, 128,64)
    batch_size = 50_000
    replace_target = 4
    regularization_strength = 0.001
    memory_size = 1_000_000
    
    
    epsilon = 0.99
    epsilon_min=0.1
    epsilon_dec=0.976


    target_reward = 400
    heuristic_factor = 1.5
    straight_line_reward = 3


    ddqn = DDQN(lr, gamma, batch_size=batch_size, layers=layers, state_size=12, action_size=15, replace_target=replace_target, regularization_strength=regularization_strength)
    memory = Memory(memory_size, state_size=12)

    episodes_per_training = 100
    trainings_per_world = 100
    num_worlds = 10

    steps_per_simulation = 500

    training_counter = 0
    trainings_per_save = 10

    episode_counter = 0

    file_directory = "weights_save_random\\" 
    files = glob.glob(f'{file_directory}/*')
    for f in files:
        os.remove(f)
    # Save parameters
    with open(f"{file_directory}/hyperparameters.txt", 'w') as f:
        f.write(f"lr = {lr}\n")
        f.write(f"gamma =  {gamma}\n")
        f.write(f"epsilon = {epsilon}\n")
        f.write(f"epsilon dec = {epsilon_dec}\n")
        f.write(f"layers = {layers}\n")
        f.write(f"heuristic_factor = {heuristic_factor}\n")
        f.write(f"straight_line_reward = {straight_line_reward}\n")
        f.write(f"regularization: {regularization_strength}\n")
        f.write(f"batch_size {batch_size}\n")
        f.write(f"mem_size: {memory_size}\n")
        f.write(f"target_reward: {target_reward}\n")
        f.write(f"reward_table: {helper.REWARD_TABLE}\n")
        f.write(f"move penalty: {Agent.MOVE_PENALTY}\n")



    max_diamond = float('-inf')
    low_diamond = float('inf')
    best_poicy = float('-inf')

    for _ in range(num_worlds):
        # world_data_extractor.run() # To get different "worlds", just teleport agent to another chunk on the same world really high up, then call fall
        dic = helper.unpickle("terrain_data.pck")
        terrain_data = dic["terrain_data"]
        terrain_height = dic["starting_height"]

        # terrain_data, terrain_height = helper.create_custom_world(400, 400, [(3, "air"), (5, "stone") ,(2,"diamond_ore"), (5, "stone")])
        terrain_height -= 2

        sim = Simulation_deep_q(terrain_data, terrain_height)
        # sim.fall()

        sim.calculate_diamond_locations()
        diamond_locations = sim.diamond_locations

        policy_history = []
        #PER WORLD
        for _ in range(trainings_per_world): # Simulating same world episodes_per_training amount of times
            
            diamonds_sum = 0
            best_diamonds = 0 
            best_reward = 0
            reward_sum = 0
            #PER TRAINING SESSION


            for episode in range(episodes_per_training): 
                
                #reset simulation
                sim = Simulation_deep_q(terrain_data, terrain_height)
                sim.fall()

                sim.diamond_locations = diamond_locations
                stored_epsilon = epsilon

                print_statistics = episode == (episodes_per_training - 1)
                reward_cumul = 0
                if print_statistics:
                    epsilon = 0
                    

                for step in range(steps_per_simulation): # Simulating a single world

                    state = sim.get_current_state()

                    # Change state to mapped version
                    action = sim.choose_move(epsilon, state, ddqn)

                    heuristic_diamond = sim.diamond_heuristic(action, heuristic_factor)
                    
                    reward, dead = sim.get_reward(state, action)
                    new_state = sim.get_current_state()

                    
                    # if heuristic_diamond > max_diamond:
                    #     print("New Max:", heuristic_diamond)
                    #     max_diamond = heuristic_diamond
                    # if heuristic_diamond < low_diamond:
                    #     print("New Min:", heuristic_diamond)
                    #     low_diamond = heuristic_diamond

                    #diamond heuristic
                    d_h = reward == 0
                    if d_h: # So other ores aren't weighted negatively
                        reward += heuristic_diamond

                    # Straight line heuristic
                    s_l_r = action[0] != 'M' and state[10] == action and action != "U"
                    if s_l_r:
                       reward += straight_line_reward

                    # if epsilon == 0:
                        # print(reward, heuristic_diamond * d_h, straight_line_reward * s_l_r)
                        # print(sim.agent_xyz())
                    reward_cumul += reward
     
                    reward_sum += reward
                    if reward > best_reward:
                        best_reward = reward

                    #adjust height
                    state[-1] = state[-1] // HEIGHT_MOD
                    new_state[-1] = new_state[-1] // HEIGHT_MOD
                    
                    done = dead or step == steps_per_simulation - 1
                    memory.store((state, action, reward, new_state, done))

                    if dead:
                        break

                policy_history.append(reward_cumul)

                diamonds_found = sim.agent.inventory["diamond_ore"]
                diamonds_sum += diamonds_found

                if diamonds_found > best_diamonds:
                    best_diamonds = diamonds_found

                episode_counter += 1
                if print_statistics:
                    print()
                    print("Episode:", episode_counter)
                    print("\tMost diamonds found (per 100 episode):", best_diamonds)
                    print("\tAverage diamonds mined:", diamonds_sum / episodes_per_training)
                    print("\tBest diamond policy (ep=0): ", diamonds_found)
                    print("\tAverage Reward:", reward_sum/episodes_per_training)
                    print("\tBest Reward (per 100 episodes):", best_reward)
                    print("\tCumulative Reward Policy (ep=0):", reward_cumul)
                    print("\tInventory: ", sim.agent.inventory)
                    epsilon = stored_epsilon
                    line.set_data(np.append(line.get_xdata(), episode_counter), np.append(line.get_ydata(), reward_cumul))

                    training_counter += 1
   
                    if reward_cumul > best_poicy:
                        filename = f"{file_directory}/gen NN at " + str(training_counter) + " trainings.h5"
                        ddqn.save_model(filename)
                        best_poicy = reward_cumul
                        print("Saved")
            

                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()
                    # plt.pause(0.001)
                    # if len(line.get_xdata()) > 50:
                    #     line.set_data(line.get_xdata()[1:], line.get_ydata()[1:])
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                        
            # #TRAIN
            print("Episode:", episode_counter)
            print("Training...")
            ddqn.train(memory, ACTION_MAP, BLOCK_MAP, sim)

            # Decrease epsilon every training
            epsilon *= epsilon_dec
            if epsilon < epsilon_min:
                epsilon = epsilon_min

      
            # # Adjusting Epsilon based on Rewards
            # if len(policy_history) >= 500: #Update every 5 x 100 episodes

            #     # Rolling average
            #     avg = sum(policy_history) / len(policy_history) 
            #     print("Rolling Avg:", avg)
            
            #     # If rolling average is greater than some eps threshold, decrease eps
            #     if avg > target_reward:
            #         print("Decrease")
            #         epsilon *= 0.4
            #     else:
            #         # Else increase epsilon
            #         print("Increase")
            #         epsilon *= 1.2
            #     policy_history.clear()
            # else:
            #     # Decrease epsilon 
            #     epsilon *= epsilon_dec
            #     if epsilon < epsilon_min:
            #         epsilon = epsilon_min
    
            if epsilon > .99:
                epsilon = .99
            elif epsilon < epsilon_min:
                epsilon = epsilon_min

            print(epsilon)

            '''
            #store network every n trainings
            if training_counter % trainings_per_save:
                filename = "NN at " + str(training_counter) + " trainings.h5"
                #ddqn.save(filename)
                
            '''
            
            


"""
Ideas/TODO:
Increase tau as rewards increase
Scale reward weights (including heuristic)
"""