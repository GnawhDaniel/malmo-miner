from simulation_q import Simulation, Agent
import world_data_extractor
from helper import pickilizer, unpickle
import helper
import random, numpy as np
from model import DDQN, Memory
import tensorflow as tf
import copy
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
        # print(possible_moves)
        # print(state)
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
                        print(ALL_MOVES[action], save_state, self.closest_diamond, end=" ")

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

        best_dist = np.linalg.norm(self.closest_diamond - loc)
            
        #CALCULATE HEURISTIC BASED ON THE BEST DIAMOND LOCATION
        relative_coords = {"N": (0, 0, -1),"S":(0, 0, 1),"W":(-1, 0, 0),"E":(1, 0, 0),"U":(0, 1, 0), "D": (0, -1, 0)}
        
        updated_move = loc + relative_coords[move[0]] if move[0] != "M" else loc + relative_coords[move[2]]

        new_distance = np.linalg.norm(updated_move - self.closest_diamond)

        value = (best_dist - new_distance) * multiplier

        if value < 0:
            value *= 3

        return value


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_ylim(0, 20)
    line, = ax.plot([], [], lw=2)

    lr = 0.001
    gamma = 0.95
    epsilon = 0.99
    epsilon_min=0.2
    epsilon_dec=0.992


    heuristic_factor = 5
    straight_line_reward = 10

    ddqn = DDQN(lr, gamma, batch_size=200, layers=(64,128,128,64), state_size=12, action_size=15, replace_target=7, regularization_strength=0.0612)
    memory = Memory(250_000, state_size=12)

    episodes_per_training = 100
    trainings_per_world = 100
    num_worlds = 10

    steps_per_simulation = 500

    training_counter = 0
    trainings_per_save = 10

    episode_counter = 0


    for _ in range(num_worlds):
        # terrain_data, terrain_height = world_data_extractor.run()
        dic = helper.unpickle("terrain_data.pck")
        terrain_data = dic["terrain_data"]
        terrain_height = dic["starting_height"]

        # terrain_data, terrain_height = helper.create_custom_world(3, 3, [(3, "air"), (5, "stone") ,(2,"diamond_ore")])
        terrain_height -= 2

        sim = Simulation_deep_q(terrain_data, terrain_height)
        # sim.fall()

        sim.calculate_diamond_locations()
        diamond_locations = sim.diamond_locations

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

                if print_statistics:
                    epsilon = 0

                for step in range(steps_per_simulation): # Simulating a single world

                    state = sim.get_current_state()

                    # Change state to mapped version
                    action = sim.choose_move(epsilon, state, ddqn)

                    heuristic_diamond = sim.diamond_heuristic(action, heuristic_factor)
                    
                    

                    reward, dead = sim.get_reward(state, action)
                    new_state = sim.get_current_state()


                    #diamond heuristic

                    d_h = reward == 0
                    if d_h: # So other ores aren't weighted negatively
                        reward += heuristic_diamond

                    # Straight line heuristic
                    s_l_r = action[0] != 'M' and state[10] == action and action != "U"
                    if s_l_r:
                        reward += straight_line_reward

                    if epsilon == 0:
                        print(reward, heuristic_diamond * d_h, straight_line_reward * s_l_r)
     
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

                # print(episode)
                epsilon = stored_epsilon


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
                    print("\tBest Reward Policy (ep=0):", reward)
                    print("\tEpsilon:", epsilon)
                    print("\tInventory: ", sim.agent.inventory)

                    line.set_data(np.append(line.get_xdata(), episode_counter), np.append(line.get_ydata(), diamonds_found))

                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()
                    # plt.pause(0.001)
                    # if len(line.get_xdata()) > 50:
                    #     line.set_data(line.get_xdata()[1:], line.get_ydata()[1:])
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                        
            # #TRAIN
            ddqn.train(memory, ACTION_MAP, BLOCK_MAP, sim)

            #decrease epsilon every training
            epsilon *= epsilon_dec
            if epsilon < epsilon_min:
                epsilon = epsilon_min

            #store network every n trainings
            if training_counter % trainings_per_save:
                filename = "NN at " + str(training_counter) + " trainings.h5"
                #ddqn.save(filename)
            
            



# #CALCULATE AND PROPOGATE THE DISCOUNTED FUTURE REWARD
# future_reward = 0
# #go backwards through episode
# for i in range(len(episode) - 1, -1, -1):
#     #discount the future reward
#     future_reward *= discount_rate
#     #store the state's current reward
#     temp = episode[i, 2]
#     #add future reward to the state's reward
#     episode[i, 2] += future_reward
#     #add the current state's reward to the future reward for next states
#     future_reward += temp

# #MINI-BATCH and add to training data
# SAR_to_train_from += random.sample(episode, states_sampled_per_episode)

