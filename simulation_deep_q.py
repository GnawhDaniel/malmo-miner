from simulation_q import Simulation, Agent
#import world_data_extractor
from helper import pickilizer, unpickle
import helper
import random, numpy as np
from model import DDQN, Memory
import tensorflow as tf

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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

HEIGHT_MOD = 3

#ONE HOT ENCODING
BLOCK_MAP, ACTION_MAP = helper.enumerate_one_hot()
ALL_MOVES = ["N","S","W","E","U","M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]


class Simulation_deep_q(Simulation):
    def choose_move(self, epsilon, state, dqn):
        possible_moves = self.agent.get_possible_moves(state)

        #EPSILON
        if (random.random() < epsilon):
            return possible_moves[random.randint(0, len(possible_moves)-1)]
        #GREEDY
        else:
            #convert to onehot
            height = state[-1] % HEIGHT_MOD
            state = [BLOCK_MAP[s] for s in state[:-1]]
            # state = helper.convert_bits(state)
            state.append(height)

            state = np.array(state, dtype=np.float32)
            state = state[np.newaxis, :]


            #state = tf.convert_to_tensor(state)
            
            actions = dqn.q_eval.predict(state, verbose=0)[0]

            # --> [q_score1, q_score2, q_score3, ...]
            
            #SORTED INDECES
            sorted_actions = np.flip(np.argsort(actions))

            # best_sorted = np.argmax(actions)
            # print(best_sorted)

            #choose best move only if move is possible in current state
            for action in sorted_actions:
                if ALL_MOVES[action] in possible_moves:
                    return ALL_MOVES[action]
            
            return ALL_MOVES[0] #should never happen
        
    def calculate_diamond_locations(self): 
        diamonds = []

        #find diamond coordinates in the terrain data
        ys, xs, zs = np.where(self.terrain_data == "diamond_ore")

        #add to diamonds list
        for i in range(len(ys)):
            diamonds.append(xs[i], ys[i], zs[i] - terrain_height)

        self.diamond_locations = np.array(diamonds)
    
    def diamond_heuristic(self, x, y, z):
        try:
            loc = np.array([x,y,z])
            best = self.diamond_locations[0]
            best_dist = np.Inf
            
            for i in self.diamond_locations:
                #compare distances
                dist = np.linalg.norm(loc - i)
                if (dist) < best_dist:
                    best = i
                    best_dist = dist

            #CALCULATE HEURISTIC BASED ON THE BEST DIAMOND LOCATION

            return #reward
        
        except:
            return 0


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    lr = 0.05
    gamma = 0.5
    epsilon = 0.9
    epsilon_min=0.1
    epsilon_dec=0.99

    ddqn = DDQN(lr, gamma, batch_size=512, layers=(256, 256), state_size=11, action_size=15)
    memory = Memory(250_000, state_size=11)

    episodes_per_training = 100
    trainings_per_world = 100
    num_worlds = 10

    steps_per_simulation = 200

    training_counter = 0
    trainings_per_save = 10

    episode_counter = 0


    for _ in range(num_worlds):
        #terrain_data, terrain_height = world_data_extractor.run()
        terrain_data, terrain_height = helper.create_custom_world(50, 50, [(3, "air"), (5, "stone") ,(2,"diamond_ore")])
        terrain_height -= 1

        #PER WORLD
        for _ in range(trainings_per_world): # Simulating same world episodes_per_training amount of times
            
            diamonds_sum = 0
            best_diamonds = 0
            
            #PER TRAINING SESSION
            for episode in range(episodes_per_training:=20): 
                

                #reset simulation
                sim = Simulation_deep_q(terrain_data, terrain_height)

                prevState = None

                stored_epsilon = epsilon

                print_statistics = episode == (episodes_per_training - 1)

                if print_statistics:
                    epsilon = 0

                for step in range(steps_per_simulation): # Simulating a single world

                    state = sim.get_current_state()

                    # Change state to mapped version
                    action = sim.choose_move(epsilon, state, ddqn)
                    
                    reward, dead = sim.get_reward(state, action)
                    new_state = sim.get_current_state()

                    #TODO: ADD HEURISTICs TO REWARD HERE
                    #reward += heuristic(state, actio)

                    #adjust height
                    state[-1] = state[-1] % HEIGHT_MOD
                    new_state[-1] = new_state[-1] % HEIGHT_MOD
                    
                    # print(new_state, state)

                    done = dead or step == steps_per_simulation - 1
                    memory.store((state, action, reward, new_state, done))
                    #break early
                    if dead:
                        break
            
                epsilon = stored_epsilon

                diamonds_found = sim.agent.inventory["diamond_ore"]
                diamonds_sum += diamonds_found

                if diamonds_found > best_diamonds:
                    best_diamonds = diamonds_found

                episode_counter += 1

                if print_statistics:
                    print("Episode:", episode_counter)
                    print("\tAverage diamonds mined:", diamonds_sum / episodes_per_training)
                    print("\tMost diamonds found:", best_diamonds)
                    print("\tBest policy: ", diamonds_found) 
                        
                
            #TRAIN
            ddqn.learn(memory, ACTION_MAP, BLOCK_MAP)

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