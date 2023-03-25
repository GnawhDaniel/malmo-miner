from simulation_q import Simulation, Agent
# import world_data_extractor
from helper import pickilizer, unpickle
import helper
import random, numpy as np
from model import DDQN, Memory
import copy, os, glob, signal, time


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


class Simulation_deep_q(Simulation):
    def choose_move(self, epsilon, state, dqn):
        possible_moves = self.agent.get_possible_moves(state)
        save_state = copy.deepcopy(state) # Debug Purposes
        
        #EPSILON
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

            #SORTED INDICES
            sorted_actions = np.flip(np.argsort(actions))

            #choose best move only if move is possible in current state
            for action in sorted_actions:
                if helper.ALL_MOVES[action] in possible_moves:
                    if epsilon == 0:
                        """Debug"""
                        print(f"{helper.ALL_MOVES[action]} {save_state[-1]}", end= ", ")
                    self.last_move = helper.ALL_MOVES[action]
                    return self.last_move
        print(state)
        raise 
    
    def calculate_diamond_locations(self): 
        diamonds = []

        # Find diamond coordinates in the terrain data
        ys, xs, zs = np.where(self.terrain_data == "diamond_ore")

        # Add to diamonds list
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
                    # Compare distances
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
       
        # CALCULATE HEURISTIC BASED ON THE BEST DIAMOND LOCATION
        new_distance = np.linalg.norm(updated_move - self.closest_diamond)
        value = (best_dist - new_distance) * multiplier
        # print(best_dist - new_distance, best_dist, new_distance)

        return value


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Graphing Lines

    # plt.ion()
    # fig, ax = plt.subplots(3, figsize=(13, 13), gridspec_kw={'height_ratios': [1, 1, 1]})
    # plt.subplots_adjust(top=1.25)
    # # fig.tight_layout(pad=5.0)
    # line1, = ax[0].plot([], [], lw=2)
    # line2, = ax[1].plot([], [], lw=2)
    # line3, = ax[2].plot([], [], lw=2)

    # ax[2].set_xlabel("Episodes")
    # ax[0].set_ylabel("Best Policy")
    # ax[1].set_ylabel("Diamonds Found")
    # ax[2].set_ylabel("Ores Found")


    # Hyperparameters 
    lr = 0.001
    gamma = 0.10 # 0.95
    layers = (64,256,256,64)
    batch_size = 3000
    replace_target = 10
    regularization_strength = 0.001
    memory_size = 1_000_000
    tau = 0.001
    
    epsilon = 0.25
    epsilon_min=0.1
    epsilon_dec=0.99

    target_reward = 500 # To change epsilon (rolling window)


    # Construct NN and Experience Replay Buffer
    ddqn = DDQN(lr, gamma, batch_size=batch_size, layers=layers, state_size=12, action_size=15, replace_target=replace_target, regularization_strength=regularization_strength, tau=tau)
    memory = Memory(memory_size, state_size=12)
    # ddqn.load_model("C:\\Users\\danie\\Desktop\\DIAMOND_NN_280.h5")

    # Heurtistic Parameters
    heuristic_factor = 5
    straight_line_reward = 10
    walk_back = -5
    opposite_direction = {"N": "S", "S": "N", "E":"W", "W":"E", "M_D": "U", "U": "M_D"}

    # Training Duration
    episodes_per_training = 100
    trainings_per_world = 50
    num_worlds = 10
    steps_per_simulation = 500
    training_counter = 0
    trainings_per_save = 10
    episode_counter = 0

    # Save Stats
    stats = {"ep_step": episodes_per_training,"best_policies": [], "diamonds": [], "ores": []}
    def handler(signum, frame):
        global stats
        inp = input("Save training? Y/N: ").strip().lower()
        if inp == "y":
            helper.pickilizer(stats, "save_stats.json")
            print("Saved stats.")
        inp = input("Kill training? Y/N: ").strip().lower()
        if inp == "y":
            print("Exited.")
            exit(1)

    signal.signal(signal.SIGINT, handler)

    # Save hyperparameters & reward table
    file_directory = "weights_save_random\\" 
    files = glob.glob(f'{file_directory}/*')
    for f in files:
        os.remove(f)
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

    best_policy = float('-inf')
    for world_num in range(num_worlds):
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
        # PER WORLD
        for train_num in range(trainings_per_world): # Simulating same world episodes_per_training amount of times
            
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

                    # diamond heuristic
                    d_h = reward == 0
                    if d_h: # So other ores aren't weighted negatively
                        reward += heuristic_diamond

                    # # Straight line heuristic
                    # s_l_r = action[0] != 'M' and state[10] == action and action != "U"
                    # if s_l_r:
                    #    reward += straight_line_reward
                    
                    # Walk back and forth heuristic
                    bool_ = (action in opposite_direction) and (state[10] in opposite_direction) \
                             and (action == opposite_direction[state[10]])
                    if bool_:
                        reward += walk_back

                    # For calculating average reward policy
                    reward_sum += reward
                    if reward > best_reward:
                        best_reward = reward

                    # Adjust height
                    state[-1] = state[-1] // HEIGHT_MOD
                    new_state[-1] = new_state[-1] // HEIGHT_MOD

                    done = dead or step == steps_per_simulation - 1
                    memory.store((state, action, reward, new_state, done))

                    reward_cumul += reward
                    
                    if dead is True:
                        print("Died")
                        break
                
                diamonds_found = sim.agent.inventory["diamond_ore"]
                diamonds_sum += diamonds_found

                if diamonds_found > best_diamonds:
                    best_diamonds = diamonds_found

                episode_counter += 1
                if print_statistics:
                    print()
                    print(f"Episode {episode_counter} @ World {world_num}/{train_num}:", episode_counter)
                    print(f"\tMost diamonds found (per {episodes_per_training} episode):", best_diamonds)
                    print("\tAverage diamonds mined:", diamonds_sum / episodes_per_training)
                    print("\tBest diamond policy (ep=0): ", diamonds_found)
                    print("\tAverage Reward:", reward_sum/episodes_per_training)
                    print(f"\tBest Reward (per {episodes_per_training} episodes):", best_reward)
                    print("\tCumulative Reward Policy (ep=0):", reward_cumul)
                    print("\tInventory: ", sim.agent.inventory)
                    epsilon = stored_epsilon
                    training_counter += 1

                    # Save stats
                    stats["best_policies"].append(reward_cumul)
                    stats["diamonds"].append(diamonds_found)
                    stats["ores"].append(sum(sim.agent.inventory.values())-sim.agent.inventory["stone"])

                    # Save weights if notable rewards occur
                    # ores = sum(sim.agent.inventory.values())-sim.agent.inventory["stone"]
                    # if sim.agent.inventory["diamond_ore"] > 0:
                    #     filename = f"{file_directory}/DIAMOND_NN_{episode_counter}.h5"
                    #     ddqn.save_model(filename)
                    # elif ores > 5:
                    #     filename = f"{file_directory}/ores{ores}_NN_{episode_counter}.h5"
                    #     ddqn.save_model(filename)
                    #     best_policy = reward_cumul
                    # if reward_cumul > best_policy:
                    #     filename = f"{file_directory}/NN_{episode_counter}.h5"
                    #     ddqn.save_model(filename)
                    #     best_policy = reward_cumul

                    # Draw Graph
                    # line1.set_data(np.append(line1.get_xdata(), episode_counter), np.append(line1.get_ydata(), reward_cumul))
                    # line2.set_data(np.append(line2.get_xdata(), episode_counter), np.append(line2.get_ydata(), diamonds_found))
                    # line3.set_data(np.append(line3.get_xdata(), episode_counter), np.append(line3.get_ydata(), sum(sim.agent.inventory.values())))
                    # ax[0].relim()
                    # ax[0].autoscale_view()
                    # ax[1].relim()
                    # ax[1].autoscale_view()
                    # ax[2].relim()
                    # ax[2].autoscale_view()
                    # plt.draw()
                    # fig.canvas.draw()
                    # fig.canvas.flush_events()

                # TRAIN
                ddqn.train(memory, ACTION_MAP, BLOCK_MAP, sim)


                policy_history.append(reward_cumul)
                # Adjusting Epsilon based on Rewards
                if len(policy_history) >= 20: #Update every 5 x 100 episodes

                    # Rolling average
                    avg = sum(policy_history) / len(policy_history) 
                    print("Rolling Avg:", avg)
                
                    # If rolling average is greater than some eps threshold, decrease eps
                    if avg > target_reward:
                        print("Decrease")
                        epsilon *= 0.8
                    else:
                        # Else increase epsilon
                        print("Increase")
                        epsilon *= 1.1
                    policy_history.clear()

       
                if epsilon > .99:
                    epsilon = .99
                elif epsilon < epsilon_min:
                    epsilon = epsilon_min
                
                print(f"Current epsilon @ ep {episode_counter}:", epsilon)
        

            