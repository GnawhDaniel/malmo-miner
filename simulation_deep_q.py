from simulation_q import Simulation, Agent
import world_data_extractor
from helper import pickilizer, unpickle
import random, numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier


class Simulation_deep_q(Simulation):

    def choose_move(self, epsilon, state, q_function):
        possible_moves = self.agent.get_possible_moves(state)
        #EPSILON
        if (random.random() < epsilon or state not in q_table.keys()):
            return possible_moves[random.randint(0, len(possible_moves) - 1)]
        else:
            
            #implement this
            
            return #best move
        
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


class Classifier:
    def __init__(self, lr, gamma, layers=(30,30)):
        # Regression for critic: Q-values
        # Classifier for actor: Predict best move
        self.lr = lr
        self.layers = layers
        self.gamma = gamma

        self.actor = self.create_actor()
        self.critic = self.create_critic()
    
    def create_actor(self):
        mlp = MLPClassifier(hidden_layer_sizes=self.layers, 
                            alpha=self.lr,
                            )
        return mlp
    
    def create_critic(self):
        mlp = MLPRegressor(hidden_layer_sizes=self.layers, 
                            alpha=self.lr,)
        return mlp

    def _train_actor(self, samples):
        for sample in samples:
            s, a, r = sample
            input = (*s, a)
            a_pred = self.actor.predict(tuple(*s))
            r_pred = self.critic.predict(tuple(*s))
            

    def _train_critic(self, samples):
        pass

if __name__ == "__main__":

    episodes_per_training = 100
    episodes_per_world = 10000
    num_worlds = 10

    steps_per_simulation = 500
    states_sampled_per_episode = steps_per_simulation / 10

    num_trainings = 0
    trainings_per_pickle = 100

    lr = 0.1
    discount_rate = 0.95 
    epsilon = 0.2

    q_function = Classifier(...)

    for i in range(num_worlds):
        terrain_data, terrain_height = world_data_extractor.run()

        episodes_run_in_world = 0

        #PER WORLD
        while episodes_run_in_world < episodes_per_world:
            episodes_run_in_training = 0

            SAR_to_train_from = []
            
            #PER TRAINING SESSION
            while episodes_run_in_training < episodes_per_training:

                #reset simulation
                sim = Simulation(terrain_data, terrain_height)
                steps = 0
                episode = []

                prevState = None

                while steps < steps_per_simulation:

                    state = sim.get_current_state()

                    action = sim.choose_move(epsilon, state, q_function)

                    reward, dead = sim.get_reward(state, action)

                    steps += 1
                    episode.append((state, action, reward))

                    if dead:
                        break

                #CALCULATE AND PROPOGATE THE DISCOUNTED FUTURE REWARD
                future_reward = 0
                #go backwards through episode
                for i in range(len(episode) - 1, -1, -1):
                    #discount the future reward
                    future_reward *= discount_rate
                    #store the state's current reward
                    temp = episode[i, 2]
                    #add future reward to the state's reward
                    episode[i, 2] += future_reward
                    #add the current state's reward to the future reward for next states
                    future_reward += temp

                #MINI-BATCH and add to training data
                SAR_to_train_from += random.sample(episode, states_sampled_per_episode)

                #increment
                episodes_run_in_training += 1
                episodes_run_in_world += 1

            #TRAIN

            #IMPLEMENT THIS
            #q_learning_thing.partial_fit(SAR_to_train_from)

            num_trainings += 1

            #store result function every so often
            if num_trainings % trainings_per_pickle:
                #pickilizer(q_learning_thing,    "NN at " + str(num_trainings) + " trainings.pck")
                pass


    


    