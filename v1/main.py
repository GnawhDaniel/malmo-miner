from simulation import Simulation,REWARD_TABLE
import helper
import random
from sklearn.neural_network import MLPClassifier
import numpy as np




if __name__ == "__main__":
    # terrain_data = unpickle("terrain_data.pck")
    # starting_height = 70 # TODO: Change later to proper starting height using terrain_data.pck

    # # Pre-process: convert unimportant blocks to stone
    # all_blocks = set(np.unique(terrain_data))
    # for i in all_blocks - EXCLUSION:
    #     terrain_data[terrain_data==i] = "stone"


    num_epi = 100000
    num_epi_permap = num_epi    #fix this later


    max_step = 200
    moving_penatly = 1
    mining_penatly = 2
    MINING_REWARD = 0
    MOVING_REWARD = -10

    reward_track = []  # store reward for each epsiode

    critic = MLPClassifier()

    terrain_data, terrain_height = helper.create_custom_world(50, 50, [(3, "air"), (5, "stone"), (2, "diamond_ore")])

    starting_height = terrain_height - 1


    file = open("results.txt", 'w')

    move = ["N", "S", "W", "E", "U", "M_NL", "M_NU", "M_EL", "M_EU", "M_SL", "M_SU", "M_WL", "M_WU", "M_U", "M_D"]

    mapping = []

    lr = 0.1  # learning rate
    discount_rate = 0.99


    diamond_sum = 0
    for i in range(int(num_epi / num_epi_permap)):

        terrain_data, terrain_height = helper.create_custom_world(50, 50, [(3, "air"), (5, "stone"), (2, "diamond_ore")])
        starting_height = terrain_height - 1

        epsilon = 1   #see if this will be problem

        for i2 in range(num_epi_permap):
            #print(i2)
            #print("testing")
            s = Simulation(terrain_data, starting_height)

            steps = max_step

            state = s.get_current_state()

            # Every 100th, show best policy
            ep = epsilon #if i2 % 100 else 0

            d = 0  # track if death in epsidoe

            SAR = []
            while steps > 0:

                action = s.choose_move(ep, state, critic,mapping)

                reward, d = s.get_reward(state, action)


                if not action.startswith('M_'):
                    reward += MOVING_REWARD



                steps = steps - mining_penatly if action.startswith('M_') else steps - moving_penatly


                new_state = s.get_current_state()
                if i2>1000:
                    pass
                    #p_m = s.agent.get_possible_moves(new_state)
                    #temp_new_SA = [list(new_state)+[i] for i in p_m]
                    #temp_new_SA = s.convert(temp_new_SA,mapping)
                    #critic.predict(temp_new_SA)
                    #reward  = (1-lr)*reward + lr*(reward+discount_rate*max(critic.predict(temp_new_SA)))





                SAR.append([list(state), action, float(reward)])
                state = new_state
                # break if meet lava
                if d:
                    break

            epsilon *= 0.999
            #update the SAR table with future vale train the critcic model in future

            #take random batch
            batch = random.sample(SAR, len(SAR)//2)
            X = [S+[A] for S,A,R in batch]
            #convert to int
            X = s.convert(X,mapping)

            Y = np.asarray([R for S, A, R in batch])
            critic.fit(X,Y)

            diamonds_mined = s.agent.inventory["diamond_ore"]

            diamond_sum += diamonds_mined

            total_reward = 0
            while s.agent.inventory:
                k, v = s.agent.inventory.popitem()
                if k in REWARD_TABLE.keys():
                    total_reward += REWARD_TABLE[k] * v

            reward_track.append(total_reward)

            if i2 % 100 == 0 :
                print("Episode:", i2)
                print("\tAverage diamonds mined:", diamond_sum / 100)
                print("\tBest policy: ", diamonds_mined)

                # episode, q_table, avg_diamonds
                #file.write(f"{i}, {diamond_sum / 100}, {diamonds_mined}\n")

                # print("\tQ_TABLE:", len(q_table))
                diamond_sum = 0

        file.close()