from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
import helper, copy


class Memory:
    def __init__(self, mem_size, state_size) -> None:
        self.counter = 0
        self.mem_size = mem_size

        #S A R S' done
        
        self.states = np.zeros((mem_size, state_size), dtype=object)
        self.rewards = np.zeros((mem_size,), dtype=object)
        self.new_state = np.zeros((mem_size, state_size), dtype=object)
        self.actions = np.zeros((mem_size,), dtype=object)
        self.terminals = np.zeros((mem_size,), dtype=object)

    def store(self, tup):

        index = self.counter % self.mem_size

        state, action, reward, new_state, done = tup

        # for i in range(len(action)):
        self.states[index] = state
        self.new_state[index] = new_state

        self.actions[index] = action
        self.rewards[index] = reward
        self.terminals[index] = done

        self.counter += 1
        if self.counter % self.mem_size == 0:
            print("!!!Restarted Memory!!!")

    def batch(self, batch_size):
        max_size = min(self.counter, self.mem_size)
        batch = np.random.choice(max_size, batch_size) 
        return self.states[batch], self.actions[batch], self.rewards[batch], self.new_state[batch], self.terminals[batch] 

class DDQN:
    def __init__(self, lr, gamma, batch_size, layers=(256, 256), state_size=11, action_size=15, replace_target=5, regularization_strength=0.05):
        self.lr = lr
        self.gamma = gamma
        self.layers = layers
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size=batch_size
        self.replace_target = replace_target
        self.counter = 0 # Counter for replace_target
        self.regularization_strength = regularization_strength

        self.action_space = [i for i in range(self.action_size)]
        # Model
        self.q_eval = self.create_NN()
        # Target
        self.q_target = self.create_NN()

        # Two nets
        # Only train target/online, then replace weights of the target network every N iterations
        

    def create_NN(self):
        state_input = Input(shape=(self.state_size,))  
        h = Dense(self.layers[0], activation='tanh')(state_input) # TODO: Consider input shape size for batch training

        for layer in self.layers[1:]:
            h = Dense(layer, activation='tanh')(h)
        
        output = Dense(self.action_size, activation="linear", \
                       kernel_regularizer=regularizers.l1(self.regularization_strength))(h)

        model = Model(inputs=state_input, outputs=output)
        adam = Adam(self.lr)
        model.compile(loss="mse", optimizer=adam)

        return model
    
    def train(self, memory, ACTION_MAP, BLOCK_MAP, sim, single = False, item=None):
        # self.states[batch], self.actions[batch], self.rewards[batch], self.new_state[batch], self.terminals[batch] 
        if single or memory.counter > self.batch_size:
            state, action, reward, new_state, done = None, None, None, None, None,
            if not single:
                state, action, reward, new_state, done = \
                    memory.batch(self.batch_size)
            else:
                state, action, reward, new_state, done = item
                state = np.array(state)[np.newaxis, :] 
                action = np.array([action])[np.newaxis, :] 
                reward = np.array([reward])[np.newaxis, :] 
                new_state = np.array(new_state)[np.newaxis, :] 
                done = np.array([done])[np.newaxis, :] 
            

            new_state_copy = copy.deepcopy(new_state)

            # Brute force replace strings with enumerated values
            def replace(array, map_object):
                for i in range(len(array)):
                    for j in range(len(array[i])-1):
                        array[i, j] = map_object[array[i, j]]
                    array[i, len(array[0])-1] = ACTION_MAP[array[i, len(array[0])-1]] # Change previous move to int

            replace(state[:, :-1], BLOCK_MAP)
            replace(new_state[:, :-1], BLOCK_MAP)
            new_state = new_state.astype(np.float32)
            state = state.astype(np.float32)
            

            # action_values = np.array(self.action_space) # , dtype=np.int8
            # action_indices = np.dot(action, action_values) # TODO: I have a feeling that this won't work
            index_func = np.vectorize(lambda x: helper.ALL_MOVES.index(x))
            action_indices = index_func(action)


            q_next = self.q_target.predict(new_state, verbose=0) # Batch predict
            q_eval = self.q_eval.predict(new_state, verbose=0)

            q_pred = self.q_eval.predict(state, verbose=0)

            
            # Choose playable moves, instead max of all possible moves
            # max_actions = np.argmax(q_eval, axis=1) 
            max_actions = []
            for ind, actions in enumerate(q_eval):
                # actions = [a1, a2, ..., a_n]
                actions = np.flip(np.argsort(actions)) # Sort action weights
                counter = 0
                while counter < len(actions):
                    possible_moves = sim.agent.get_possible_moves(new_state_copy[ind])
                    possible_indices = [helper.ALL_MOVES.index(move) for move in possible_moves]
                    if actions[counter] in possible_indices:
                        max_actions.append(actions[counter])
                        break # Break in while loop does not break outer for loop
                    counter += 1
            max_actions = np.array(max_actions)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32) if not single else np.array([0])

            q_target[batch_index, action_indices] = reward + (self.gamma * q_next[batch_index, max_actions.astype(int)] * np.invert(done))
            
            self.q_eval.fit(state, q_target, verbose=0)
            # print(self.q_eval.layers[1].weights)
            
            if not single:
                self.counter += 1

                if self.counter % self.replace_target == 0:
                    # print(memory.counter, self.replace_target)
                    print("Replaced Q Target Network w/ Q Eval")
                    self.update_parameters()
        else:
            print("Skipped Training")

    
    
    def update_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self, filename):
        self.q_eval.save(filename)

    def load_model(self, filename):
        self.q_eval = load_model(filename)
        self.q_target = self.q_eval
