import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Add
from keras.optimizers import Adam
import numpy as np
import helper


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

    def batch(self, batch_size):
        max_size = min(self.counter, self.mem_size)
        batch = np.random.choice(max_size, batch_size) 
        return self.states[batch], self.actions[batch], self.rewards[batch], self.new_state[batch], self.terminals[batch] 

class DDQN:
    def __init__(self, lr, gamma, batch_size, layers=(256, 256), state_size=11, action_size=15, replace_target = 100):
        self.lr = lr
        self.gamma = gamma
        self.layers = layers
        self.action_size = action_size
        self.state_size = state_size
        self.batch_size=batch_size
        self.replace_target = replace_target

        self.action_space = [i for i in range(self.action_size)]
        # Model
        self.q_eval = self.create_NN()
        # Target
        self.q_target = self.create_NN()

        # Two nets
        # Only train target/online, then replace weights of the target network every N iterations
        

    def create_NN(self):
        state_input = Input(shape=(self.state_size,))  
        h = Dense(self.layers[0], activation='relu')(state_input) # TODO: Consider input shape size for batch training

        for layer in self.layers[1:]:
            h = Dense(layer, activation='relu')(h)
        
        output = Dense(self.action_size, activation="relu")(h)

        model = Model(inputs=state_input, outputs=output)
        adam = Adam(self.lr)
        model.compile(loss="mse", optimizer=adam)

        return model
    
    def learn(self, memory, ACTION_MAP, BLOCK_MAP):
        # self.states[batch], self.actions[batch], self.rewards[batch], self.new_state[batch], self.terminals[batch] 
        if memory.counter > self.batch_size:
            state, action, reward, new_state, done = \
                memory.batch(self.batch_size)
    

            # One Hot Encoded
            # myfunc_vec = np.vectorize(lambda x: ACTION_MAP[x])
            # action_ohe = myfunc_vec(action)
            # print(action_ohe)
    
            #brute force replace strings with hot codes
            def replace(array, map_object):
                for i in range(len(array)):
                    for j in range(len(array[i])):
                        array[i, j] = map_object[array[i, j]]

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

            max_actions = np.argmax(q_eval, axis=1) # TODO: convert this to choose playable moves, instead max of all possible moves
                
            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma * q_next[batch_index, max_actions.astype(int)] * done

            self.q_eval.fit(state, q_target, verbose=0)

            if memory.counter % self.replace_target == 0:
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
