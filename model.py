### imports ###
from pathlib import Path

import numpy as np

### loading files ###
BASE_DIR = Path(__file__).resolve().parent

path = BASE_DIR / "models" / "predictor_weights.npz"
path_o = BASE_DIR / "models" / "offensive_weights.npz"

path_data = BASE_DIR / "models" / "pong_dataset_predictor.npy"
path_data_o = BASE_DIR / "models" / "pong_dataset_offensive.npy"

path.parent.mkdir(exist_ok=True)


class Model:
    def __init__(self):

        ### predictor model supervised learning ###
        self.learning_rate = 0.01
        self.epochs = 3000
        self.batch_size = 128

        ### offensive model policy gradient ###
        self.learning_rate_o = 0.001
        self.sigma = 0.3

        ### loading data ###
        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.W3 = data["W3"]
        self.b3 = data["b3"]

        data2 = np.load(path_o)
        self.W1O = data2["W1O"]
        self.b1O = data2["b1O"]
        self.W2O = data2["W2O"]
        self.b2O = data2["b2O"]
        self.W3O = data2["W3O"]
        self.b3O = data2["b3O"]

        ### local data objects ###
        self.dataset = np.load(path_data)
        self.dataset_list = self.dataset.tolist()

        self.dataset_o = np.load(path_data_o)
        self.dataset_list_o = self.dataset_o.tolist()

    ### activation functions ###
    def sigmoid(self, v):
        return 1 / (1 + np.exp(-v))

    def leaky_relu(self, x):
        return np.where(x > 0, x, 0.01 * x)

    def tanh(self, x):
        return np.tanh(x)

    ### ### ### ### ### ### ###

    # =====# TRAJECTORY PREDICTOR MODEL #=====#

    ### training predictor model ###
    def train(self):
        self.dataset = np.load(path_data)
        for epoch in range(self.epochs):
            np.random.shuffle(self.dataset)

            for batch in range(0, self.dataset.shape[0], self.batch_size):
                batch_states = self.dataset[batch : batch + self.batch_size, :3]
                batch_targets = self.dataset[batch : batch + self.batch_size, 3:4]

                self.learn(batch_states, batch_targets)

    ### forward pass predictor model ###
    def predict(self, state):
        state = np.array(state, dtype=float)

        layer_1 = state @ self.W1 + self.b1
        layer_1_activated = self.leaky_relu(layer_1)

        layer_2 = layer_1_activated @ self.W2 + self.b2
        layer_2_activated = self.leaky_relu(layer_2)

        output = layer_2_activated @ self.W3 + self.b3
        output_activated = self.sigmoid(output)

        return output_activated.item()

    def forward_pass(self, batch):

        layer_1 = batch @ self.W1 + self.b1
        layer_1_activated = self.leaky_relu(layer_1)

        layer_2 = layer_1_activated @ self.W2 + self.b2
        layer_2_activated = self.leaky_relu(layer_2)

        output = layer_2_activated @ self.W3 + self.b3
        output_activated = self.sigmoid(output)

        return (
            batch,
            layer_1,
            layer_1_activated,
            layer_2,
            layer_2_activated,
            output,
            output_activated,
        )

    ### backward pass predictor model ###
    def learn(self, batch_states, batch_targets):
        (
            input,
            layer_1,
            layer_1_activated,
            layer_2,
            layer_2_activated,
            _,
            output_activated,
        ) = self.forward_pass(batch_states)

        dL_doutput_activated = (
            2 * (output_activated - batch_targets) / batch_states.shape[0]
        )

        doutput_activated_doutput = output_activated * (1 - output_activated)

        dL_doutput = dL_doutput_activated * doutput_activated_doutput

        dL_dW3 = layer_2_activated.T @ dL_doutput
        dL_db3 = np.sum(dL_doutput, axis=0)

        dL_dlayer_2_activated = dL_doutput @ self.W3.T
        dL_dlayer_2 = dL_dlayer_2_activated * np.where(layer_2 > 0, 1.0, 0.01)

        dL_dW2 = layer_1_activated.T @ dL_dlayer_2
        dL_db2 = np.sum(dL_dlayer_2, axis=0)

        dL_dlayer_1_activated = dL_dlayer_2 @ self.W2.T
        dL_dlayer_1 = dL_dlayer_1_activated * np.where(layer_1 > 0, 1.0, 0.01)

        dL_dW1 = input.T @ dL_dlayer_1
        dL_db1 = np.sum(dL_dlayer_1, axis=0)

        self.W3 -= dL_dW3 * self.learning_rate
        self.b3 -= dL_db3 * self.learning_rate
        self.W2 -= dL_dW2 * self.learning_rate
        self.b2 -= dL_db2 * self.learning_rate
        self.W1 -= dL_dW1 * self.learning_rate
        self.b1 -= dL_db1 * self.learning_rate

    # =====#  #=====#  #=====#  #=====#  #=====#

    # =====#   OFFENSIVE POLICY  MODEL   #=====#

    ### training offensive model ###
    def train_offensive(self, state, action, reward):
        self.save_offensive_data(state, action, reward)

        batch_states = np.array(state, dtype=float).reshape(1, -1)
        batch_actions = np.array([[action]], dtype=float)
        batch_rewards = np.array([[reward]], dtype=float)

        self.learn_offensive(batch_states, batch_actions, batch_rewards)
        self.save()

    ### forward pass offensive model ###
    def predict_offensive(self, state):
        state = np.array(state, dtype=float)

        layer_1 = state @ self.W1O + self.b1O
        layer_1_activated = self.leaky_relu(layer_1)

        layer_2 = layer_1_activated @ self.W2O + self.b2O
        layer_2_activated = self.leaky_relu(layer_2)

        output = layer_2_activated @ self.W3O + self.b3O
        mu = self.tanh(output) / 2

        a = np.random.normal(mu, self.sigma)
        action = np.clip(a, -0.5, 0.5)

        return mu.item(), action.item()

    def forward_pass_offensive(self, batch):
        layer_1 = batch @ self.W1O + self.b1O
        layer_1_activated = self.leaky_relu(layer_1)

        layer_2 = layer_1_activated @ self.W2O + self.b2O
        layer_2_activated = self.leaky_relu(layer_2)

        output = layer_2_activated @ self.W3O + self.b3O
        output_activated = self.tanh(output) / 2

        return (
            batch,
            layer_1,
            layer_1_activated,
            layer_2,
            layer_2_activated,
            output,
            output_activated,
        )

    ### backward pass offensive model ###
    def learn_offensive(self, batch_states, batch_actions, batch_rewards):
        (
            input,
            layer_1,
            layer_1_activated,
            layer_2,
            layer_2_activated,
            output,
            output_activated,
        ) = self.forward_pass_offensive(batch_states)

        dL_doutput_activated = (
            -batch_rewards
            * (batch_actions - output_activated)
            / (self.sigma**2 * batch_states.shape[0])
        )

        doutput_activated_doutput = 0.5 * (1 - np.tanh(output) ** 2)

        dL_doutput = dL_doutput_activated * doutput_activated_doutput

        dL_dW3 = layer_2_activated.T @ dL_doutput
        dL_db3 = np.sum(dL_doutput, axis=0)

        dL_dlayer_2_activated = dL_doutput @ self.W3O.T
        dL_dlayer_2 = dL_dlayer_2_activated * np.where(layer_2 > 0, 1.0, 0.01)

        dL_dW2 = layer_1_activated.T @ dL_dlayer_2
        dL_db2 = np.sum(dL_dlayer_2, axis=0)

        dL_dlayer_1_activated = dL_dlayer_2 @ self.W2O.T
        dL_dlayer_1 = dL_dlayer_1_activated * np.where(layer_1 > 0, 1.0, 0.01)

        dL_dW1 = input.T @ dL_dlayer_1
        dL_db1 = np.sum(dL_dlayer_1, axis=0)

        self.W3O -= dL_dW3 * self.learning_rate_o
        self.b3O -= dL_db3 * self.learning_rate_o
        self.W2O -= dL_dW2 * self.learning_rate_o
        self.b2O -= dL_db2 * self.learning_rate_o
        self.W1O -= dL_dW1 * self.learning_rate_o
        self.b1O -= dL_db1 * self.learning_rate_o

    # =====#  #=====#  #=====#  #=====#  #=====#

    ### helpers ###
    def save(self):
        np.savez(
            path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3
        )
        np.savez(
            path_o,
            W1O=self.W1O,
            b1O=self.b1O,
            W2O=self.W2O,
            b2O=self.b2O,
            W3O=self.W3O,
            b3O=self.b3O,
        )

    def save_train_data(self, state, target):
        self.dataset_list.append([*state, target])
        np.save(path_data, np.array(self.dataset_list))

    def save_offensive_data(self, state, action, reward):
        self.dataset_list_o.append([*state, action, reward])
        np.save(path_data_o, np.array(self.dataset_list_o))

    ### ### ### ###
