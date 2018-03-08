import numpy as np
import _pickle as pickle

from utils import sigmoid

INPUT_SIZE = 80 * 80
HIDDEN_SIZE = 200
DECAY_RATE = 0.99 # RMSProp
LEARNING_RATE = 1e-4

class PolicyNetwork:
    def __init__(self, is_resume):
        if is_resume:
            model = pickle.load(open("model.p", "rb"))
        else:
            # Xavier initialization
            model = {}
            model["w1"] = np.random.randn(HIDDEN_SIZE, INPUT_SIZE) / np.sqrt(INPUT_SIZE)
            model["w2"] = np.random.randn(HIDDEN_SIZE) / np.sqrt(HIDDEN_SIZE)

        self.model = model
        self.grads_buffer = { k: np.zeros_like(v) for k, v in model.items() } # grads sum over a batch
        self.rmsprop_memory = { k: np.zeros_like(v) for k, v in model.items() }

    def forward(self, x):
        self.h = np.dot(self.model["w1"], x)
        self.h[self.h <= 0] = 0 # ReLU nonlinearity

        logp = np.dot(self.model["w2"], self.h)
        p = sigmoid(logp)
        return p

    def backward(self, episode_x, episode_h, drewards_sum):
        dw2 = np.dot(episode_h.T, drewards_sum).ravel()
        dh = np.outer(drewards_sum, self.model["w2"])
        dh[dh <= 0] = 0 # ReLU backprop
        dw1 = np.dot(dh.T, episode_x)
        return { "w1": dw1, "w2": dw2 }

    def update(self):
        for k, v in self.model.items():
            grad = self.grads_buffer[k]

            self.rmsprop_memory[k] = DECAY_RATE * self.rmsprop_memory[k] + (1 - DECAY_RATE) * grad**2
            self.model[k] += LEARNING_RATE * grad / (np.sqrt(self.rmsprop_memory[k]) + 1e-5)

            self.grads_buffer[k] = np.zeros_like(v)

