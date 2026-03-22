import numpy as np
import random
from collections import deque

class DQNAgent:
    def __init__(self, model, target_model):
        self.model = model
        self.target_model = target_model

        self.replay_buffer = deque(maxlen=500_000)

        self.gamma = 0.98
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_decay_steps = 800_000

        self.total_steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(3)
        q = self.model(state, training=False)[0]
        return np.argmax(q)

    def store(self, s, a, r, ns, d):
        self.replay_buffer.append((s, a, r, ns, d))

    def train(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)

        s, a, r, ns, d = zip(*batch)

        s = np.vstack(s)
        ns = np.vstack(ns)
        a = np.array(a)
        r = np.array(r)
        d = np.array(d, dtype=int)

        q = self.model(s, training=False).numpy()
        q_next = self.target_model(ns, training=False).numpy()

        target = q.copy()
        max_q_next = np.max(q_next, axis=1)

        target[np.arange(len(batch)), a] = r + (1 - d) * self.gamma * max_q_next

        self.model.fit(s, target, epochs=1, verbose=0)

    def update_epsilon(self):
        self.epsilon = max(self.eps_min, 1.0 - self.total_steps / self.eps_decay_steps)