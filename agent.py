import numpy as np
import random

class Agent:
    def __init__(self, id, num_states=3, num_actions=2):
        self.id = id
        self.q_table = np.zeros((num_states, num_actions))
        self.epsilon = 0.1 # Exploration rate
        self.alpha = 0.1 # Learning rate
        self.gamma = 0.9 # Discount facor

    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1]) # Explore
        return np.argmax(self.q_table[state]) # Exploit
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table based on observed reward and next state."""
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (reward + self.gamma * self.q_table[next_state, best_next_action] - self.q_table[state, action])
