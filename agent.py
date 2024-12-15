import numpy as np
import random

class Agent:
    def __init__(self, id, max_tokens, num_actions=2):
        self.id = id
        self.max_tokens = max_tokens
        self.actions = ["normal power mode", "high power mode"] # Numeric action codes: 0 = normal, 1 = high
        self.num_actions = num_actions 
        self.q_table = np.zeros((max_tokens + 1, num_actions))  # State: no. of tokens = token value, Actions: power modes
        self.alpha = 0.1 # Agent-specific Learning rate
        self.gamma = 0.9 # Agent-specific Discount factor
        self.low_power_mode_count = 0  # Track how many times the agent chose low power mode

    def select_action(self, state, epsilon):
        """Select an action using epsilon-greedy policy."""
        if random.uniform(0, 1) < epsilon:
            action = random.choice(range(self.num_actions))  # Explore actions within valid action range, irrespective of q-value
        else:
            action = np.argmax(self.q_table[state])  # Exploit - choose action based on highest q-value
        return action

    def update_q_table(self, state, action, reward, next_state):
        """Update the Q-table based on observed reward and next state."""
        self.q_table[state, action] += self.alpha * (
            reward + self.gamma * np.max(self.q_table[next_state, :]) - self.q_table[state, action]
        )
    
    def record_low_power_mode(self, action):
        """Record instances of choosing low power mode."""
        if action == 0:  # Normal power mode
            self.low_power_mode_count += 1