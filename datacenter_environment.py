import numpy as np

class DataCenterEnvironment:
    def __init__(self, agents, token_manager, max_tokens):
        self.agents = agents
        self.token_manager = token_manager
        self.max_tokens = max_tokens
        self.states = np.array([max_tokens] * len(agents))  # Each agent starts with max 
        self.delta = 0.1 # Discount factor for future rewards

    def reset(self):
            """Reset the environment for a new episode."""
            self.states = np.array([self.max_tokens] * len(self.agents))
            for agent in self.agents:
                agent.low_power_mode_count = 0
            
    def step(self, actions):
        """
        Simulate the outcome of actions taken by agents.
        """
        rewards = []
        renewable_energy_availability = self.token_manager.get_renewable_energy_availability()
        # print(f'Total renewable energy = {renewable_energy_availability:.2f} megawatthours')
        self.token_manager.update_token_price(renewable_energy_availability)
        token_price = self.token_manager.get_token_price()
        high_power_mode_cost = token_price * 2
        low_power_mode_cost = token_price
        print(f'Token price: {token_price:.2f}')
        
        for i in range(len(self.agents)):
            action = actions[i]
            print("action", action)
                                    
            # Reward calculation - consists of immediate reward (10 or 20) due to performance and discounted reward due to conserving tokens
            if action == 0:
                self.states[i] -= low_power_mode_cost # Spend tokens
                reward = 10 + self.delta * self.states[i]
            else:
                self.states[i] -= high_power_mode_cost # Spend tokens
                reward = 20 + self.delta * self.states[i]
            rewards.append(reward)

            # if self.states[i] < token_cost:
            #     rewards.append(-1)  # Penalty for insufficient tokens, no state change
            #     continue
            
            # Replenish tokens based on low power mode usage
            # if renewable_energy_availability < 10000000:  # Trigger replenishment under low renewable energy
            #     # Determine the agent with the most low power mode usage
            #     most_deferring_agent = max(range(len(self.agents)), key=lambda i: self.agents[i].low_power_mode_count)
            #     self.states[most_deferring_agent] += 2  # Reward with extra tokens
            if self.agents[i].low_power_mode_count >= 1 and self.states[i] < 0:
            # if self.states[i] < 2:
                self.states[i] += 2

        return rewards