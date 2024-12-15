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
        print(f'Total renewable energy = {renewable_energy_availability:.2f} megawatthours')
        self.token_manager.update_token_price(renewable_energy_availability)
        token_price = self.token_manager.get_token_price()
        high_power_mode_cost = token_price
        low_power_mode_cost = token_price / 10
        print(f'Token price: {token_price:.2f}')
        
        for i in range(len(self.agents)):
            action = actions[i]
            if (action == 1):
                token_cost = high_power_mode_cost
            else:
                token_cost = low_power_mode_cost
            performance_gains = [10, 15]  # Corresponding performance gains 

            if self.states[i] < token_cost:
                rewards.append(-1)  # Penalty for insufficient tokens, no state change
                continue

            print("action", action)
            self.states[i] -= token_cost # Spend tokens
            print(f"performance gains: {performance_gains[action]:.2f}")
            print(f"token cost: {token_cost:.2f}")
            immediate_reward = performance_gains[action]
            print(f"immediate reward: {immediate_reward:.2f}")
            future_value = self.delta * self.states[i]  # Reward conserving tokens
            print(f"future value: {future_value:.2f}")
            rewards.append(immediate_reward + future_value)
            
            # Replenish tokens based on low power mode usage
            if renewable_energy_availability < 10000000:  # Trigger replenishment under low renewable energy
                # Determine the agent with the most low power mode usage
                most_deferring_agent = max(range(len(self.agents)), key=lambda i: self.agents[i].low_power_mode_count)
                self.states[most_deferring_agent] += 20  # Reward with extra tokens
            
            # Replenish tokens based on low power mode usage
            if self.agents[i].low_power_mode_count > 1 and self.states[i] < 2:
            # if self.states[i] < 2:
                self.states[i] += 20

        return rewards