from datacenter_environment import DataCenterEnvironment
from rl_agent import RLAgent
from token_manager import TokenManager
import requests
import numpy as np

class Simulation:
    def __init__(self, num_gpus, num_episodes, total_power_tokens=1000, scarcity_factor=1.0):
        self.num_gpus = num_gpus
        self.num_episodes = num_episodes
        self.env = DataCenterEnvironment(num_gpus)
        self.state_size = self.env._get_observation().shape[0]
        self.action_size = 2
        self.agents = [RLAgent(self.state_size, self.action_size, agent_id=i) for i in range(num_gpus)]
        self.manager = TokenManager(num_gpus, total_tokens=total_power_tokens)
        self.scarcity_factor = scarcity_factor  # To dynamically adjust token value

    def run(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()

            for time_step in range(500):
                # Each agent calculates their actions based on current state
                actions = [agent.act(state) for agent in self.agents]

                # Collect power requests from agents (simulate workload based power needs)
                power_requests = [np.random.uniform(50, 200) for _ in range(self.num_gpus)]

                # Token Manager distributes tokens based on requests
                allocated_power = self.manager.distribute_tokens(power_requests)

                # Environment steps forward using allocated power
                next_state, rewards, _ = self.env.step(actions)

                # Adjust workload based on distributed tokens (allocated power)
                next_state = self.manager.distribute_tokens(allocated_power)

                # Each agent trains based on feedback from the environment
                for i, agent in enumerate(self.agents):
                    agent.train_model(state, actions[i], rewards[i], next_state, False)

                state = next_state

            if episode % 10 == 0:
                self.env.render()
                print(f"Episode {episode} completed")

            # Adjust dynamic pricing after every episode based on scarcity factor
            self.manager.dynamic_pricing(self.scarcity_factor)

class EIADataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = f"https://api.eia.gov/v2/"

    def fetch_electricity_data(self, region="US"):
        url = f"{self.base_url}electricity/rto/operating/?api_key={self.api_key}&frequency=hourly&region={region}"
        response = requests.get(url)
        return response.json()

    def fetch_prices(self):
        # Fetch price-related data
        pass
