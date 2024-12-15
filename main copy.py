# Figure out how to manage all these dependencies so don't have to import in every file
from agent import Agent
from token_manager import TokenManager
from datacenter_environment import DataCenterEnvironment
import numpy as np
import matplotlib.pyplot as plt
import pickle
from training import QNetwork
import torch
import random

# Load trained model
trained_model = QNetwork()
trained_model.load_state_dict(torch.load("q_network.pth"))
trained_model.eval()

# Replace Q-table action selection with DNN inference
def select_action_with_dnn(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(2))  # Explore
    else:
        # Predict Q-values for both actions
        state_tensor = torch.tensor([state], dtype=torch.float32).view(-1, 1)
        actions_tensor = torch.tensor([0, 1], dtype=torch.float32).view(-1, 1)
        inputs = torch.cat([state_tensor.repeat(2, 1), actions_tensor], dim=1)
        q_values = trained_model(inputs).detach().numpy().flatten()
        return int(np.argmax(q_values))  # Exploit

if __name__ == "__main__":
    # Simulation setup
    num_agents = 2
    max_tokens=100
    scaling_factor = 200000000
    token_manager = TokenManager(base_price=1.0, scaling_factor=scaling_factor)
    agents = [Agent(i, max_tokens) for i in range(num_agents)]
    env = DataCenterEnvironment(agents, token_manager, max_tokens)

    # Training parameters
    num_epochs = 10
    num_training_episodes = 50
    
    # Exploration parameters
    max_epsilon = 1.0           
    min_epsilon = 0.05
    decay_rate = 0.5
    
    episodes_rewards = []
    collected_data = [] # To store state-action-reward-next state tuples
    
    for episode in range(num_training_episodes):
        print(f"Episode: {episode + 1}")
        env.reset()
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        # epsilon = 0.01
        epochs_rewards = []
        tokens_left = []
        total_episode_rewards = 0
        # Reset
        # epoch = 0
        # state = env.reset()
        # done = False
        
        # Training loop
        for epoch in range(num_epochs):
            actions = []
            
            for agent in agents:
                state = int(env.states[agent.id])
                action = agent.select_action(state, epsilon)
                actions.append(action)

            # The environment runs the chosen actions and returns the reward
            rewards = env.step(actions)
            epochs_rewards.append(sum(rewards))  # Track total rewards for the epoch

            for i, agent in enumerate(agents):
                next_state = int(env.states[i])  # Current tokens become the next state
                agent.update_q_table(state, actions[i], rewards[i], next_state)
                collected_data.append((state, actions[i], rewards[i], next_state))
            
            # Format rewards and tokens for display
            formatted_rewards = [round(float(r), 2) for r in rewards]
            formatted_tokens = env.states.tolist()

            print(f"Epoch {epoch + 1}, Rewards: {formatted_rewards}, Tokens: {env.states}")
            print("-----------------------------------------------------")
            tokens_left.append(sum(env.states))
        total_episode_rewards = sum(rewards)
        print(f"Episode {episode} Reward: {total_episode_rewards:.2f}")
        episodes_rewards.append(total_episode_rewards)
            

    print("Training complete!")
    # for agent in agents:
    #     print(f"Agent {agent.id} Q-Table:\n{agent.q_table}")

    plt.plot(episodes_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.show()
    
    # Save data to a file for external training
    with open("collected_data.pkl", "wb") as f:
        pickle.dump(collected_data, f)

    # plt.plot(epochs_rewards)
    # plt.xlabel("Epoch")
    # plt.ylabel("Total Reward")
    # plt.title("Reward per Epoch")
    # plt.show()


    # plt.plot(tokens_left)
    # plt.xlabel("Epoch")
    # plt.ylabel("Average Tokens Left")
    # plt.title("Tokens Left per Epoch")
    # plt.show()
