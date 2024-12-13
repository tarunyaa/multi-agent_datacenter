from agent import Agent
from token_manager import TokenManager
from datacenter_environment import DataCenterEnvironment

if __name__ == "__main__":
    # Simulation setup
    num_agents = 2
    token_manager = TokenManager(base_price=1.0, scaling_factor=10)
    env = DataCenterEnvironment(num_agents, token_manager)
    agents = [Agent(i) for i in range(num_agents)]

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        actions = []
        for agent in agents:
            state = int(env.states[agent.id])
            action = agent.select_action(state)
            actions.append(action)

        rewards = env.step(actions)

        for i, agent in enumerate(agents):
            state = int(env.states[i])
            next_state = int(env.states[i])  # Simplified
            agent.update_q_table(state, actions[i], rewards[i], next_state)

        print(f"Epoch {epoch + 1}, Rewards: {rewards}, Tokens: {token_manager.tokens}")

    print("Training complete!")
