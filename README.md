# Optimizing Power Efficiency in Datacenters using Multi-Agent Reinforcement Learning (MARL) with Cerebras
The inspiration for this project comes from the following paper: “Carbon Connect: An Ecosystem for Sustainable Computing”, co-authored by Professor Lee at the University of Pennsylvania. The paper suggests an intelligent power management system for data centers, known as a cognitive stack system. <br>
The cognitive stack system would be a multi-agent system where each gpu is modeled as an agent that spends tokens for power. Each agent learns a policy for spending tokens based on their state (workload, power needs, token availability) and the environment’s state (renewable energy availability, token prices). Note that the token prices are dynamic, i.e. that are more expensive when renewable energy is scarce. As a result, the cognitive stack system helps agents to balance immediate performance and long-term resource management. The system leverages Cerebras Wafer-Scale Engine (CSX) for efficient training of the RL agents, allowing for scalability in large-scale data centers with high-dimensional state spaces. More importantly, Cerbras fast-inference is key to enabling GPU agents to make power management decisions fast enough to handle real-time demands.

## Class Summaries:
### RLAgent Class:
Represents a GPU agent in the data center.
Each agent learns a Q-learning policy to request power tokens and allocate power efficiently.
The agent's decision-making is based on its current state (workload, power consumption) and future predictions of power needs.
The neural network is trained on the Cerebras Wafer-Scale Engine using Cerebras PyTorch (cstorch).
It calculates the Q-values, selects the best action (e.g., requesting more power or reducing workload), and updates the policy based on rewards using reinforcement learning.
### Token Manager Class:
Manages the distribution of power tokens to GPU agents.
Allocates tokens based on the agents' power requests and the total available power.
Implements a dynamic pricing mechanism: if renewable energy is scarce, the token prices increase, forcing agents to use power more efficiently or defer tasks.
The goal is to maintain overall data center sustainability by encouraging agents to cooperate and reduce power usage during high-demand or low-energy periods.
### DataCenterEnvironment Class:
Simulates the overall environment, coordinating the interaction between GPU agents and the Token Manager.
Oversees the state of the entire data center, including the global power consumption, workloads, and renewable energy availability.