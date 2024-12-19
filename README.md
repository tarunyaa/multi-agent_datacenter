# Optimizing Power Efficiency in Datacenters using Multi-Agent Reinforcement Learning (MARL) with Cerebras
The inspiration for this project comes from the following paper: “Carbon Connect: An Ecosystem for Sustainable Computing”, co-authored by Professor Lee at the University of Pennsylvania. The paper suggests an intelligent power management system for data centers, known as a cognitive stack system. <br>
The cognitive stack system would be a multi-agent system where each gpu is modeled as an agent that spends tokens for power. Each agent learns a policy for spending tokens based on their state (workload, power needs, token availability) and the environment’s state (renewable energy availability, token prices). Note that the token prices are dynamic, i.e. that are more expensive when renewable energy is scarce. As a result, the cognitive stack system helps agents to balance immediate performance and long-term resource management. The system leverages Cerebras Wafer-Scale Engine (CSX) for efficient training of the RL agents, allowing for scalability in large-scale data centers with high-dimensional state spaces. More importantly, Cerbras fast-inference is key to enabling GPU agents to make power management decisions fast enough to handle real-time demands.

## Class Summaries:
### Agent Class:
Represents a GPU agent in the data center.
Each agent learns a Q-learning policy to optimize its token usage for power allocation.
Decision-making considers the agent’s current state (e.g., workload and power consumption) and future power needs, balancing immediate and long-term performance.
A deep Q-network (DQN) is trained based on training data obtained from an initial round of Q-learning to estimate Q-values for state-action pairs.
The agent selects actions (e.g., requesting more power or conserving resources) and updates its policy based on rewards computed from its efficiency and performance.
### Token Manager Class:
Distributes power tokens based on agents’ requests and the total available power supply.
Implements a dynamic pricing mechanism where token costs rise when renewable energy is scarce, encouraging agents to conserve power or defer tasks.
Plays a central role in maintaining overall data center sustainability by incentivizing agents to reduce consumption during periods of high demand or limited renewable energy availability.
### DataCenterEnvironment Class:
Simulates the overall environment, coordinating the interaction between GPU agents and the Token Manager.
Oversees the state of the entire data center, including the global power consumption, workloads, and renewable energy availability.