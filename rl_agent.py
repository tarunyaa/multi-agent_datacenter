import torch
import torch.nn.functional as F
import torch.optim as optim
import cerebras_pytorch as cstorch

# https://docs.cerebras.net/en/2.0.3/wsc/tutorials/custom-training-loop.html
# https://docs.cerebras.net/en/2.0.3/wsc/tutorials/e2e-example.html#end-to-end-examples

class RLAgent(nn.Module):
    """Makes decisions (actions) based on observations of the environment (states)."""
    def __init__(self, state_size, action_size, agent_id):
        super(RLAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.model = self._build_model()
        self.compiled_model = cstorch.compile(self.model, backend="CSX")
        # Stochastic gradient descent for weight updates during training, SGD chosen as optimizer
        self.optimizer = cstorch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = torch.nn.MSELoss()

    # We use a simple feed-forward neural network to approximate Q-values - DQN
    def _build_model(self):
        """Build the Q-learning neural network model."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.action_size)
        ) # 2 hidden layers of size 64, output layer of size action_size, represents Q-values for each action

    def forward(self, state):
        return self.compiled_model(state)

    def act(self, state):
        """Select an action using the Q-network by running inference on Cerebras."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # Inference on Cerebras
        q_values = self.compiled_model(state_tensor)
        # Select action with highest Q-value
        action = torch.argmax(q_values).item()  # Greedy policy for action selection
        return action

    @cstorch.trace
    # Q-learning is a RL algorithm where the agent learns a policy that maximizes
    # cumulative reward by learning Q-values for state-action pairs
    def training_step(self, state, action, reward, next_state, done):
        """Train the model using the current experience."""
        # Prepare the state and next state tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Calculate the target Q-value
        target = reward
        if not done:
            # Bellman equation: includes immediate reward and discounted future rewards
            target += 0.95 * torch.max(self.compiled_model(next_state_tensor)).item()

        # Get current Q-values for the state
        predicted_q_values = self.compiled_model(state_tensor)
        target_q_values = predicted_q_values.clone().detach()
        target_q_values[0][action] = target  # Update Q-value for the taken action

        # Compute the loss
        loss = self.criterion(predicted_q_values, target_q_values)
        
        # Perform backpropagation
        loss.backward()
        
        # Optimize the model
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

    def request_tokens(self, power_need):
        """Calculate how many tokens the agent should request."""
        return power_need  # Request as per current need (can be made more sophisticated)
