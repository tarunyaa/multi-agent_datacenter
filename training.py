import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# Load data
with open("collected_data_test.pkl", "rb") as f:
    data = pickle.load(f)

# Prepare data for training
states, actions, rewards, next_states = zip(*data)

# Convert to tensors
states = torch.tensor(states, dtype=torch.float32).view(-1, 1)
actions = torch.tensor(actions, dtype=torch.float32).view(-1, 1)
rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
next_states = torch.tensor(next_states, dtype=torch.float32).view(-1, 1)

# Combine state and action for input
inputs = torch.cat([states, actions], dim=1)

# Define DNN model
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 32)  # State and action input size
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)  # Output: Q-value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = QNetwork()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Training loop
gamma = 0.9  # Discount factor
batch_size = 32
num_epochs = 100

for epoch in range(num_epochs):
    for i in range(0, len(inputs), batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_rewards = rewards[i:i+batch_size]
        batch_next_states = next_states[i:i+batch_size]

        # Predict Q(s, a) - what the neural net outputs
        q_values = model(batch_inputs)

        # Compute target: r + γ * max(Q(s', a'))
        # Target is what we want the neural net to output, based on Bellman equation
        # Making our own labels - using network's own predictions of the next state's Q-values to form a moving target
        with torch.no_grad():
            next_q_values = model(torch.cat([batch_next_states, actions[i:i+batch_size]], dim=1))
            targets = batch_rewards + gamma * next_q_values

        # Backpropagation
        loss = criterion(q_values, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "q_network.pth")
