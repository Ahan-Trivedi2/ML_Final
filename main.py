# Import Necessary Libraries
import random  # For sampling random actions and experiences
import torch  # PyTorch for building and training neural networks
import torch.nn as nn  # For using PyTorch neural network components
import numpy as np  # For using numerical operations
from collections import deque # For efficient storage of replay memory (double ended queue)
from gym_super_mario_bros import make # Make is a function provided by gym-super-mario-bros to create an instance of the Mario environment
from nes_py.wrappers import JoypadSpace # JoypadSpace is a wrapper from the nes_py library used to modify the action space of the NES environment
from tqdm import tqdm # Used to show the progress of training across multiple episodes
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation # Wrapper for preproccessing 
# Manually import RIGHT_ONLY --> Reduced action space ot only meaningful actions for moving Mario to the right
RIGHT_ONLY = [
    ['NOOP'],       # Do nothing
    ['right'],      # Move right
    ['right', 'A'], # Jump while moving right
    ['right', 'B'], # Run while moving right
    ['right', 'A', 'B'] # Run and jump while moving right
]

# Define a Deep Q-Network (DQN) class for estimating Q-values
class DQNSolver(nn.Module):
# Constructor
    def __init__(self, input_shape, n_actions): # Initialization
        super(DQNSolver, self).__init__() # Initialize the PyTorch nn.Module
        # Convolution layers to extract features from game frames
        self.conv = nn.Sequential( # apply sequential convolutions
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), # takes 4x84x84 input and turns it into 32 feature maps
            nn.ReLU(), # Activation function for nonlinearity
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # 32 feature maps --> 64 feature maps
            nn.ReLU(), # Activation function for nonlinearity
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # 64 feature maps --> 64 feature maps
            nn.ReLU() # Activation function for nonlinearity
        )
        # Calculate size of output from the convolution layers
        conv_out_size = self._get_conv_out(input_shape)
        # Define fully connected layers to output Q-values
        self.fc = nn.Sequential( # apply a linear layer
            nn.Linear(conv_out_size, 512), # Flattened convolution vector is passes into 512 neurons
            nn.ReLU(), # Activation function for non-linearity
            nn.Linear(512, n_actions) # Output neuron is action, and neuron value is q-value for that action 
        )
    def _get_conv_out(self, shape):
        """Calculate the flattened size of the convolutional output"""
        with torch.no_grad(): # we do not need to calculate gradients here
            o = self.conv(torch.zeros(1, *shape)) # Simulates a forward pass with dummy data
            return int(np.prod(o.size()))  # Return the flattened size of the output
    def forward(self, x):
        """Forward pass to process input through CNN and connected layers"""
        conv_out = self.conv(x).view(x.size()[0], -1)  # Run convolutions then flatten the output
        return self.fc(conv_out) # Pass through fully connected layers 

# Define a DQN Agent class that interacts with the environment and learns
class DQNAgent:
# Constructor 
    def __init__(self, state_space, action_space, max_replay_memory_size, batch_size, gamma, lr, exploration_max, exploration_min, exploration_decay):
        self.state_space = state_space # The shape of the input state
        self.action_space = action_space # The number of actions available
        self.memory = deque(maxlen=max_replay_memory_size) # Replay memory buffer with a fixed max value for the deque
        self.batch_size = batch_size # Batch size for training (how many experiences to sample from the replay memory during training.)
        self.gamma = gamma # Discount factor for future rewards ???
        self.exploration_rate = exploration_max  # Initial exploration rate
        self.exploration_min = exploration_min # Minimum exploration rate
        self.exploration_decay = exploration_decay # Rate at which exploration decays
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.local_net = DQNSolver(state_space, action_space).to(self.device) # Network used to estimate Q-values ????
        self.target_net = DQNSolver(state_space, action_space).to(self.device) # Target network for stable updates ????
        self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)  # Adam optimizer for training  
    def act(self, state):
        """The act mechanism is the decision making mechanism of the agent. 
        It uses an epsilon-greedy policy to balance exploration (trying new actions) 
        and exploitation (choosing the best-known action based on the current Q-values)."""
        # Random exploration based on the exploration rate
        if random.random() < self.exploration_rate:  # With probability epsilon, explore
            return torch.tensor([[random.randrange(self.action_space)]], dtype=torch.long).to(self.device)
        # Exploitation based on Q-values from the network
        with torch.no_grad():  # No gradient calculation for action selection
            q_values = self.local_net(state)  # Get Q-values for the current state
            return torch.argmax(q_values, dim=1, keepdim=True)  # Choose the action with the highest Q-value
    def remember(self, state, action, reward, next_state, terminal):
        """Store experiences in replay memory"""
        self.memory.append((state, action, reward, next_state, terminal)) # Append experience tuple to memory
    def experience_replay(self):
        """Trains the local network using experiences sampled from the replay memory"""
        if len(self.memory) < self.batch_size: # If there aren't enough previous experiences to form a full batch, the method returns without training
            return
        batch = random.sample(self.memory, self.batch_size) # Randomly selects self.batch_size experiences from replay memory
        states, actions, rewards, next_states, terminals = zip(*batch) # Unpacks the list of experience tuples into five seperate lists
        # Converts these lists to PyTorch tensors and moves them to device
        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        terminals = torch.cat(terminals).to(self.device)
        # Double DQN: Use local_net to select actions, target_net to evaluate
        next_actions = self.local_net(next_states).argmax(dim=1, keepdim=True) # Select the best action for each next_state
        target_q_values = rewards + self.gamma * self.target_net(next_states).gather(1, next_actions) * (1 - terminals) # calculates the target Q-value for each state-action pair in the batch using the target network (target_net)
        current_q_values = self.local_net(states).gather(1, actions) # Uses the batch of random states to compute Q-values for the actions that were taken in these states
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values) # Calculate the Huber Loss ???
        # Backpropagate and update the model (local_net)
        self.optimizer.zero_grad()  # Reset gradients to prevent accumulation
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Update weights
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay) # Reduce exploration rate
    def copy_model(self):
        """Copy the weights of the local model to the target model"""
        # Synchronize the weights
        self.target_net.load_state_dict(self.local_net.state_dict()) # Loads in the local_net state_dicts (learnable paramters) into the target net to synchronize the models

# Define a function to preprocess the environment
def make_env():
    env = make('SuperMarioBros-1-1-v3') # Create the mario environment
    env = JoypadSpace(env, RIGHT_ONLY) # Restrict actions to RIGHT_ONLY
    env = GrayScaleObservation(env, keep_dim=False) # Convert frames to grayscale (with 1 dimension rather than 3)
    env = ResizeObservation(env, (84,84)) # Resize frames to 84x84
    env = FrameStack(env, 4) # Stack the last four frames
    return env

# Main training loop
def run():
    env = make_env()  # Initialize the environment using make_env function
    state_space = (4, 84, 84)  # Represents the input shape for the neural network (4, 84, 84)
    action_space = env.action_space.n  # Number of possible actions (e.g., 5 for RIGHT_ONLY)
    # Initialize agent with hyperparameters
    agent = DQNAgent(
        state_space=state_space,
        action_space=action_space,
        max_replay_memory_size=30000,
        batch_size=32,
        gamma=0.99,
        lr=0.00025,
        exploration_max=1.0,
        exploration_min=0.02,
        exploration_decay=0.99,
    )
    num_episodes = 1000  # Total number of episodes to train
    for episode in tqdm(range(num_episodes)):  # Loop through episodes with progress bar
        obs = env.reset()  # Unpack observation and dictionary
        state = torch.tensor([obs], dtype=torch.float32).to(agent.device)  # Convert the initial state to a PyTorch tensor
        total_reward = 0  # Initialize total reward for the episode
        while True:
            env.render() # Render the environment to see the game screen
            action = agent.act(state)  # Choose an action
            next_state, reward, terminal, truncated = env.step(action.item())  # Take the action in the environment
            done = terminal or truncated  # Determine if the episode has ended
            total_reward += reward  # Accumulate the reward
            next_state = torch.tensor([next_state], dtype=torch.float32).to(agent.device)  # Convert the next state to a tensor
            reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0).to(agent.device)  # Reward as tensor
            done = torch.tensor([done], dtype=torch.float32).unsqueeze(0).to(agent.device)  # Done state as tensor
            # Store the experience and train the agent
            agent.remember(state, action, reward, next_state, done)
            agent.experience_replay()
            state = next_state  # Update the current state
            if done:  # Break the loop if the episode ends
                break
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")  # Print episode results
        if episode % 10 == 0:  # Every ten episodes, update the target network
            agent.copy_model()
    env.close()  # Close the environment when training is done

# Run the training loop if this file is executed
if __name__ == "__main__":
    run()
