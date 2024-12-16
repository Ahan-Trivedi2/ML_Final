import math
import numpy as np
from ticTacToeQlearningEnv import TicTacToeEnv

# Define the environment
n_states =   pow(3, 9)# Number of states in the grid world
n_actions = 9  # Number of possible actions (up, down, left, right)

env = TicTacToeEnv()

# Initialize Q-table with zeros
Q_table = np.zeros((n_states, n_actions))

# Define parameters
learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
epochs = 100000

done = False

# Q-learning algorithm
for epoch in range(epochs):
    current_state = env.reset()  # Start from a random state
    done = False
    
    if not epoch % 1000:
        print(f"epoch: {epoch}")
    
    while not done:
        # Choose action with epsilon-greedy strategy
        if np.random.rand() < exploration_prob:
            action = np.random.randint(0, n_actions)  # Explore
        else:
            action = np.argmax(Q_table[current_state])  # Exploit

        # Simulate the environment (move to the next state)
        # For simplicity, move to the next state
        next_state, reward, done, _ = env.step(action)

        # Define a simple reward function (1 if the goal state is reached, 0 otherwise)
        # reward = reward

        # Update Q-value using the Q-learning update rule
        Q_table[current_state, action] += learning_rate * \
            (reward - discount_factor *
             np.max(Q_table[next_state]) - Q_table[current_state, action])

        current_state = next_state  # Move to the next state
        
    
np.save('q_table.npy', Q_table)