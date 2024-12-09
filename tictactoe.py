import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ---------------------------
# TicTacToe environment
# ---------------------------
class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1  # Player 1 always starts (X)
        self.done = False
        self.winner = None
        return self.get_state()

    def step(self, action):
        if self.done:
            raise ValueError("Game is already done.")
        if self.board[action] != 0:
            # Invalid move: end game and penalize
            reward = -1.0
            self.done = True
            self.winner = -self.current_player
            return self.get_state(), reward, self.done, {}
        
        self.board[action] = self.current_player

        if self.check_winner(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif (self.board == 0).sum() == 0:
            # Draw
            self.done = True
            self.winner = 0
            reward = 0.5
        else:
            reward = 0.0

        self.current_player = -self.current_player
        return self.get_state(), reward, self.done, {}

    def get_state(self):
        return self.board.copy()

    def check_winner(self, player):
        wins = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]
        for combo in wins:
            if all(self.board[i] == player for i in combo):
                return True
        return False

    def valid_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

# ---------------------------
# DQN Network
# ---------------------------
class DQN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=9):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------------------------
# Replay Memory
# ---------------------------
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ---------------------------
# Helper Functions
# ---------------------------
def select_action(state, policy_net, epsilon, env):
    if random.random() < epsilon:
        # Explore
        valid_moves = env.valid_actions()
        return random.choice(valid_moves)
    else:
        # Exploit
        with torch.no_grad():
            q_values = policy_net(torch.FloatTensor(state).unsqueeze(0))
            valid_moves = env.valid_actions()
            q_vals_valid = q_values[0, valid_moves]
            max_action = valid_moves[q_vals_valid.argmax().item()]
            return max_action

def update_network(policy_net, target_net, optimizer, batch, gamma=0.99):
    states, actions, next_states, rewards, dones = zip(*batch)
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

    # Current Q values
    q_values = policy_net(states).gather(1, actions)

    # Compute target Q values
    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * gamma * max_next_q_values

    loss = F.mse_loss(q_values, target_q_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    for i in range(0, 9, 3):
        row = board[i:i+3]
        print("|".join(symbols[x] for x in row))
        if i < 6:
            print("-----")

# ---------------------------
# Play against saved model
# ---------------------------
def play_against_model(model_path):
    env = TicTacToeEnv()

    # Load the model
    policy_net = DQN()
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    state = env.reset()
    human_player = 1
    ai_player = -1

    while not env.done:
        print("\nCurrent board:")
        print_board(env.board)
        if env.current_player == human_player:
            # Human turn
            valid_moves = env.valid_actions()
            print("Your turn! Valid moves:", valid_moves)
            move = None
            while move not in valid_moves:
                try:
                    move = int(input("Enter your move (0-8): "))
                except:
                    continue
            next_state, reward, done, info = env.step(move)
        else:
            # AI turn - no exploration (epsilon=0)
            valid_moves = env.valid_actions()
            with torch.no_grad():
                q_values = policy_net(torch.FloatTensor(state).unsqueeze(0))
                q_vals_valid = q_values[0, valid_moves]
                ai_move = valid_moves[q_vals_valid.argmax().item()]
            next_state, reward, done, info = env.step(ai_move)

        state = next_state

    print("\nGame Over!")
    print_board(env.board)
    if env.winner == 0:
        print("It's a draw!")
    elif env.winner == human_player:
        print("You win!")
    else:
        print("The AI wins!")

# ---------------------------
# Main Training Loop (uncomment to train)
# ---------------------------
if __name__ == "__main__":
    env = TicTacToeEnv()

    # Hyperparameters
    num_episodes = 100000
    gamma = 0.99
    lr = 0.001
    batch_size = 64
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 100000
    target_update = 1000

    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(10000)

    steps_done = 0

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * steps_done / epsilon_decay)
            
            action = select_action(state, policy_net, epsilon, env)
            next_state, reward, done, info = env.step(action)

            memory.push(state, action, next_state, reward, float(done))
            state = next_state

            steps_done += 1

            # Learn every step if we have enough samples
            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                loss = update_network(policy_net, target_net, optimizer, batch, gamma=gamma)

            # Update target network periodically
            if steps_done % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        if (episode+1) % 1000 == 0:
            print(f"Episode {episode+1}/{num_episodes} completed.")

    print("Training completed.")

    # Save the trained model
    torch.save(policy_net.state_dict(), "dqn_tictactoe_model.pth")
    print("Model saved as dqn_tictactoe_model.pth")

    # Example of how to play after training:
    play_against_model("dqn_tictactoe_model.pth")
