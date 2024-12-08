import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from connect4 import ConnectFour

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ConnectFourAgent:
    def __init__(self):
        self.env = ConnectFour()
        self.model = DQN(42, 7)  # 6x7 grid flattened to 42 inputs, 7 actions
        self.target_model = DQN(42, 7)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.replay_buffer = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 64
    
    def preprocess_state(self, state):
        # Flatten the 6x7 grid
        return np.array(state.grid).flatten()
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(self.env.get_actions())
        state_tensor = torch.tensor(self.preprocess_state(state), dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state_tensor)
        valid_actions = self.env.get_actions()
        q_values = q_values.detach().numpy().flatten()
        q_values = [(q_values[i] if i in valid_actions else -float('inf')) for i in range(7)]
        return np.argmax(q_values)
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            target_q_values = rewards + self.gamma * self.target_model(next_states).max(1)[0] * (1 - dones)
        
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, episodes):
        sum = 0
        count = 0
        self.advesary = ConnectFourAgent
        for episode in range(episodes):
            state = self.env
            total_reward = 0
            while True:
                action = self.choose_action(state)
                next_state = ConnectFour()
                next_state.grid = [row[:] for row in state.grid]
                next_state.turn = state.turn
                next_state.move(action)
                
                reward = 0
                if next_state.check_win(ConnectFour.YELLOW):
                    reward = 1
                elif next_state.check_draw():
                    reward = 0
                elif next_state.check_win(ConnectFour.RED):
                    reward = -1
                
                done = reward != 0 or next_state.check_draw()
                
                self.replay_buffer.append((self.preprocess_state(state), action, reward, self.preprocess_state(next_state), done))
                state = next_state
                total_reward += reward
                
                self.train_step()
                
                if done:
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                    break
            sum += reward
            count += 1
            if episode % 100 == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                print(f"Episode {episode}, Average Reward: {sum / count}")
                sum = 0
                count = 0


if __name__ == "__main__":
    # Initialize and train the agent
    agent = ConnectFourAgent()
    agent.train(500)

    # Save the model
    torch.save(agent.model.state_dict(), "connect4_dqn.pth")
    print("Model saved as connect4_dqn.pth")