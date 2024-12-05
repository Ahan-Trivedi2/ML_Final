import torch
import numpy as np
from main import DQNSolver, ACTION_SPACE
import time
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation # Wrapper for preproccessing 	
from gymnasium import make # Make is a function provided by gym-super-mario-bros to create an instance of the Mario environment

# Action mapping for readability (adjust to your ACTION_SPACE)
ACTION_MAP = {
    0: "NOOP",
    3: "UP",
    6: "RIGHT",
    9: "LEFT",
    12: "DOWN",
    15: "A"
}

def make_env():
    env = make('ALE/MarioBros-v5', frameskip=4, repeat_action_probability=0.25, render_mode="human")  # Specify render_mode
    env = GrayscaleObservation(env)  # Convert frames to grayscale
    env = ResizeObservation(env, (84, 84))  # Resize frames to 84x84
    env = FrameStackObservation(env, stack_size=4)  # Stack the last four frames
    return env


# Load the trained weights into a DQNSolver model
def load_model(filepath, input_shape, n_actions):
    model = DQNSolver(input_shape, n_actions)
    model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))  # Load weights to CPU or GPU
    model.eval()  # Set model to evaluation mode (disables dropout, batchnorm, etc.)
    return model

# Define a function to evaluate the trained model with rendering
def evaluate_model_with_rendering(env, model, action_space, num_episodes=5, delay=0.0005):
    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}")
        obs, _ = env.reset()
        state = torch.tensor(np.array([obs]), dtype=torch.float32)
        total_reward = 0
        while True:
            # Render the environment
            env.render()

            # Get the action from the model
            with torch.no_grad():
                q_values = model(state)
                action_idx = torch.argmax(q_values).item()  # Choose the action with the highest Q-value
            action = action_space[action_idx]

            # Take the action in the environment
            next_obs, reward, terminal, truncated, _ = env.step(action)
            total_reward += reward

            # Display the action
            print(f"Action: {ACTION_MAP.get(action, 'UNKNOWN')}")

            # Update the state
            state = torch.tensor(np.array([next_obs]), dtype=torch.float32)

            # Check if the episode has ended
            if terminal or truncated:
                break

            # Add a delay for better visualization
            time.sleep(delay)

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")


def run(path):
    # Load your environment
    env = make_env()

    # Define the input shape and number of actions based on your training configuration
    state_space = (4, 84, 84)
    action_space = ACTION_SPACE

    # Load the model
    trained_model = load_model("dqn_mario_final.pth", state_space, len(action_space))

    # Evaluate the model with rendering
    evaluate_model_with_rendering(env, trained_model, action_space)

    # Close the environment
    env.close()


if __name__ == "__main__":
    run("dqn_mario.pth")

"""
# Load the trained weights into a DQNSolver model
def load_model(filepath, input_shape, n_actions):
    model = DQNSolver(input_shape, n_actions)
    model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))  # Load weights to CPU or GPU
    model.eval()  # Set model to evaluation mode (disables dropout, batchnorm, etc.)
    return model

# Define a function to evaluate the trained model
def evaluate_model(env, model, action_space, num_episodes=10):
    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = torch.tensor(np.array([obs]), dtype=torch.float32)
        total_reward = 0
        while True:
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()  # Choose the action with the highest Q-value
            next_obs, reward, terminal, truncated, _ = env.step(action)
            total_reward += reward
            state = torch.tensor(np.array([next_obs]), dtype=torch.float32)
            if terminal or truncated:
                break
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Load your environment
env = make_env()

# Define the input shape and number of actions based on your training configuration
state_space = (4, 84, 84)
action_space = len(ACTION_SPACE)

# Load the model
trained_model = load_model("dqn_mario_final.pth", state_space, action_space)

# Evaluate the model
evaluate_model(env, trained_model, ACTION_SPACE)

# Close the environment
env.close()
"""