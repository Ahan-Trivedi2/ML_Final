from connect_four_env import Connect4Env
from dqn import Agent
import torch

def train():
    env = Connect4Env()
    state_size = env.rows * env.columns
    action_size = env.columns
    agent = Agent(state_size, action_size)
    episodes = 100
    batch_size = 32

    for e in range(episodes):
        state = env.reset().flatten()  # Start with a fresh game
        done = False
        try:
            while not done:
                # Agent chooses an action
                action = agent.act(state)

                # Take the action in the environment
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.flatten()

                # Save the experience in memory
                agent.remember(state, action, reward, next_state, done)

                # Update the current state
                state = next_state

                # Perform experience replay
                agent.replay(batch_size)

            # Update the target model after each episode
            agent.update_target_model()
            print(f"Episode {e}/{episodes}, Reward: {reward}, Epsilon: {agent.epsilon:.2f}")

        except ValueError as ve:
            print(f"Episode {e} terminated early due to error: {ve}")

    # Save the model after training
    model_path = "connect4_dqn.pth"
    torch.save(agent.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    train()
