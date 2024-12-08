import os
from connect_four_env import Connect4Env
from dqn import Agent
import torch

def play_against_agent():
    env = Connect4Env()
    state_size = env.rows * env.columns
    action_size = env.columns
    agent = Agent(state_size, action_size)

    # Load the trained model
    model_path = "connect4_dqn.pth"
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path))
        agent.model.eval()
        print("Loaded trained model.")
    else:
        print("No trained model found. Exiting.")
        return

    state = env.reset()
    print("Welcome to Connect 4!")
    env.render()

    while not env.done:
        # Human move
        valid_moves = [c for c in range(env.columns) if env.is_valid_action(c)]
        human_move = -1
        while human_move not in valid_moves:
            try:
                human_move = int(input(f"Your move (valid columns: {valid_moves}): "))
            except ValueError:
                print("Invalid input. Please enter a valid column number.")

        state, reward, done, _ = env.step(human_move)
        env.render()

        if done:
            if reward == 1:
                print("You win!")
            elif reward == 0:
                print("It's a draw!")
            break

        # Agent's move
        print("Agent's turn...")
        state_flat = state.flatten()
        agent_move = agent.act(state_flat)
        state, reward, done, _ = env.step(agent_move)
        env.render()

        if done:
            if reward == 1:
                print("Agent wins!")
            elif reward == 0:
                print("It's a draw!")
            break


if __name__ == "__main__":
    play_against_agent()