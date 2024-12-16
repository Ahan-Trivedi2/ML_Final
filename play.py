from ticTacToeQlearningEnv import TicTacToeEnv
import numpy as np

Q_table = np.load('q_table.npy')


env = TicTacToeEnv()

done = False

current_state = env.reset()
env.print_board()

while not done:
    i = int(input("Play> "))
    current_state, reward, done, _ = env.step(i)
    env.print_board()
    action = np.argmax(Q_table[current_state])
    current_state, reward, done, _ = env.step(action)
    env.print_board()