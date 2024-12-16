import numpy as np
import random


class TicTacToeEnv:
    X = 1
    O = 2
    BLANK = 0
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = self.X  # Player 1 always starts (X)
        self.done = False
        self.winner = None
        return self.get_state()

    def step(self, action):
        reward: float
        if self.done:
            raise ValueError("Game is already done.")
        if self.board[action] != 0:
            # Invalid move: end game and penalize
            # print(f"Invalid move {(self.board == 0).sum()}", end="\t")
            reward = -10
            self.done = True
            self.winner = -self.current_player # TODO: Delete
            return self.get_state(), reward, self.done, {}
        
        self.board[action] = self.current_player

        if TicTacToeEnv.check_winner(self.board, self.current_player):
            # print("this is a win", end="\t")
            self.done = True
            self.winner = self.current_player
            reward = 100
        elif (self.board == 0).sum() == 0:
            # print("this is a draw", end="\t")
            # Draw
            self.done = True
            self.winner = 0
            reward = 0
        else:
            # print("valid move")
            reward = 0
        
        self.current_player = self.X if self.current_player == self.O else self.O
        
        return self.get_state(), reward, self.done, {}

    def check_draw(self):
        return (self.board == TicTacToeEnv.BLANK).sum() == 0

    def check_loss(self):
        for i in range(9):
                #print(self.board)
                #print(self.board)
                if self.board[i] == 0:
                    self.board[i] == self.current_player
                    if TicTacToeEnv.check_winner(self.board, self.current_player):
                        return True
                    self.board[i] == 0
        return False

    def get_state(self):
        j = 0
        for i in range(9):
            j *= 3
            j += self.board[i]
        return j

    def update_board(self, j):
        for i in range(8,-1,-1):
            self.board[i] = j % 3
            j /= 3

    def check_winner(board, player):
        wins = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]
        for combo in wins:
            if all(board[i] == player for i in combo):
                return True
        return False


    def valid_actions(self):
        return [i for i in range(9) if self.board[i] == 0]
    
    def print_board(self):
        symbols = {self.X: "X", self.O: "O", self.BLANK: " "}
        for i in range(0, 9, 3):
            row = [symbols[self.board[i+j]] for j in range(3)]
            print(" | ".join(row))
            if i < 6:
                print("---------")
