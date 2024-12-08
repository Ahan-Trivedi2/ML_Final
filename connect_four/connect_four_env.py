import numpy as np

class Connect4Env:
    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.done = False
        self.current_player = 1

    def reset(self):
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.done = False
        self.current_player = 1
        return self.board

    def step(self, action):
        # Check if the column is valid
        if self.done or not self.is_valid_action(action):
            raise ValueError("Invalid move or game is already over!")

        # Drop the piece in the column
        row = self.get_next_open_row(action)
        self.board[row][action] = self.current_player

        # Check for a win
        if self.check_win(self.current_player):
            self.done = True
            reward = 1
        elif self.is_draw():
            self.done = True
            reward = 0
        else:
            reward = 0

        # Switch players
        self.current_player = 3 - self.current_player
        return self.board, reward, self.done, {}

    def is_valid_action(self, action):
        return self.board[0][action] == 0

    def get_next_open_row(self, column):
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][column] == 0:
                return r

    def is_draw(self):
        return not any(self.is_valid_action(c) for c in range(self.columns))

    def check_win(self, player):
        # Check horizontal, vertical, and diagonal wins
        for c in range(self.columns - 3):
            for r in range(self.rows):
                if all(self.board[r][c + i] == player for i in range(4)):
                    return True
        for c in range(self.columns):
            for r in range(self.rows - 3):
                if all(self.board[r + i][c] == player for i in range(4)):
                    return True
        for c in range(self.columns - 3):
            for r in range(self.rows - 3):
                if all(self.board[r + i][c + i] == player for i in range(4)):
                    return True
        for c in range(self.columns - 3):
            for r in range(3, self.rows):
                if all(self.board[r - i][c + i] == player for i in range(4)):
                    return True
        return False

    def render(self):
        print(np.flip(self.board, 0))
