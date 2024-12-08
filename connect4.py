class ConnectFour():
    BLANK = 0
    RED = 1
    YELLOW = 2
    
    def __init__(self) -> None:
        self.grid = [[ConnectFour.BLANK for _ in range(6)] for _ in range(7)]
        self.turn = ConnectFour.YELLOW
        
    def get_actions(self) -> int:
        return [i for i in range(7) if self.grid[i][-1] == ConnectFour.BLANK]
    
    def change_turn(self):
        self.turn = 1 if self.turn == 2 else 2
    
    def move(self, column):
        for i in range(6):
            if self.grid[column][i] == ConnectFour.BLANK:
                self.grid[column][i] = self.turn
                self.change_turn()
                return
    
    def check_draw(self) -> bool:
        for i in range(7):
            if self.grid[i][5] == ConnectFour.BLANK:
                return False
        return True
    
    def check_win(self, turn) -> bool:
        # Check horizontal
        for row in range(6):
            for col in range(4):  # Only need to check up to column 3
                if (self.grid[col][row] == turn and 
                    self.grid[col+1][row] == turn and 
                    self.grid[col+2][row] == turn and 
                    self.grid[col+3][row] == turn):
                    return True

        # Check vertical
        for col in range(7):
            for row in range(3):  # Only need to check up to row 2
                if (self.grid[col][row] == turn and 
                    self.grid[col][row+1] == turn and 
                    self.grid[col][row+2] == turn and 
                    self.grid[col][row+3] == turn):
                    return True

        # Check diagonal (bottom-left to top-right)
        for col in range(4):  # Only need to check up to column 3
            for row in range(3):  # Only need to check up to row 2
                if (self.grid[col][row] == turn and 
                    self.grid[col+1][row+1] == turn and 
                    self.grid[col+2][row+2] == turn and 
                    self.grid[col+3][row+3] == turn):
                    return True

        # Check diagonal (top-left to bottom-right)
        for col in range(4):  # Only need to check up to column 3
            for row in range(3, 6):  # Start checking from row 3 to row 5
                if (self.grid[col][row] == turn and 
                    self.grid[col+1][row-1] == turn and 
                    self.grid[col+2][row-2] == turn and 
                    self.grid[col+3][row-3] == turn):
                    return True
        return False