import tkinter as tk
from tkinter import messagebox
import numpy as np
import torch

from main_connectfour import ConnectFour, ConnectFourAgent  # Import the ConnectFour class and agent


class ConnectFourGame:
    def __init__(self, root):
        self.env = ConnectFour()
        
        # Use the model from ConnectFourAgent
        agent = ConnectFourAgent()
        agent.train(5000)  # Train the model
        self.model = agent.model
        self.model.eval()  # Set the model to evaluation mode
        
        self.root = root
        self.frame = tk.Frame(root)
        self.frame.pack()
        self.buttons = [tk.Button(self.frame, text=f"Drop\n{i+1}", command=lambda c=i: self.human_move(c))
                        for i in range(7)]
        for btn in self.buttons:
            btn.pack(side=tk.LEFT)
        
        self.canvas = tk.Canvas(root, width=420, height=360)
        self.canvas.pack()
        self.restart_button = tk.Button(root, text="Restart", command=self.reset_game)
        self.restart_button.pack()
        
        self.draw_board()
        self.player_turn = True  # Human goes first by default
        
        self.choose_turn()
    
    def preprocess_state(self):
        return np.array(self.env.grid).flatten()
    
    def draw_board(self):
        self.canvas.delete("all")
        for col in range(7):
            for row in range(6):
                x0, y0 = col * 60, (5 - row) * 60
                x1, y1 = x0 + 60, y0 + 60
                color = "white"
                if self.env.grid[col][row] == ConnectFour.RED:
                    color = "red"
                elif self.env.grid[col][row] == ConnectFour.YELLOW:
                    color = "yellow"
                self.canvas.create_oval(x0, y0, x1, y1, fill=color)
    
    def human_move(self, column):
        if not self.player_turn or column not in self.env.get_actions():
            return
        
        self.env.move(column)
        self.draw_board()
        self.check_game_status(ConnectFour.RED)
        self.player_turn = False
        self.root.after(500, self.ai_move)
    
    def ai_move(self):
        if self.env.check_win(ConnectFour.RED) or self.env.check_draw():
            return
        
        state_tensor = torch.tensor(self.preprocess_state(), dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state_tensor).detach().numpy().flatten()
        valid_actions = self.env.get_actions()
        q_values = [(q_values[i] if i in valid_actions else -float('inf')) for i in range(7)]
        action = np.argmax(q_values)
        
        self.env.move(action)
        self.draw_board()
        self.check_game_status(ConnectFour.YELLOW)
        self.player_turn = True
    
    def check_game_status(self, turn):
        if self.env.check_win(turn):
            winner = "You" if turn == ConnectFour.RED else "AI"
            messagebox.showinfo("Game Over", f"{winner} won!")
            self.reset_game()
        elif self.env.check_draw():
            messagebox.showinfo("Game Over", "It's a draw!")
            self.reset_game()
    
    def reset_game(self):
        self.env = ConnectFour()
        self.player_turn = True
        self.draw_board()
    
    def choose_turn(self):
        response = messagebox.askyesno("Choose Turn", "Do you want to go first?")
        self.player_turn = response
        if not self.player_turn:
            self.root.after(500, self.ai_move)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Connect Four")
    game = ConnectFourGame(root)
    root.mainloop()
