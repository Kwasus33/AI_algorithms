import numpy as np

from game import N_ROWS


class GameTUI:
    def __init__(self, game, player_x, player_o):
        self.game = game
        self.player_x = player_x
        self.player_o = player_o

    def mainloop(self):
        end_game = False
        while not end_game:
            self.board = self.game.board
            self.reset_board = False
            self.print_board()
            while not self.reset_board:
                self.gameplay()
            if (
                str(input("Press p if u want to play again, otherwise any other key: "))
                == "p"
            ):
                self.game.play_again()
            else:
                end_game = True

    def gameplay(self):
        self.move_player()
        self.print_board()
        self.check_winner()

    def move_player(self):
        x, y = 0, 0
        if (self.player_x.isHuman() and self.game.player_x_turn) or (
            self.player_o.isHuman() and not self.game.player_x_turn
        ):
            while x not in range(1, N_ROWS + 1) or y not in range(1, N_ROWS + 1):
                x = int(input("Podaj x: "))
                y = int(input("Podaj y: "))

        position = np.array([x - 1, y - 1])
        position = (
            self.player_x.get_move(position)
            if self.game.player_x_turn
            else self.player_o.get_move(position)
        )

        self.game.move(position)
        print("Player x moved") if self.game.player_x_turn else print("Player o moved")

    def check_winner(self):
        if self.game.get_winner() in ["x", "o", "t"]:
            (
                print(f"Winner is {self.game.get_winner()}\n")
                if self.game.get_winner() in ["x", "o"]
                else print("It's tie\n")
            )
            self.reset_board = True

    def print_board(self):
        for row in self.board:
            print("_" + "_|_".join(row) + "_" + "\n")
