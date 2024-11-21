from abc import ABC, abstractmethod

import numpy as np
import math, copy

HEURISTIC = [[3, 2, 3], [2, 4, 2], [3, 2, 3]]

def build_player(player_config, game):
    assert player_config["type"] in ["human", "random", "minimax"]

    if player_config["type"] == "human":
        return HumanPlayer(game)

    if player_config["type"] == "random":
        return RandomComputerPlayer(game)

    if player_config["type"] == "minimax":
        return MinimaxComputerPlayer(game, player_config)


class Player(ABC):
    def __init__(self, game):
        self.game = game
        self.score = 0

    @abstractmethod
    def get_move(self, event_position):
        pass


class HumanPlayer(Player):
    def get_move(self, event_position):
        return event_position


class RandomComputerPlayer(Player):
    def get_move(self, event_position):
        available_moves = self.game.available_moves()
        move_id = np.random.choice(len(available_moves))
        return available_moves[move_id]


class MinimaxComputerPlayer(Player):
    def __init__(self, game, config):
        super().__init__(game)
        self.depth = int(config["depth"]) if config["depth"] else 0

    def get_move(self, event_position):
        best_move = None
        best_eval = -math.inf
        for move in self.game.available_moves():
            moves = copy.deepcopy(self.game.available_moves())
            depth = copy.copy(self.depth)
            alpha = -math.inf
            beta = math.inf
            moves = moves[moves != move]
            moveEval = self._minimax(move, moves, depth, False, alpha, beta)     
            best_move = max(moveEval, best_move)
            if moveEval > best_eval:
                best_move = move
                best_eval = moveEval
            
        return best_move


    def _minimax(self, moved, moves, depth, isMax, alpha, beta):
        self.moves = moves

        if depth == 0 or len(moves) == 0:
            x, y = moved
            return HEURISTIC[x][y]
        
        if isMax:
            MaxEval = -math.inf
            for move in self.moves:
                self.moves = self.moves[self.moves != move]
                eval = self._minimax(move, moves, depth-1, not isMax, alpha, beta)
                MaxEval = max(MaxEval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    return MaxEval
            return MaxEval
        
        else:
            MinEval = math.inf
            for move in moves:
                self.moves = self.moves[self.moves != move]
                eval = self._minimax(move, moves, depth-1, isMax, alpha, beta)
                MinEval = min(MinEval, eval)
                beta = min(MinEval, eval)
                if beta <= alpha:
                    return MinEval
            return MinEval
