import numpy as np
from game_utils import *
from agents.random_agent.random import generate_move_random

def test_generate_move_random_valid_column():
    """Test that the random move generator selects a valid column."""
    board = np.full((6, 7), NO_PLAYER)  # empty board
    player = PLAYER1
    action, _ = generate_move_random(board, player, None)

    assert board[-1, action] == NO_PLAYER