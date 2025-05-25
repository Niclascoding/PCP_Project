import numpy as np
import pytest
from game_utils import *
from agents.minimax_agent.minimax import score, generate_move_minimax, alphabeta


def test_score_all_opponent_pieces():
    """Test the score function with a board full of opponent pieces."""
    board = np.full((BOARD_ROWS, BOARD_COLS), PLAYER2, dtype=np.int8)
    expected_score = -np.sum(np.array([
        [3, 4, 5,   7,  5, 4, 3],
        [4, 6, 8,  10,  8, 6, 4],
        [5, 7, 11, 13, 11, 7, 5],
        [5, 7, 11, 13, 11, 7, 5],
        [4, 6, 8, 10, 8,   6, 4],
        [3, 4, 5, 7, 5,    4, 3]
    ]))
    
    result = score(board, PLAYER1)
    assert result == expected_score

def test_score_empty_board():
    """Test the score function on an empty board."""
    board = np.full((BOARD_ROWS, BOARD_COLS), NO_PLAYER, dtype=np.int8)
    result = score(board, PLAYER1)
    assert result == 0

def test_score_all_player_pieces():
    """Test the score function with a board full of the player's pieces."""
    board = np.full((BOARD_ROWS, BOARD_COLS), PLAYER1, dtype=np.int8)
    expected_score = np.sum(np.array([
        [3, 4, 5,   7,  5, 4, 3],
        [4, 6, 8,  10,  8, 6, 4],
        [5, 7, 11, 13, 11, 7, 5],
        [5, 7, 11, 13, 11, 7, 5],
        [4, 6, 8, 10, 8,   6, 4],
        [3, 4, 5, 7, 5,    4, 3]
    ]))

    result = score(board, PLAYER1)
    assert result == expected_score

def test_alphabeta_edge_case_zerodepth():
    """Test alphabeta with depth 0."""
    board = string_to_board("""
    |==============|
    |X     X O   O |
    |X O O X O X X |
    |O X X X O O O |
    |X O O O X X O |
    |X O X O O X X |
    |X O O X X O O |
    |==============|
    |0 1 2 3 4 5 6 |""")
    # Simulate a depth=0 or full board so expect no move and a numeric score
    move, score_val, state = alphabeta(
        board=board,
        depth=0,
        alpha=-np.inf,
        beta=np.inf,
        current_player=PLAYER1,
        root_player=PLAYER1,
        maximizing=True,
        saved_state=None,
    )

    assert move is None and np.isscalar(score_val)

def test_alphabeta_edge_case_terminal_state():
    """Test alphabeta on a full board."""
    board = string_to_board("""
    |==============|
    |X O X X O X O |
    |X O O X O X X |
    |O X X X O O O |
    |X O O O X X O |
    |X O X O O X X |
    |X O O X X O O |
    |==============|
    |0 1 2 3 4 5 6 |""")
    
    # Simulate full board so expect no move and a numeric score
    move, score_val, state = alphabeta(
        board=board,
        depth=4,
        alpha=-np.inf,
        beta=np.inf,
        current_player=PLAYER1,
        root_player=PLAYER1,
        maximizing=True,
        saved_state=None,
    )

    assert move is None and np.isscalar(score_val)


def test_generate_move_minimax_empty_board():
    """Ensure the function returns a valid move on an empty board."""
    board = np.full((BOARD_ROWS, BOARD_COLS), NO_PLAYER, dtype=np.int8)
    action, _ = generate_move_minimax(board, PLAYER1, None)
    assert 0 <= action < BOARD_COLS 

def test_generate_move_minimax_blocks_opponent():
    """Test if block_opponent triggers and is used instead of alphabeta."""

    board = string_to_board("""
    |==============|
    |              |
    |              |
    |              |
    |              |
    |X X X         |
    |O O O         |
    |==============|
    |0 1 2 3 4 5 6 |""")
    
    action, _ = generate_move_minimax(board, PLAYER1, None)
    assert action == 3  

def test_generate_move_minimax_one_column_left():
    board = np.full((BOARD_ROWS, BOARD_COLS), PLAYER1, dtype=np.int8)
    board[:, 3] = NO_PLAYER  # Only column 3 is empty
    action, _ = generate_move_minimax(board, PLAYER2, None)
    assert action == 3

    