import pytest
import numpy as np
from game_utils import *
"""from game_utils import (
    BOARD_COLS,
    BOARD_ROWS,
    initialize_game_state,
    BOARD_SHAPE,
    BoardPiece,
    NO_PLAYER,
    string_to_board,
    pretty_print_board,
    PLAYER1,
    PLAYER2,
    NO_PLAYER,
    PLAYER1_PRINT,
    PLAYER2_PRINT,
    NO_PLAYER_PRINT,
    apply_player_action,
    PlayerAction)"""

def test_initialize_game_state_returns_right_shape_board():
    board = initialize_game_state()
    assert isinstance(board, np.ndarray) and board.shape == BOARD_SHAPE

def test_initialize_game_state_returns_correct_dtype_board():
    board = initialize_game_state()
    assert board.dtype == BoardPiece

def test_initialize_game_state_returns_empty_board():
    # Every entry should be NO_PLAYER (i.e. zero)
    board = initialize_game_state()
    assert np.all(board == NO_PLAYER)




def test_pretty_print_board_returns_string():
    """Test the pretty_print_board function to ensure it returns a string."""
    board = initialize_game_state()
    output = pretty_print_board(board)
    assert isinstance(output, str)

def test_pretty_print_board_correct_representation():
    """Test the pretty_print_board function to ensure it returns the correct string representation."""
    # create a board with a few pieces
    board = initialize_game_state()
    board[0, 0] = PLAYER1  # bottom-left
    board[0, 6] = PLAYER2  # bottom-right
    board[1, 0] = PLAYER1
    board[0, 2] = PLAYER2

    output = pretty_print_board(board)

    # check that the correct symbols are present
    assert PLAYER1_PRINT in output and PLAYER2_PRINT in output

def test_pretty_print_board_check_that_the_board_grid_contains_the_column_headers():
    board = initialize_game_state()
    output = pretty_print_board(board)
    for col in range(7):
        assert str(col) in output

def test_pretty_print_board_correct_order():
    """The board is printed from bottom to top, so the first row of the
    board should be the last row in the printed string representation. """
    # create a board with a few pieces
    board = initialize_game_state()
    board[0, 0] = PLAYER1  # bottom-left
    board[0, 6] = PLAYER2  # bottom-right
    board[1, 0] = PLAYER1
    board[0, 2] = PLAYER2

    output = pretty_print_board(board)

    # check that the pieces are in the correct order (bottom to top)
    output = output.split('\n')
    assert (output[6][1] == PLAYER1_PRINT and
     output[6][5] == PLAYER2_PRINT and
     output[5][1] == PLAYER1_PRINT and
     output[5][5] == NO_PLAYER_PRINT and
     output[4][1] == NO_PLAYER_PRINT and
     output[6][13] == PLAYER2_PRINT)

def test_string_to_board():
    """Test the string_to_board function to ensure it converts a string back to a board."""
    # create a board with a few pieces
    board = initialize_game_state()
    board[0, 0] = PLAYER1  # bottom-left
    board[0, 6] = PLAYER2  # bottom-right
    board[1, 0] = PLAYER1
    board[0, 2] = PLAYER2

    output = pretty_print_board(board)
    new_board = string_to_board(output)

    # check that the new board matches the original
    assert np.array_equal(board, new_board)

def test_apply_player_action():
    """Test the apply_player_action function to ensure it applies the action correctly."""
    # create a board with a few pieces
    board = initialize_game_state()
    board[0, 0] = PLAYER1  # bottom-left
    board[0, 6] = PLAYER2  # bottom-right
    board[1, 0] = PLAYER1
    board[0, 2] = PLAYER2

    # apply an action after doing it manually
    
    new_board = string_to_board(pretty_print_board(board))
    new_board[1, 2] = PLAYER1
    apply_player_action(board,PlayerAction(2),PLAYER1)

    # check that the new board matches the original
    assert np.array_equal(board, new_board)

    #waht happens if full
    with pytest.raises(ValueError, match="No open row found in the specified column."):
        for i in range(5):
            apply_player_action(board,PlayerAction(2),PLAYER1)


def test_check_end_state():
    """Test the GameState function to ensure it initializes the game state correctly."""
    # create a full board 
    boardfull= string_to_board("""
    |==============|
    |X O X X O X O |
    |X O O X O X X |
    |O X X X O O O |
    |X O O O X X O |
    |X O X O O X X |
    |X O O X X O O |
    |==============|
    |0 1 2 3 4 5 6 |""")
    assert check_end_state(boardfull,PLAYER1)==GameState.IS_DRAW

def test_check_end_state_winner():
    # create a board with a winner
    boardwin= string_to_board("""
    |==============|
    |              |
    |    O X O X X |
    |X X X X O O O |
    |X O O O X X O |
    |X O X O O X X |
    |X O O X X O O |
    |==============|
    |0 1 2 3 4 5 6 |""")
    assert check_end_state(boardwin, PLAYER1)==GameState.IS_WIN

def test_check_end_state_playing():
    # create a board with no winner and still playoing
    boardstill= string_to_board("""
    |==============|
    |              |
    |X   O X O X X |
    |O X X X O O O |
    |X O O O X X O |
    |X O X O O X X |
    |X O O X X O O |
    |==============|
    |0 1 2 3 4 5 6 |""")
    assert check_end_state(boardstill, PLAYER1)==GameState.STILL_PLAYING

def test_check_move_status():
    """Test the check_move_status function to ensure it returns the correct status."""
    # create a board with a few pieces
    board = initialize_game_state()
    board[0, 0] = PLAYER1  # bottom-left
    board[0, 6] = PLAYER2  # bottom-right
    board[1, 0] = PLAYER1
    board[0, 2] = PLAYER2

    # check that the move is valid
    assert check_move_status(board, PlayerAction(2)) == MoveStatus.IS_VALID

    # check that the move is invalid (column full)
    for i in range(5):
        apply_player_action(board, PlayerAction(2), PLAYER1)
    assert check_move_status(board, PlayerAction(2)) == MoveStatus.FULL_COLUMN
    # check that the move is invalid (wrongtype)
    assert check_move_status(board, 4) == MoveStatus.WRONG_TYPE
    # check that the move is invalid (out of bounds)
    assert check_move_status(board, PlayerAction(7)) == MoveStatus.OUT_OF_BOUNDS
