from game_utils import *
import numpy as np


def score(board: np.ndarray, player: BoardPiece) -> int:
    """A heuristic function to evaluate the board state for the minimax algorithm using a weight matrix.

    Parameters:
    ----------
        board (np.ndarray): The current game board.
        player (BoardPiece): The player for whom the score is being evaluated.
        
    Returns
    -------
    int
        A score representing the favorability of the board state for the player.
    """

    Weight_matrix = np.array([ 
        [3, 4, 5,   7,  5, 4, 3],
        [4, 6, 8,  10,  8, 6, 4],
        [5, 7, 11, 13, 11, 7, 5],
        [5, 7, 11, 13, 11, 7, 5],
        [4, 6, 8, 10, 8,   6, 4],
        [3, 4, 5, 7, 5,    4, 3]
         ])
    score = 0
    # Iterate through the board to calculate the score
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row, col] == player:
                # Increase score based on the position of the player's piece
                score+= Weight_matrix[row, col]
            elif board[row, col] != NO_PLAYER:
                # Decrease score for opponent's pieces
                score -= Weight_matrix[row, col]
    return score





def alphabeta(board: np.ndarray, depth: int, alpha: float, beta: float,
              current_player: BoardPiece, root_player: BoardPiece, maximizing: bool, saved_state: SavedState | None) -> tuple[PlayerAction, float, SavedState | None]:
    
    """ Recursive Function implementing alphabeta pruning for the minimax algorithm 
        Finding the best move for the current player according to teh heuristic score evaluation.

    Parameters:
    ----------
        board (np.ndarray): The current game board.
        depth (int): The current depth in the game tree.
        alpha (float): The best score that the maximizer currently can guarantee at that level or above.
        beta (float): The best score that the minimizer currently can guarantee at that level or above.
        current_player (int): The player making the current move.
        root_player (int): The player for whom the score is being evaluated.
        maximizing (bool): True if the current player is the maximizer, False if the minimizer.
        saved_state (SavedState | None): Optional saved state (not needed for this implementation).
        
    Returns
    -------
        tuple[PlayerAction, float, SavedState | None]
            The best move, score of the board state, and saved state (None in this case)
    """
    

    
    # Base case: leaf node or game over
    if depth == 0 or check_end_state(board, root_player) != GameState.STILL_PLAYING:
        
        return None, score(board, root_player), None


    valid_moves = [col for col in range(board.shape[1]) if board[-1, col] == NO_PLAYER]

    best_move = None

    # The following algorithm updates the alpha and beta values based on the simulated current player's turn.

    if maximizing:
        max_score = -np.inf
        for move in valid_moves:
            new_board = board.copy()
            apply_player_action(new_board, PlayerAction(move), current_player)

            # unpack all 3 return values of the alphabeta function
            _, score_eval, _ = alphabeta(
                new_board,
                depth - 1,
                alpha,
                beta,
                PLAYER2 if current_player == PLAYER1 else PLAYER1,
                root_player,
                False,  # Next player is minimizing
                None    
            )

            if score_eval > max_score:
                max_score = score_eval
                best_move = PlayerAction(move)

            alpha = max(alpha, score_eval)
            if beta <= alpha:
                break  # Beta cut-off (pruning)
                
        return best_move, max_score, None # Note: returning best move is only relevant for the root player

    else:  # minimizing
        min_score = np.inf
        for move in valid_moves:
            new_board = board.copy()
            apply_player_action(new_board, PlayerAction(move), current_player)

            # unpack all 3 return values of the alphabeta function
            _, score_eval, _ = alphabeta(
                new_board,
                depth - 1,
                alpha,
                beta,
                PLAYER2 if current_player == PLAYER1 else PLAYER1,
                root_player,
                True,   # Next player is maximizing
                None    
            )

            if score_eval < min_score:
                min_score = score_eval
                best_move = PlayerAction(move)

            beta = min(beta, score_eval)
            if beta <= alpha:
                break  # Alpha cut-off (pruning)
                
        return best_move, min_score, None # Note: returning best move is only relevant for the root player
    
def generate_move_minimax(board: np.ndarray, current_player: BoardPiece, saved_state: SavedState | None, depth=7) -> tuple[PlayerAction, SavedState | None]:
    """Get the best move for the current player using alpha-beta pruning
    Parameters:
    ----------
        board (np.ndarray): The current game board.
        current_player (BoardPiece): The player making the move.
        saved_state (SavedState | None): Any saved state from previous moves.   
        Returns
    -------
        tuple[PlayerAction, SavedState | None]
            The best move for the current player and the saved state (None in this case).
    """
    if block_opponent(board, current_player) is None:
        best_move, best_score, _ = alphabeta(
            board=board,
            depth=depth,              
            alpha=-np.inf,        # Initialize alpha to negative infinity
            beta=np.inf,          # Initialize beta to positive infinity
            current_player=current_player,
            root_player=current_player,  # We're evaluating for the current player
            maximizing=True,      # Current player is maximizing at root
            saved_state=None
        )
    else:
        best_move = block_opponent(board, current_player)
    
    return best_move, saved_state


def block_opponent(board: np.ndarray, current_player: BoardPiece) -> PlayerAction:
    """A simple function to block the opponent's winning move if possible.
    
    Parameters:
    ----------
        board (np.ndarray): The current game board.
        current_player (BoardPiece): The player making the move.
        
    Returns
    -------
    PlayerAction
        The column to block the opponent's winning move, or None if no blocking is needed.
    """
    
    opponent = PLAYER2 if current_player == PLAYER1 else PLAYER1
    
    for col in range(BOARD_COLS):
        if board[-1, col] == NO_PLAYER:
            # Simulate the opponent's move
            new_board = board.copy()
            apply_player_action(new_board, PlayerAction(col), opponent)
            if check_end_state(new_board, opponent) == GameState.IS_WIN:
                return PlayerAction(col)
    
    return None  # No blocking move found

def block_opponent(board: np.ndarray, current_player: BoardPiece) -> PlayerAction | None:
    """
    Attempts to preempt a loss by blocking an opponent's winning move.

    Parameters
    ----------
    board : np.ndarray
        The current Connect Four board.
    current_Player : BoardPiece
         the player whose turn it is.

    Returns
    -------
    PlayerAction or None
        The column index to block the threat, or None if no danger is detected.
    """
    opponent = PLAYER1 if current_player == PLAYER2 else PLAYER2

    for column in range(BOARD_COLS):
        # Skip columns that are already full
        if board[0, column] != NO_PLAYER:
            continue

        # Test placing opponent piece
        board_sim = board.copy()
        apply_player_action(board_sim, PlayerAction(column), opponent)

        # Evaluate board for opponent win
        if check_end_state(board_sim, opponent) == GameState.IS_WIN:
            return PlayerAction(column)

    return None
