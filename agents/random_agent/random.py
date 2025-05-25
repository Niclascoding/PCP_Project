
import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER



def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    """A simple function that selects a random valid column adn returns it as action.

    Parameters:
    ----------
        board (np.ndarray): The current game board.
        player (BoardPiece): The player making the move.
        saved_state (SavedState | None): Any saved state from previous moves.
        
    Returns
    -------
    Playeraction, SavedState
        The action selected (and the saved state, which is None in this case).
    """
    valid_actions = [col for col in range(board.shape[1]) if board[-1, col] == NO_PLAYER]
    action = PlayerAction(np.random.choice(valid_actions))

    
    return action, saved_state