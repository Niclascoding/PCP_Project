
import numpy as np
from game_utils import BoardPiece, PlayerAction, SavedState, NO_PLAYER



def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: SavedState | None
) -> tuple[PlayerAction, SavedState | None]:
    """A simple agent that selects a random valid column."""
    valid_actions = [col for col in range(board.shape[1]) if board[-1, col] == NO_PLAYER]
    action = PlayerAction(np.random.choice(valid_actions))

    # Choose a valid, non-full column randomly and return it as `action`
    return action, saved_state