"""
GomokuBoard: Enhanced Gomoku board module
Contains board state, rule enforcement, and efficient pattern evaluation
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Any
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gomoku_board.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GomokuBoard")

class GomokuBoard:
    """
    Represents a Gomoku game board with enhanced pattern evaluation and efficient state management
    """
    
    # Define patterns and their corresponding threat levels
    PATTERNS = {
        # Pattern name: (pattern, open ends, score)
        'five': ('11111', 0, 100000),  # Five in a row
        'open_four': ('011110', 2, 10000),  # Open four
        'four': ('011112', 1, 1000),  # Blocked four
        'four_b': ('211110', 1, 1000),  # Blocked four variant
        'four_c': ('11101', 1, 1000),  # Jump blocked four
        'four_d': ('10111', 1, 1000),  # Jump blocked four variant
        'open_three': ('01110', 2, 1000),  # Open three
        'three': ('211100', 1, 100),  # Blocked three
        'three_b': ('001112', 1, 100),  # Blocked three variant
        'three_c': ('11100', 1, 100),  # Blocked three variant
        'three_d': ('00111', 1, 100),  # Blocked three variant
        'open_two': ('00110', 2, 10),  # Open two
        'open_two_b': ('01100', 2, 10),  # Open two variant
        'two': ('11000', 1, 6),  # Blocked two
        'two_b': ('00011', 1, 6),  # Blocked two variant
    }
    
    # Score normalization factor
    SCORE_FACTOR = 1.0
    
    def __init__(self, size: int = 15):
        """
        Initialize the board
        
        Args:
            size: Board size, default is 15x15
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)  # 0: empty, 1: black, 2: white
        self.current_player = 1  # Black goes first
        self.move_history = []
        self.last_move = None
        self.game_over = False
        self.winner = None
        
        # Pattern evaluation caches and incremental calculation
        self._pattern_cache = {}  # Cache patterns at each position in each direction
        self._move_scores = {}    # Cache scores for each possible move
        self._line_strings = {}   # Cache line strings in each direction
        self._threat_positions = set()  # Positions that need special attention as threats
        
        # Precompute board position importance (Manhattan distance from center)
        self._position_importance = self._calculate_position_importance()
        
        logger.info(f"Initialized {size}x{size} GomokuBoard")
    
    def _calculate_position_importance(self) -> np.ndarray:
        """Calculate board position importance (based on distance from center)"""
        importance = np.zeros((self.size, self.size), dtype=np.float32)
        center = self.size // 2
        
        for r in range(self.size):
            for c in range(self.size):
                # Calculate Manhattan distance to center, and convert to importance score (closer = higher)
                dist = abs(r - center) + abs(c - center)
                max_dist = 2 * center
                importance[r, c] = 1.0 - (dist / max_dist) * 0.8  # Keep 0.2 as base value
        
        return importance
    
    def reset(self):
        """Reset the board to initial state"""
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1
        self.move_history = []
        self.last_move = None
        self.game_over = False
        self.winner = None
        
        # Reset caches
        self._pattern_cache = {}
        self._move_scores = {}
        self._line_strings = {}
        self._threat_positions = set()
        
        logger.info("Board reset to initial state")
    
    def make_move(self, row: int, col: int) -> bool:
        """
        Place a stone at the specified position
        
        Args:
            row: Row index (0-based)
            col: Column index (0-based)
            
        Returns:
            bool: True if move is valid and successful, False otherwise
        """
        # Check if game is already over
        if self.game_over:
            logger.warning("Attempted move on finished game")
            return False
            
        # Check if position is within bounds
        if not (0 <= row < self.size and 0 <= col < self.size):
            logger.warning(f"Move ({row}, {col}) out of bounds")
            return False
            
        # Check if position is empty
        if self.board[row, col] != 0:
            logger.warning(f"Position ({row}, {col}) already occupied")
            return False
        
        # Place stone
        self.board[row, col] = self.current_player
        pos = row * self.size + col  # Convert to 1D position
        self.move_history.append(pos)
        self.last_move = (row, col)
        
        # Invalidate caches affected by this move
        self._invalidate_caches(row, col)
        
        # Check for win
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
            logger.info(f"Player {self.current_player} wins at move {len(self.move_history)}")
        # Check for draw
        elif len(self.move_history) == self.size * self.size:
            self.game_over = True
            self.winner = 0  # Draw
            logger.info("Game ended in a draw")
        
        # Switch player
        self.current_player = 3 - self.current_player  # Switch between 1 and 2
        return True
    
    def make_move_1d(self, pos: int) -> bool:
        """
        Make a move using 1D position (0 to size²-1)
        
        Args:
            pos: 1D position (row * size + column)
            
        Returns:
            bool: True if move is valid and successful
        """
        row, col = pos // self.size, pos % self.size
        return self.make_move(row, col)
    
    def _invalidate_caches(self, row: int, col: int):
        """Invalidate caches affected by a specific move"""
        # Clear affected pattern caches
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            # Check affected range (typically 4 positions)
            for offset in range(-4, 5):
                r, c = row + dr * offset, col + dc * offset
                if 0 <= r < self.size and 0 <= c < self.size:
                    key = (r, c, dr, dc)
                    if key in self._pattern_cache:
                        del self._pattern_cache[key]
        
        # Clear move score cache completely - it's safer than trying to selectively invalidate
        self._move_scores = {}
        
        # Clear affected line string caches
        for dr, dc in directions:
            # Calculate row start point
            start_r, start_c = row, col
            while 0 <= start_r - dr < self.size and 0 <= start_c - dc < self.size:
                start_r -= dr
                start_c -= dc
            key = (start_r, start_c, dr, dc)
            if key in self._line_strings:
                del self._line_strings[key]
    
    def get_legal_moves(self) -> List[int]:
        """
        Get all legal moves (1D format)
        
        Returns:
            List[int]: List of legal positions, in 1D index form
        """
        if self.game_over:
            return []
            
        legal_moves = []
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row, col] == 0:
                    pos = row * self.size + col
                    legal_moves.append(pos)
        
        return legal_moves
    
    def get_focused_moves(self, distance: int = 2, consider_threats: bool = True) -> List[int]:
        """
        Get legal moves that are within a certain distance of existing stones
        
        Args:
            distance: Maximum distance to consider
            consider_threats: Whether to especially consider threat positions
            
        Returns:
            List[int]: List of legal moves, focused on relevant positions
        """
        # If board is empty, return center point and surrounding positions
        if not self.move_history:
            mid = self.size // 2
            result = [(mid * self.size + mid)]  # Center point
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)]:
                r, c = mid + dr, mid + dc
                if 0 <= r < self.size and 0 <= c < self.size:
                    result.append(r * self.size + c)
            return result
        
        if self.game_over:
            return []
        
        # Find all positions to consider
        candidates = set()
        
        # Get all positions with stones
        occupied = set()
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    occupied.add((r, c))
        
        # Find empty positions within specified distance of any stone
        for r, c in occupied:
            for dr in range(-distance, distance + 1):
                for dc in range(-distance, distance + 1):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.size and 0 <= nc < self.size and 
                        self.board[nr, nc] == 0 and 
                        (dr != 0 or dc != 0)):  # Exclude the stone itself
                        candidates.add((nr, nc))
        
        # If considering threat positions, add them to candidates
        if consider_threats and self._threat_positions:
            for r, c in self._threat_positions:
                if 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 0:
                    candidates.add((r, c))
        
        # Convert to 1D positions
        return [r * self.size + c for r, c in candidates]
    
    def _check_win(self, row: int, col: int) -> bool:
        """
        Check if the last move created a winning line
        
        Args:
            row: Row of last move
            col: Column of last move
            
        Returns:
            bool: True if this move created a winning line
        """
        player = self.board[row, col]
        
        # Define four directions: horizontal, vertical, diagonal, anti-diagonal
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1  # Count consecutive stones (including current stone)
            
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            
            # If 5 or more consecutive stones, win
            if count >= 5:
                return True
                
        return False
    
    def evaluate_position(self, player_perspective: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate current board position, calculating pattern counts and total score for both players
        
        Args:
            player_perspective: Which player's perspective to evaluate from (1=black, 2=white, None=current player)
            
        Returns:
            Dict: Evaluation result containing pattern counts and total score
        """
        if player_perspective is None:
            player_perspective = self.current_player
        
        opponent = 3 - player_perspective
        
        # Initialize result
        result = {
            player_perspective: {"patterns": {}, "score": 0, "threats": []},
            opponent: {"patterns": {}, "score": 0, "threats": []}
        }
        
        # Initialize pattern counts
        for player in [player_perspective, opponent]:
            for pattern_name in self.PATTERNS.keys():
                result[player]["patterns"][pattern_name] = 0
        
        # Scan all rows, columns, and diagonals on the board
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # Track detected threat positions
        threats = set()
        
        # Scan rows
        for start_row in range(self.size):
            for start_col in range(self.size):
                for dr, dc in directions:
                    # Ensure this is a start point of a new line
                    if (start_row > 0 and dr != 0) or (start_col > 0 and dc != 0):
                        continue
                        
                    line_key = (start_row, start_col, dr, dc)
                    if line_key in self._line_strings:
                        line_str = self._line_strings[line_key]
                    else:
                        # Extract line
                        line = []
                        r, c = start_row, start_col
                        while 0 <= r < self.size and 0 <= c < self.size:
                            line.append(self.board[r, c])
                            r += dr
                            c += dc
                        
                        # Skip if line is too short
                        if len(line) < 5:
                            continue
                            
                        # Convert to string for pattern matching
                        line_str = ''.join(map(str, line))
                        self._line_strings[line_key] = line_str
                    
                    # Match patterns for both players
                    for player in [player_perspective, opponent]:
                        # Replace opponent stones with '2', own stones with '1', empty with '0'
                        if player == player_perspective:
                            player_str = str(player)
                            opponent_str = str(opponent)
                        else:
                            player_str = str(opponent)
                            opponent_str = str(player_perspective)
                            
                        pattern_str = line_str.replace(player_str, '1').replace(opponent_str, '2').replace('0', '0')
                        
                        # Check for matching patterns
                        for pattern_name, (pattern, _, score) in self.PATTERNS.items():
                            # Calculate number of matching patterns for this player
                            matches = self._find_pattern_matches(pattern_str, pattern)
                            if matches > 0:
                                result[player]["patterns"][pattern_name] += matches
                                result[player]["score"] += matches * score
                                
                                # If threat pattern, mark threat positions
                                if pattern_name in ['five', 'open_four', 'four', 'four_b', 'four_c', 'four_d']:
                                    # Find threat positions (where next move could win)
                                    threat_pos = self._find_threat_positions(
                                        pattern_str, pattern, start_row, start_col, dr, dc
                                    )
                                    for thr in threat_pos:
                                        threats.add(thr)
                                        if player == self.current_player:  # If current player's threat
                                            result[player]["threats"].append(thr)
        
        # Update threat positions set
        self._threat_positions = threats
        
        # Final score calculation
        # Consider not just pattern scores but also position importance
        for player in [player_perspective, opponent]:
            # Add position importance score
            position_score = 0
            for i in range(self.size):
                for j in range(self.size):
                    if self.board[i, j] == player:
                        position_score += self._position_importance[i, j] * 10  # Weight by position importance
            
            result[player]["score"] += position_score
            
            # Normalize score
            result[player]["score"] *= self.SCORE_FACTOR
        
        return result
    
    def _find_pattern_matches(self, line_str: str, pattern: str) -> int:
        """Count occurrences of a pattern in a line"""
        count = 0
        for i in range(len(line_str) - len(pattern) + 1):
            if line_str[i:i+len(pattern)] == pattern:
                count += 1
        return count
    
    def _find_threat_positions(self, line_str: str, pattern: str, 
                              start_row: int, start_col: int, 
                              dr: int, dc: int) -> List[Tuple[int, int]]:
        """Find threat positions (positions that need defense or could be attacked)"""
        threat_positions = []
        
        for i in range(len(line_str) - len(pattern) + 1):
            if line_str[i:i+len(pattern)] == pattern:
                # For threats, we look for empty positions that could form five
                # For example, in "011110", positions 0 and 5 are threat positions
                for j in range(len(pattern)):
                    if pattern[j] == '0':  # Empty position
                        r = start_row + (i + j) * dr
                        c = start_col + (i + j) * dc
                        if 0 <= r < self.size and 0 <= c < self.size:
                            threat_positions.append((r, c))
        
        return threat_positions
    
    def get_score_for_move(self, row: int, col: int, player: int) -> float:
        """
        Calculate move score for a specific position
        Uses incremental evaluation rather than full recalculation
        
        Args:
            row, col: Move position
            player: Player ID (1 or 2)
            
        Returns:
            float: Move score for this position
        """
        # Check if position is legal
        if not (0 <= row < self.size and 0 <= col < self.size) or self.board[row, col] != 0:
            return float('-inf')
            
        # Check cache
        cache_key = (row, col, player)
        if cache_key in self._move_scores:
            return self._move_scores[cache_key]
        
        # Create temporary board state
        original_value = self.board[row, col]
        self.board[row, col] = player
        
        # Evaluate all 4 directions affected by this move
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        total_score = 0
        
        try:
            for dr, dc in directions:
                # Scan 11 positions in this direction (maximum affected range)
                line = []
                for offset in range(-5, 6):
                    r, c = row + dr * offset, col + dc * offset
                    if 0 <= r < self.size and 0 <= c < self.size:
                        line.append(str(self.board[r, c]))
                    else:
                        line.append('X')  # Out of bounds
                
                line_str = ''.join(line)
                
                # Replace opponent stones with '2', own stones with '1', empty with '0'
                opponent = 3 - player
                pattern_str = line_str.replace(str(player), '1').replace(str(opponent), '2') \
                                      .replace('0', '0').replace('X', 'X')
                
                # Check for all possible patterns
                for pattern_name, (pattern, openness, score) in self.PATTERNS.items():
                    matches = self._find_pattern_matches(pattern_str, pattern)
                    if matches > 0:
                        # Adjust score based on openness and player
                        adjusted_score = score
                        if player == self.current_player:
                            adjusted_score *= 1.1  # Slight preference for offense
                        total_score += matches * adjusted_score
            
            # Also consider position value
            total_score += self._position_importance[row, col] * 5
            
        except Exception as e:
            logger.error(f"Error evaluating move ({row}, {col}): {e}")
            total_score = 0
        
        # Restore original board state
        self.board[row, col] = original_value
        
        # Cache result
        self._move_scores[cache_key] = total_score
        return total_score
    
    def detect_immediate_threat(self) -> Optional[int]:
        """
        Detect immediate threats, returning defensive or winning position
        
        Returns:
            Optional[int]: 1D threat position, or None if no immediate threat
        """
        try:
            # First check if we can win in one move
            for r in range(self.size):
                for c in range(self.size):
                    if self.board[r, c] == 0:
                        # Temporarily place stone
                        self.board[r, c] = self.current_player
                        if self._check_win(r, c):
                            # Found winning position
                            self.board[r, c] = 0  # Restore
                            return r * self.size + c
                        self.board[r, c] = 0  # Restore
            
            # Check if opponent can win in one move
            opponent = 3 - self.current_player
            for r in range(self.size):
                for c in range(self.size):
                    if self.board[r, c] == 0:
                        # Simulate opponent move
                        self.board[r, c] = opponent
                        if self._check_win(r, c):
                            # Found defensive position
                            self.board[r, c] = 0  # Restore
                            return r * self.size + c
                        self.board[r, c] = 0  # Restore
            
            # Check for complex threats like double-open-three and open-four
            eval_result = self.evaluate_position()
            if eval_result[self.current_player]["threats"]:
                # Found offensive threat position
                r, c = eval_result[self.current_player]["threats"][0]
                return r * self.size + c
            
            # Check for opponent's threat positions
            eval_result = self.evaluate_position(opponent)
            if eval_result[opponent]["threats"]:
                # Found defensive threat position
                r, c = eval_result[opponent]["threats"][0]
                return r * self.size + c
                
        except Exception as e:
            logger.error(f"Error detecting immediate threat: {e}")
            
        return None
    
    def get_zobrist_hash(self) -> int:
        """
        Calculate Zobrist hash of the board, for fast state comparison
        
        Returns:
            int: Zobrist hash value
        """
        # Use fixed random numbers as seeds, for hash consistency
        if not hasattr(self, '_zobrist_table'):
            # Initialize Zobrist table (first call)
            np.random.seed(42)
            self._zobrist_table = np.random.randint(0, 2**64, (3, self.size, self.size), dtype=np.uint64)
        
        # Calculate hash
        h = 0
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    h ^= self._zobrist_table[self.board[r, c], r, c]
        
        return h
    
    def print_board(self):
        """Print current board state to console"""
        symbols = {0: ".", 1: "X", 2: "O"}
        
        # Print column indices
        print("   " + " ".join(str(i % 10) for i in range(self.size)))
        
        # Print rows with row indices
        for i, row in enumerate(self.board):
            print(f"{i:2d} " + " ".join(symbols[cell] for cell in row))
            
        # Print player turn
        player_name = "Black" if self.current_player == 1 else "White"
        print(f"\nCurrent player: {player_name} ({symbols[self.current_player]})")
        
        # Print game state
        if self.game_over:
            if self.winner == 0:
                print("Game ended in a draw")
            else:
                winner_name = "Black" if self.winner == 1 else "White"
                print(f"Game over. Winner: {winner_name}")
    
    def get_symmetries(self, move: int) -> List[Tuple[np.ndarray, int]]:
        """
        Get all symmetric transformations of current board and move
        Used for data augmentation and reducing state space
        
        Args:
            move: 1D move position
            
        Returns:
            List[Tuple[board, move]]: List of symmetric transformations
        """
        row, col = move // self.size, move % self.size
        symmetries = []
        
        # Original board
        board = np.copy(self.board)
        
        # Rotate 90°, 180°, 270°
        for i in range(4):
            # Rotate board
            rot_board = np.rot90(board, i)
            # Calculate rotated move position
            if i == 0:  # 0°
                new_row, new_col = row, col
            elif i == 1:  # 90°
                new_row, new_col = col, self.size - 1 - row
            elif i == 2:  # 180°
                new_row, new_col = self.size - 1 - row, self.size - 1 - col
            else:  # 270°
                new_row, new_col = self.size - 1 - col, row
            
            new_move = new_row * self.size + new_col
            symmetries.append((rot_board, new_move))
            
            # Horizontal flip
            flip_board = np.fliplr(rot_board)
            flip_col = self.size - 1 - new_col
            flip_move = new_row * self.size + flip_col
            symmetries.append((flip_board, flip_move))
        
        return symmetries