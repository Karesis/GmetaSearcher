"""
GomokuAI: Advanced Gomoku (Five in a Row) AI decision engine
Combines FiberTree learning, pattern recognition, threat space search, and Monte Carlo methods
"""

import random
import numpy as np
import time
import pickle
import os
import logging
from typing import List, Tuple, Dict, Set, Optional, Any, Union
from collections import defaultdict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gomoku_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GomokuAI")

# Import board - needed for type checking
from board import GomokuBoard

# Import FiberTree - dynamically handle potential import errors
try:
    from fbtree import create_tree, Move, FiberTree, load_tree
except ImportError as e:
    logger.error(f"Failed to import FiberTree module: {e}")
    logger.error("Please ensure the fbtree module is in your PYTHONPATH")
    
    # Create minimal stand-in classes for type checking 
    class Move:
        def __init__(self, value): self.value = value
    
    def create_tree(*args, **kwargs): return None
    def load_tree(*args, **kwargs): return None
    
    class FiberTree:
        def get_best_continuation(self, *args, **kwargs): return []
        def start_path(self): pass
        def add_move(self, move): pass
        def record_outcome(self, outcome): pass
        def end_path(self): pass
        def find_path(self, moves): return None
        def get_statistics(self, fiber_id): return {"visit_count": 0, "win_count": 0}
        def prune_tree(self, min_visits): return 0
        def analyze_path_diversity(self): return {}
        def merge(self, other_tree, conflict_strategy="stats_sum"): return 0
        def save(self, file_path, compress=True): pass

# Constants
DEFAULT_BOARD_SIZE = 15
MAX_CACHE_SIZE = 100000

class GomokuAI:
    """
    Advanced Gomoku AI using multiple strategies and learning methods
    """
    
    # Opening book - common initial patterns
    OPENING_BOOK = {
        # Center opening variants
        0: [112, 98, 126, 140, 96, 128, 82, 142, 68, 156],  # First move at center position variants
        
        # Common opening sequences - example: [first move, second move, common next moves...]
        112: [98, 126, 97, 127, 113, 111],  # Sequence starting from center point
        98: [112, 84, 126, 97, 127, 82],    # Upper left offset opening
        96: [112, 80, 128, 97, 95, 111],    # Left offset opening
    }
    
    # Recursive search depth limit
    MAX_THREAT_SEARCH_DEPTH = 4
    
    def __init__(self, 
                board_size: int = DEFAULT_BOARD_SIZE, 
                tree: Optional[FiberTree] = None,
                exploration_factor: float = 1.0,
                max_depth: int = 10,
                use_patterns: bool = True,
                use_opening_book: bool = True, 
                storage_type: str = 'memory',
                db_path: Optional[str] = None,
                time_limit: float = 5.0,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Gomoku AI
        
        Args:
            board_size: Size of the board
            tree: Existing FiberTree, a new one is created if not provided
            exploration_factor: Controls exploration vs exploitation balance (higher = more exploration)
            max_depth: Maximum search depth
            use_patterns: Whether to use pattern-based evaluation
            use_opening_book: Whether to use the opening book
            storage_type: 'memory' or 'sqlite' for FiberTree
            db_path: Database path when using sqlite storage
            time_limit: Time limit per move (seconds)
            config: Additional configuration parameters
        """
        self.board_size = board_size
        self.exploration_factor = exploration_factor
        self.max_depth = max_depth
        self.use_patterns = use_patterns
        self.use_opening_book = use_opening_book
        self.player_id = None  # Set when starting a game
        self.time_limit = time_limit
        
        # Default configuration
        self.config = {
            'rollout_depth': 8,         # Monte Carlo rollout depth
            'cache_size': MAX_CACHE_SIZE,  # Transposition table size
            'adaptive_exploration': True,  # Adapt exploration factor based on game phase
            'use_symmetry': True,       # Use board symmetries to reduce state space
            'learning_rate': 0.1,       # Rate at which new experiences are incorporated
            'prune_frequency': 50,      # Number of games between tree pruning
        }
        
        # Update configuration with provided values
        if config:
            self.config.update(config)
        
        # Monte Carlo tree search configuration
        self.mcts_config = {
            'c_puct': 5.0,            # PUCT constant (exploration weight)
            'num_simulations': 800,   # Default simulation count
            'dirichlet_noise': 0.03,  # Dirichlet noise amount (to promote exploration)
            'value_weight': 0.15      # Value network weight
        }
        
        # Initialize FiberTree
        try:
            self.tree = tree if tree else create_tree(
                storage_type=storage_type, 
                db_path=db_path,
                max_cache_size=self.config['cache_size']
            )
        except Exception as e:
            logger.error(f"Failed to create FiberTree: {e}")
            self.tree = None  # Fallback to null object pattern
            
        # Cache and state
        self._transposition_table = {}  # Cache of state -> evaluation
        self._move_history = []  # Record of movement history
        self._state_visits = defaultdict(int)  # Record of state visit counts
        self._opening_phase = True  # Whether in opening phase
        self._endgame_phase = False  # Whether in endgame phase
        
        # Initialize threat assessment scores
        self.threat_weights = {
            'win': 100000,            # Steps that can win directly
            'block_win': 80000,       # Block opponent from winning
            'fork': 20000,            # Create dual threats (like double-four)
            'block_fork': 15000,      # Prevent opponent from creating dual threats
            'connect4': 5000,         # Form a four
            'block_connect4': 4000,   # Block opponent's four
            'connect3': 1000,         # Form a three
            'block_connect3': 800,    # Block opponent's three
        }
        
        # Opening history
        self.opening_stats = {}  # Record opening effectiveness statistics
        
        # Load opening statistics if available
        self._load_opening_stats()
        
        logger.info(f"GomokuAI initialized with board size {board_size}")
        
    def _load_opening_stats(self):
        """Load opening statistics from file with error handling"""
        try:
            stats_path = Path('opening_stats.pkl')
            if stats_path.exists():
                with open(stats_path, 'rb') as f:
                    self.opening_stats = pickle.load(f)
                logger.info(f"Loaded opening statistics with {len(self.opening_stats)} entries")
        except Exception as e:
            logger.error(f"Error loading opening statistics: {e}")
            # Continue with empty statistics rather than crash
            self.opening_stats = {}
    
    def start_game(self, player_id: int):
        """
        Set the AI's player ID (1 for black, 2 for white)
        
        Args:
            player_id: The player ID this AI will play as
        """
        if player_id not in (1, 2):
            raise ValueError("Player ID must be 1 (black) or 2 (white)")
            
        self.player_id = player_id
        self._move_history = []
        self._transposition_table = {}  # Clear cache between games
        self._opening_phase = True
        self._endgame_phase = False
        
        logger.info(f"AI starting game as player {player_id}")
    
    def select_move(self, board: GomokuBoard) -> int:
        """
        Select the best move for the current board state
        
        Args:
            board: Current board state
            
        Returns:
            int: Selected move (1D format)
        """
        # Start timing
        start_time = time.time()
        
        # Check for game over
        if board.game_over:
            logger.warning("Attempt to select move on a completed game")
            return -1
        
        # Get movement history so far
        move_history = [Move(pos) for pos in board.move_history]
        self._move_history = move_history
        
        # Check board state to determine game phase
        self._update_game_phase(board)
        
        # Adapt exploration factor if configured
        if self.config['adaptive_exploration']:
            self._adjust_exploration_factor(board)
        
        # 1. Check for immediate threats
        immediate_threat = board.detect_immediate_threat()
        if immediate_threat is not None:
            logger.info(f"Playing immediate threat at position {immediate_threat}")
            return immediate_threat
        
        # 2. If in opening and using opening book, check for preset moves
        if self.use_opening_book and self._opening_phase:
            book_move = self._get_book_move(board)
            if book_move is not None:
                logger.info(f"Playing book move at position {book_move}")
                return book_move
        
        # 3. Get legal moves
        legal_moves = board.get_focused_moves(distance=3, consider_threats=True)
        
        # If no focused moves or early game, get all legal moves
        if not legal_moves or len(move_history) < 4:
            legal_moves = board.get_legal_moves()
        
        # If only one legal move, return it directly
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # 4. Use threat space search to find forcing moves
        if not self._endgame_phase and len(move_history) > 6:
            forcing_move = self._threat_space_search(board)
            if forcing_move is not None:
                logger.info(f"Playing forcing move at position {forcing_move}")
                return forcing_move
        
        # 5. Try to get best continuation from FiberTree
        fiber_move = self._get_best_fiber_move(board, legal_moves)
        if fiber_move is not None:
            return fiber_move
        
        # 6. Select decision method based on game phase
        elapsed = time.time() - start_time
        if self._endgame_phase or elapsed > self.time_limit * 0.3:
            # Endgame or limited time: use pattern evaluation
            return self._select_move_by_patterns(board, legal_moves)
        else:
            # Mid-game and late-opening: use Monte Carlo tree search
            remaining_time = max(0.1, self.time_limit - elapsed)
            return self._select_move_by_mcts(board, legal_moves, remaining_time)
    
    def _get_best_fiber_move(self, board: GomokuBoard, legal_moves: List[int]) -> Optional[int]:
        """
        Try to get the best move from FiberTree history
        
        Args:
            board: Current board
            legal_moves: List of legal moves
            
        Returns:
            Optional[int]: Best move or None if no good move found
        """
        if len(self._move_history) > 8 and self.tree:  # When there's enough history
            try:
                best_continuations = self.tree.get_best_continuation(
                    self._move_history, 
                    top_n=min(5, len(legal_moves)),
                    min_visits=2
                )
                
                # If continuations with enough data found
                if best_continuations and best_continuations[0]['visits'] >= 3:
                    # Probabilistically select based on win rate and visit count
                    total_score = sum(c['win_rate'] * (c['visits'] ** 0.5) for c in best_continuations)
                    if total_score > 0:
                        # Introduce randomness, but prefer high win rates
                        rand_val = random.random() * total_score
                        cumulative = 0
                        for cont in best_continuations:
                            cumulative += cont['win_rate'] * (cont['visits'] ** 0.5)
                            if cumulative >= rand_val:
                                # Verify move is legal
                                if cont['move'].value in legal_moves:
                                    logger.info(f"Playing move from FiberTree with win rate {cont['win_rate']:.2f}")
                                    return cont['move'].value
                    
                    # Fallback: use best move if legal
                    if best_continuations[0]['move'].value in legal_moves:
                        logger.info(f"Playing best move from FiberTree with win rate {best_continuations[0]['win_rate']:.2f}")
                        return best_continuations[0]['move'].value
            except Exception as e:
                logger.error(f"Error getting move from FiberTree: {e}")
        
        return None
    
    def _update_game_phase(self, board: GomokuBoard):
        """Update the game phase based on board state"""
        move_count = len(board.move_history)
        
        # Opening phase determination (first 12 steps)
        self._opening_phase = move_count <= 12
        
        # Endgame phase determination (board filled over 60% or critical threats present)
        threshold = (self.board_size ** 2) * 0.6
        self._endgame_phase = move_count >= threshold
        
        # Also consider endgame if critical threat patterns detected
        if not self._endgame_phase and move_count > 30:
            try:
                evaluation = board.evaluate_position()
                # If either player has advanced threats, consider it endgame
                for player in [1, 2]:
                    if (evaluation[player]["patterns"].get("five", 0) > 0 or
                        evaluation[player]["patterns"].get("open_four", 0) > 0 or
                        evaluation[player]["patterns"].get("four", 0) > 1):
                        self._endgame_phase = True
                        break
            except Exception as e:
                logger.error(f"Error evaluating position for phase update: {e}")
    
    def _adjust_exploration_factor(self, board: GomokuBoard):
        """Adaptively adjust exploration factor based on game phase"""
        move_count = len(board.move_history)
        board_capacity = self.board_size ** 2
        
        if self._opening_phase:
            # Opening phase: encourage exploration
            self.exploration_factor = 1.4
        elif self._endgame_phase:
            # Endgame: reduce exploration, focus on exploitation
            self.exploration_factor = 0.6
        else:
            # Mid-game: linear decrease from 1.2 to 0.8
            mid_game_progress = min(1.0, (move_count - 12) / (board_capacity * 0.6 - 12))
            self.exploration_factor = 1.2 - mid_game_progress * 0.4
    
    def _get_book_move(self, board: GomokuBoard) -> Optional[int]:
        """Get a move from the opening book"""
        move_history = board.move_history
        
        # Special handling for first move
        if not move_history:
            if 0 in self.OPENING_BOOK:
                # Randomly select an opening variant
                candidate_moves = self.OPENING_BOOK[0]
                # Select best performing opening from statistics
                if self.opening_stats:
                    best_moves = []
                    best_win_rate = 0
                    for move in candidate_moves:
                        stats = self.opening_stats.get(move, {"wins": 0, "games": 0})
                        if stats["games"] > 0:
                            win_rate = stats["wins"] / stats["games"]
                            if win_rate > best_win_rate:
                                best_win_rate = win_rate
                                best_moves = [move]
                            elif win_rate == best_win_rate:
                                best_moves.append(move)
                    
                    if best_moves and best_win_rate > 0.4:  # Only use statistics if win rate is sufficient
                        return random.choice(best_moves)
                
                return random.choice(candidate_moves)
        
        # Check if current history matches an opening sequence
        if move_history:
            last_move = move_history[-1]
            if last_move in self.OPENING_BOOK:
                responses = self.OPENING_BOOK[last_move]
                for response in responses:
                    row, col = response // board.size, response % board.size
                    if board.board[row, col] == 0:  # Ensure position is unoccupied
                        return response
        
        return None
    
    def _threat_space_search(self, board: GomokuBoard) -> Optional[int]:
        """
        Threat space search for forcing moves
        
        Args:
            board: Current board
            
        Returns:
            Optional[int]: Forcing move if found, None otherwise
        """
        try:
            # Find all threat positions for current player
            eval_result = board.evaluate_position()
            threats_current = eval_result[self.player_id]["threats"]
            threats_opponent = eval_result[3 - self.player_id]["threats"]
            
            # Stage 1: Check direct threats
            if threats_current:
                # Check for winning threats
                for r, c in threats_current:
                    # Simulate placing a stone
                    board.board[r, c] = self.player_id
                    if board._check_win(r, c):
                        # If this move can win, return it
                        board.board[r, c] = 0  # Restore
                        return r * board.size + c
                    board.board[r, c] = 0  # Restore
            
            # Stage 2: Threat extension search
            # We search for positions that create overlapping threats (e.g., double-four, three-four)
            best_move = None
            best_score = -1
            
            # Get focused moves
            focused_moves = board.get_focused_moves(distance=2)
            
            for move in focused_moves:
                row, col = move // board.size, move % board.size
                
                # Simulate this move
                board.board[row, col] = self.player_id
                
                # Calculate threat potential of this move
                threat_score = self._evaluate_threat_potential(board, row, col, self.player_id, 1)
                
                # Restore board
                board.board[row, col] = 0
                
                # Update best move
                if threat_score > best_score:
                    best_score = threat_score
                    best_move = move
            
            # If a move with sufficient threat potential is found, return it
            if best_score >= self.threat_weights['connect4']:
                return best_move
            
            # Stage 3: Block opponent threats
            if threats_opponent:
                return threats_opponent[0][0] * board.size + threats_opponent[0][1]
                
        except Exception as e:
            logger.error(f"Error in threat space search: {e}")
        
        # No forcing move found
        return None
    
    def _evaluate_threat_potential(self, board: GomokuBoard, row: int, col: int, player: int, depth: int) -> float:
        """
        Evaluate the threat potential of a move
        
        Args:
            board: Game board
            row, col: Move position
            player: Current player
            depth: Current search depth
            
        Returns:
            float: Threat score
        """
        if depth > self.MAX_THREAT_SEARCH_DEPTH:
            return 0
        
        opponent = 3 - player
        threat_score = 0
        
        # Check for win
        if board._check_win(row, col):
            return self.threat_weights['win'] / depth  # Earlier wins valued higher
        
        # Evaluate threats created by this move
        try:
            eval_current = board.evaluate_position(player)
            
            # Calculate threat score
            patterns = eval_current[player]["patterns"]
            
            # Calculate direct threats
            if patterns.get("open_four", 0) > 0:
                threat_score += self.threat_weights['connect4']
            if patterns.get("four", 0) > 0:
                threat_score += self.threat_weights['connect4'] * 0.8
            
            # Detect multiple threats
            # e.g., double-three, double-four, or three-four combinations
            threat_count = (patterns.get("open_three", 0) + 
                          patterns.get("three", 0) * 0.5 + 
                          patterns.get("open_four", 0) * 2 + 
                          patterns.get("four", 0))
            
            if threat_count >= 2:
                threat_score += self.threat_weights['fork'] / depth
            
            # Recursively check opponent's counter, but limit depth
            if depth < self.MAX_THREAT_SEARCH_DEPTH and threat_score > 0:
                # Get opponent's best counter
                best_counter_threat = 0
                
                # Get opponent's possible counters (focus on threat area)
                opponent_moves = []
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        r, c = row + dr, col + dc
                        if 0 <= r < board.size and 0 <= c < board.size and board.board[r, c] == 0:
                            opponent_moves.append((r, c))
                
                # If too many possible counters, only consider threatening positions
                if len(opponent_moves) > 6:
                    eval_opponent = board.evaluate_position(opponent)
                    opponent_threats = eval_opponent[opponent]["threats"]
                    if opponent_threats:
                        opponent_moves = [(r, c) for r, c in opponent_threats if board.board[r, c] == 0]
                
                # Evaluate each of opponent's counters
                for r, c in opponent_moves:
                    board.board[r, c] = opponent
                    counter_threat = self._evaluate_threat_potential(board, r, c, opponent, depth + 1)
                    board.board[r, c] = 0
                    
                    best_counter_threat = max(best_counter_threat, counter_threat)
                
                # If opponent has strong counter, reduce our threat value
                if best_counter_threat > self.threat_weights['connect3']:
                    threat_score *= 0.5
                    
        except Exception as e:
            logger.error(f"Error evaluating threat potential: {e}")
            return 0
        
        return threat_score
    
    def _select_move_by_patterns(self, board: GomokuBoard, legal_moves: List[int]) -> int:
        """
        Select move based on pattern recognition and heuristic evaluation
        
        Args:
            board: Current board
            legal_moves: List of legal moves
            
        Returns:
            int: Selected move
        """
        best_move = None
        best_score = float('-inf')
        opponent = 3 - self.player_id
        
        # Try each move and evaluate resulting position
        for move in legal_moves:
            row, col = move // board.size, move % board.size
            
            try:
                # Use board's scoring method to quickly evaluate (avoiding creating temporary boards)
                my_score = board.get_score_for_move(row, col, self.player_id)
                opponent_score = board.get_score_for_move(row, col, opponent)
                
                # Balance offense and defense
                # Give blocking opponent moves higher weight (1.2x)
                combined_score = my_score - opponent_score * 1.2
                
                # If opening, add position preference
                if len(board.move_history) < 10:
                    center = board.size // 2
                    distance_from_center = abs(row - center) + abs(col - center)
                    combined_score -= distance_from_center * 5
                
                # Add some randomness for variety
                combined_score += random.random() * 5
                
                # Update best move
                if combined_score > best_score:
                    best_score = combined_score
                    best_move = move
            except Exception as e:
                logger.error(f"Error evaluating move {move}: {e}")
                continue
        
        # If a good move was found, return it
        if best_move is not None:
            return best_move
        
        # Fallback: choose a random legal move
        return random.choice(legal_moves) if legal_moves else -1
    
    def _select_move_by_mcts(self, board: GomokuBoard, legal_moves: List[int], time_limit: float) -> int:
        """
        Select move using Monte Carlo Tree Search
        
        Args:
            board: Current board
            legal_moves: List of legal moves
            time_limit: Search time limit (seconds)
            
        Returns:
            int: Selected move
        """
        class MCTSNode:
            """Monte Carlo Tree Search Node"""
            def __init__(self, prior=0):
                self.visit_count = 0
                self.value_sum = 0
                self.children = {}
                self.prior = prior
            
            def get_value(self, parent_visit_count):
                # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
                if self.visit_count == 0:
                    return float('inf') if self.prior > 0 else 0
                
                # Exploration term
                exploration = (c_puct * self.prior * 
                              (parent_visit_count ** 0.5) / 
                              (1 + self.visit_count))
                
                # Value term (average of values)
                value = self.value_sum / self.visit_count
                
                return value + exploration
        
        try:
            # Create a temporary board from current state for MCTS search
            mcts_board = type(board)(board.size)
            mcts_board.board = np.copy(board.board)
            mcts_board.current_player = board.current_player
            
            # Search configuration
            c_puct = self.mcts_config['c_puct']
            num_simulations = self.mcts_config['num_simulations']
            
            # Root node
            root = MCTSNode()
            root.visit_count = 1  # Avoid division by zero
            
            # Initialize root node's children
            for move in legal_moves:
                # Initialize with simple prior probability
                prior_prob = 1.0 / len(legal_moves)
                
                # If FiberTree statistics available, improve prior
                if self.tree:
                    try:
                        move_obj = Move(move)
                        fiber_id = self.tree.find_path(self._move_history + [move_obj])
                        if fiber_id:
                            stats = self.tree.get_statistics(fiber_id)
                            if stats['visit_count'] > 0:
                                # Use past win rate as prior
                                win_rate = stats['win_count'] / stats['visit_count']
                                prior_prob = max(prior_prob, win_rate)
                    except Exception as e:
                        logger.error(f"Error getting FiberTree statistics: {e}")
                
                # Create child node
                root.children[move] = MCTSNode(prior=prior_prob)
            
            # Add Dirichlet noise to increase exploration
            if legal_moves:
                noise = np.random.dirichlet([self.mcts_config['dirichlet_noise']] * len(legal_moves))
                for i, move in enumerate(legal_moves):
                    # 95% weight to prior, 5% to noise
                    root.children[move].prior = 0.95 * root.children[move].prior + 0.05 * noise[i]
            
            # MCTS main loop
            start_time = time.time()
            num_simulations_completed = 0
            
            while (time.time() - start_time < time_limit and 
                  num_simulations_completed < num_simulations):
                # Selection phase: select most valuable path
                node = root
                search_path = [node]
                sim_board = type(board)(board.size)
                sim_board.board = np.copy(board.board)
                sim_board.current_player = board.current_player
                current_moves = []
                
                # Selection phase
                while node.children and not sim_board.game_over:
                    # Select action with highest UCB value
                    max_value = float('-inf')
                    best_move = None
                    
                    for move, child in node.children.items():
                        ucb_value = child.get_value(node.visit_count)
                        if ucb_value > max_value:
                            max_value = ucb_value
                            best_move = move
                    
                    if best_move is None:
                        break
                    
                    # Execute best action
                    node = node.children[best_move]
                    search_path.append(node)
                    
                    row, col = best_move // sim_board.size, best_move % sim_board.size
                    sim_board.make_move(row, col)
                    current_moves.append(best_move)
                
                # Expansion phase
                if not sim_board.game_over:
                    # Expand this node
                    moves = sim_board.get_focused_moves()
                    if not moves:
                        moves = sim_board.get_legal_moves()
                    
                    # Initialize each possible action
                    for move in moves:
                        if move not in node.children:
                            node.children[move] = MCTSNode(prior=1.0/len(moves))
                
                # Simulation phase (Rollout)
                if sim_board.game_over:
                    # Game already ended, use the determined result
                    if sim_board.winner == self.player_id:
                        value = 1.0
                    elif sim_board.winner == 0:  # Draw
                        value = 0.0
                    else:  # Opponent wins
                        value = -1.0
                else:
                    # Perform fast rollout
                    value = self._fast_rollout(sim_board)
                
                # Backpropagation phase
                for node in search_path:
                    node.visit_count += 1
                    node.value_sum += value
                    value = -value  # Alternate perspective
                
                num_simulations_completed += 1
            
            # Select most visited action
            max_visits = -1
            best_move = legal_moves[0] if legal_moves else -1
            
            for move, child in root.children.items():
                if child.visit_count > max_visits:
                    max_visits = child.visit_count
                    best_move = move
            
            # Add search result to FiberTree
            if self.tree:
                try:
                    self.tree.start_path()
                    for move in self._move_history:
                        self.tree.add_move(move)
                    self.tree.add_move(Move(best_move))
                    
                    # Add slight prior indicating we chose this action but don't know result yet
                    self.tree.record_outcome('draw')  # Temporarily record as draw
                    self.tree.end_path()
                except Exception as e:
                    logger.error(f"Error updating FiberTree: {e}")
            
            return best_move
            
        except Exception as e:
            logger.error(f"Error in MCTS selection: {e}")
            # Fallback to pattern-based selection
            return self._select_move_by_patterns(board, legal_moves)
    
    def _fast_rollout(self, board: GomokuBoard) -> float:
        """
        Perform fast random rollout to estimate position value
        
        Args:
            board: Current board
            
        Returns:
            float: Estimated value (from current player perspective)
        """
        try:
            sim_board = type(board)(board.size)
            sim_board.board = np.copy(board.board)
            sim_board.current_player = board.current_player
            depth = 0
            
            # Use more advanced rollout strategy rather than pure random
            while not sim_board.game_over and depth < self.config['rollout_depth']:
                # First check for winning or defensive moves
                immediate_move = sim_board.detect_immediate_threat()
                if immediate_move is not None:
                    row, col = immediate_move // sim_board.size, immediate_move % sim_board.size
                    sim_board.make_move(row, col)
                else:
                    # Get legal moves
                    moves = sim_board.get_focused_moves(distance=2)
                    if not moves:
                        moves = sim_board.get_legal_moves()
                    
                    if not moves:
                        break
                    
                    # Evaluate all moves and select probabilistically (not purely random)
                    scores = []
                    for move in moves:
                        row, col = move // sim_board.size, move % sim_board.size
                        score = sim_board.get_score_for_move(row, col, sim_board.current_player)
                        scores.append(max(1.0, score))  # Ensure all moves have at least some probability
                    
                    # Random selection, but favor high-scoring moves
                    total = sum(scores)
                    r = random.random() * total
                    cum_sum = 0
                    selected_move = moves[0]
                    
                    for i, score in enumerate(scores):
                        cum_sum += score
                        if cum_sum >= r:
                            selected_move = moves[i]
                            break
                    
                    row, col = selected_move // sim_board.size, selected_move % sim_board.size
                    sim_board.make_move(row, col)
                
                depth += 1
            
            # If game ended during rollout
            if sim_board.game_over:
                if sim_board.winner == 0:  # Draw
                    return 0.0
                return 1.0 if sim_board.winner == self.player_id else -1.0
            
            # Use simple heuristic to evaluate unfinished games
            eval_result = sim_board.evaluate_position()
            my_score = eval_result[self.player_id]["score"]
            opp_score = eval_result[3 - self.player_id]["score"]
            
            # Normalize to -1 to 1 range
            normalized_score = np.tanh((my_score - opp_score) / 1000.0)
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error in fast rollout: {e}")
            return 0.0  # Neutral evaluation as fallback
    
    def learn_from_game(self, board: GomokuBoard, winner: int):
        """
        Update FiberTree from completed game
        
        Args:
            board: Final board state
            winner: Game winner (0=draw, 1=black, 2=white)
        """
        if not board.game_over:
            logger.warning("Attempted to learn from incomplete game")
            return
        
        if not self.tree:
            logger.warning("Cannot learn without FiberTree")
            return
            
        try:
            # Convert move history to FiberTree moves
            move_sequence = [Move(pos) for pos in board.move_history]
            
            # Record result
            outcome = 'draw'
            if winner == self.player_id:
                outcome = 'win'
            elif winner != 0:  # Not a draw
                outcome = 'loss'
            
            # Store path in FiberTree
            self.tree.start_path()
            for move in move_sequence:
                self.tree.add_move(move)
            self.tree.record_outcome(outcome)
            self.tree.end_path()
            
            # Update opening statistics
            if self.use_opening_book and board.move_history:
                first_move = board.move_history[0]
                if first_move not in self.opening_stats:
                    self.opening_stats[first_move] = {"wins": 0, "games": 0}
                
                self.opening_stats[first_move]["games"] += 1
                if (self.player_id == 1 and winner == 1) or (self.player_id == 2 and winner == 2):
                    self.opening_stats[first_move]["wins"] += 1
                
                # Save opening statistics
                self._save_opening_stats()
            
            # If using symmetry, also store equivalent transformations
            if self.config['use_symmetry']:
                self._learn_from_symmetries(board, move_sequence, outcome)
                
        except Exception as e:
            logger.error(f"Error learning from game: {e}")
    
    def _save_opening_stats(self):
        """Save opening statistics with error handling"""
        try:
            with open('opening_stats.pkl', 'wb') as f:
                pickle.dump(self.opening_stats, f)
        except Exception as e:
            logger.error(f"Error saving opening statistics: {e}")
    
    def _learn_from_symmetries(self, board: GomokuBoard, move_sequence: List[Move], outcome: str):
        """Learn from symmetric board positions"""
        if not self.tree:
            return
            
        try:
            last_state = None
            for i, move in enumerate(move_sequence):
                # Store symmetric transformations for first 20 moves, later steps have less symmetry value
                if i < 20:
                    # Rebuild board to this step
                    if last_state is None:
                        temp_board = type(board)(board.size)
                        for j in range(i):
                            prev_move = move_sequence[j].value
                            r, c = prev_move // board.size, prev_move % board.size
                            temp_board.make_move(r, c)
                        last_state = temp_board
                    else:
                        # Continue from previous state
                        r, c = move_sequence[i-1].value // board.size, move_sequence[i-1].value % board.size
                        last_state.make_move(r, c)
                    
                    # Get symmetric transformations of current move
                    symmetries = last_state.get_symmetries(move.value)
                    
                    # Store each symmetric transformation
                    for sym_board, sym_move in symmetries:
                        # Skip original state
                        if sym_move == move.value:
                            continue
                            
                        # Create new path
                        self.tree.start_path()
                        
                        # Add moves for this symmetric transformation
                        for j in range(i):
                            prev_move = move_sequence[j].value
                            sym_prev_move = None
                            
                            # Determine corresponding position in symmetric transformation
                            r, c = prev_move // board.size, prev_move % board.size
                            for _, test_move in last_state.get_symmetries(prev_move):
                                if test_move != prev_move:
                                    sym_prev_move = test_move
                                    break
                            
                            if sym_prev_move is not None:
                                self.tree.add_move(Move(sym_prev_move))
                            else:
                                self.tree.add_move(Move(prev_move))
                        
                        # Add current symmetric move
                        self.tree.add_move(Move(sym_move))
                        
                        # Record result and end path
                        self.tree.record_outcome(outcome)
                        self.tree.end_path()
        except Exception as e:
            logger.error(f"Error learning from symmetries: {e}")
    
    def save_knowledge(self, file_path: str, compress: bool = True):
        """
        Save FiberTree to file
        
        Args:
            file_path: Path to save knowledge
            compress: Whether to compress output
        """
        if not self.tree:
            logger.warning("No FiberTree to save")
            return
            
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.tree.save(file_path, compress=compress)
            logger.info(f"Saved knowledge to {file_path}")
            
            # Save opening statistics
            try:
                opening_stats_path = str(Path(file_path).with_suffix('')) + '_openings.pkl'
                with open(opening_stats_path, 'wb') as f:
                    pickle.dump(self.opening_stats, f)
            except Exception as e:
                logger.error(f"Error saving opening statistics: {e}")
        except Exception as e:
            logger.error(f"Error saving knowledge: {e}")
    
    def load_knowledge(self, file_path: str):
        """
        Load FiberTree from file
        
        Args:
            file_path: Path to knowledge file
        """
        try:
            self.tree = load_tree(file_path)
            logger.info(f"Loaded knowledge from {file_path}")
            
            # Try to load opening statistics
            try:
                opening_stats_path = str(Path(file_path).with_suffix('')) + '_openings.pkl'
                if os.path.exists(opening_stats_path):
                    with open(opening_stats_path, 'rb') as f:
                        self.opening_stats = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading opening statistics: {e}")
        except Exception as e:
            logger.error(f"Error loading knowledge: {e}")
    
    def prune_knowledge(self, min_visits: int = 3) -> int:
        """
        Prune FiberTree, removing rarely visited paths
        
        Args:
            min_visits: Minimum visit count for paths to be retained
            
        Returns:
            int: Number of pruned paths
        """
        if not self.tree:
            logger.warning("No FiberTree to prune")
            return 0
            
        try:
            pruned = self.tree.prune_tree(min_visits=min_visits)
            logger.info(f"Pruned {pruned} paths from knowledge tree")
            return pruned
        except Exception as e:
            logger.error(f"Error pruning knowledge: {e}")
            return 0
    
    def analyze_knowledge(self) -> Dict[str, Any]:
        """
        Analyze knowledge in FiberTree
        
        Returns:
            Dict: Analysis of FiberTree
        """
        if not self.tree:
            logger.warning("No FiberTree to analyze")
            return {}
            
        try:
            analysis = self.tree.analyze_path_diversity()
            
            # Add opening statistics analysis
            if self.opening_stats:
                top_openings = []
                for move, stats in self.opening_stats.items():
                    if stats["games"] >= 5:
                        win_rate = stats["wins"] / stats["games"]
                        top_openings.append((move, win_rate, stats["games"]))
                
                top_openings.sort(key=lambda x: x[1], reverse=True)
                analysis["top_openings"] = top_openings[:5]
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing knowledge: {e}")
            return {}
    
    def get_best_first_moves(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get best opening moves based on statistics
        
        Args:
            top_n: Number of top moves to return
            
        Returns:
            List[Dict]: Information about best opening moves
        """
        if not self.tree:
            logger.warning("No FiberTree for move analysis")
            return []
            
        try:
            # Check all possible first moves
            first_moves = []
            for pos in range(self.board_size * self.board_size):
                # Check FiberTree statistics
                fiber_id = self.tree.find_path([Move(pos)])
                if fiber_id:
                    stats = self.tree.get_statistics(fiber_id)
                    if stats['visit_count'] > 0:
                        win_rate = stats['win_count'] / stats['visit_count']
                        first_moves.append({
                            'position': pos,
                            'row': pos // self.board_size,
                            'col': pos % self.board_size,
                            'visits': stats['visit_count'],
                            'win_rate': win_rate,
                            'fiber_id': fiber_id
                        })
                
                # Merge opening book statistics
                if pos in self.opening_stats:
                    book_stats = self.opening_stats[pos]
                    if book_stats["games"] > 0:
                        book_win_rate = book_stats["wins"] / book_stats["games"]
                        
                        # Check if already in list
                        found = False
                        for move in first_moves:
                            if move['position'] == pos:
                                # Combine both statistics
                                total_games = move['visits'] + book_stats["games"]
                                combined_win_rate = ((move['win_rate'] * move['visits']) + 
                                                  (book_win_rate * book_stats["games"])) / total_games
                                
                                move['visits'] = total_games
                                move['win_rate'] = combined_win_rate
                                found = True
                                break
                        
                        if not found:
                            first_moves.append({
                                'position': pos,
                                'row': pos // self.board_size,
                                'col': pos % self.board_size,
                                'visits': book_stats["games"],
                                'win_rate': book_win_rate,
                                'from_book': True
                            })
            
            # Sort by win rate and visit count
            sorted_moves = sorted(first_moves, 
                                key=lambda x: (x['win_rate'], x['visits']), 
                                reverse=True)
            
            return sorted_moves[:top_n]
            
        except Exception as e:
            logger.error(f"Error getting best first moves: {e}")
            return []