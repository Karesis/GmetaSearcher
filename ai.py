"""
GomokuAI: Advanced Gomoku (Five in a Row) AI decision engine with root parallelization
"""

import random
import numpy as np
import time
import pickle
import os
from typing import List, Tuple, Dict, Set, Optional, Any, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Import FiberTree
from fbtree import create_tree, Move, FiberTree, load_tree

class GomokuAI:
    """Advanced Gomoku AI using multiple strategies and learning methods with parallelization"""
    
    # Opening book - common initial patterns
    OPENING_BOOK = {
        # Center opening variants
        0: [112, 98, 126, 140, 96, 128, 82, 142, 68, 156],  # First move at center position
        
        # Common opening sequences
        112: [98, 126, 97, 127, 113, 111],  # Sequence starting from center point
        98: [112, 84, 126, 97, 127, 82],    # Upper left offset opening
        96: [112, 80, 128, 97, 95, 111],    # Left offset opening
    }
    
    # Recursive search depth limit
    MAX_THREAT_SEARCH_DEPTH = 4
    
    def __init__(self, board_size: int = 15, tree: Optional[FiberTree] = None, num_threads: int = 4):
        """Initialize the Gomoku AI with parallel processing capability"""
        self.board_size = board_size
        self.player_id = None  # Set when starting a game
        
        # Advanced AI configuration with adjustable parameters
        self.exploration_factor = 0.8
        self.max_depth = 10
        self.time_limit = 2.0  # Reduced from 5.0 to 2.0 seconds for faster play
        self.num_threads = min(num_threads, os.cpu_count() or 4)  # Use at most available CPU cores
        
        # MCTS configuration
        self.mcts_config = {
            'c_puct': 3.0,              # PUCT constant (exploration weight)
            'num_simulations': 300,     # Reduced from 800 to 300 simulations per thread
            'dirichlet_noise': 0.01,    # Dirichlet noise amount for exploration
            'virtual_loss': 1.5,        # Virtual loss for thread coordination
        }
        
        # Initialize tree with optimized cache size
        self.tree = tree if tree else create_tree(max_cache_size=50000)
        
        # Cache and state
        self._transposition_table = {}  # Cache of state -> evaluation
        self._move_history = []  # Record of movement history
        self._state_visits = defaultdict(int)  # Record of state visit counts
        self._opening_phase = True  # Whether in opening phase
        self._endgame_phase = False  # Whether in endgame phase
        
        # Threat assessment scores
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
        
        # Opening stats
        self.opening_stats = {}
        self._load_opening_stats()
    
    def _load_opening_stats(self):
        """Load opening statistics from file if available"""
        try:
            stats_path = 'opening_stats.pkl'
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    self.opening_stats = pickle.load(f)
        except Exception:
            self.opening_stats = {}
    
    def start_game(self, player_id: int):
        """Set the AI's player ID (1 for black, 2 for white)"""
        if player_id not in (1, 2):
            raise ValueError("Player ID must be 1 (black) or 2 (white)")
            
        self.player_id = player_id
        self._move_history = []
        self._transposition_table = {}  # Clear cache between games
        self._opening_phase = True
        self._endgame_phase = False
    
    def select_move(self, board):
        """
        Select the best move for the current board state with parallel processing
        
        Args:
            board: Current board state
            
        Returns:
            int: Selected move (1D format)
        """
        # Start timing
        start_time = time.time()
        
        # Check for game over
        if board.game_over:
            return -1
        
        # Get movement history so far
        move_history = [Move(pos) for pos in board.move_history]
        self._move_history = move_history
        
        # Check board state to determine game phase
        self._update_game_phase(board)
        
        # Progress indicator
        print(f"Thinking... ", end='', flush=True)
        
        # 1. Check for immediate threats - this is fast and high priority
        immediate_threat = board.detect_immediate_threat()
        if immediate_threat is not None:
            print(f"Found immediate threat ({time.time() - start_time:.2f}s)")
            return immediate_threat
        
        # 2. If in opening, check opening book - also fast
        if self._opening_phase:
            book_move = self._get_book_move(board)
            if book_move is not None:
                print(f"Using opening book ({time.time() - start_time:.2f}s)")
                return book_move
        
        # 3. Get focused legal moves
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
                print(f"Using threat space search ({time.time() - start_time:.2f}s)")
                return forcing_move
        
        # 5. Try to get best continuation from FiberTree - leverages learning
        fiber_move = self._get_best_fiber_move(board, legal_moves)
        if fiber_move is not None:
            print(f"Using fiber tree ({time.time() - start_time:.2f}s)")
            return fiber_move
        
        # 6. Select decision method based on game phase
        elapsed = time.time() - start_time
        if self._endgame_phase or elapsed > self.time_limit * 0.3:
            # Endgame or limited time: use pattern evaluation
            move = self._select_move_by_patterns(board, legal_moves)
            print(f"Using pattern evaluation ({time.time() - start_time:.2f}s)")
            return move
        else:
            # Mid-game and late-opening: use parallel Monte Carlo tree search
            remaining_time = max(0.1, self.time_limit - elapsed)
            move = self._parallel_mcts(board, legal_moves, remaining_time)
            print(f"Using parallel MCTS ({time.time() - start_time:.2f}s)")
            return move
    
    def _get_best_fiber_move(self, board, legal_moves):
        """Try to get the best move from FiberTree history with optimized probability weighting"""
        if len(self._move_history) > 6 and self.tree:  # Reduced threshold from 8 to 6 to use fiber tree earlier
            try:
                # Get more candidate moves
                best_continuations = self.tree.get_best_continuation(
                    self._move_history, 
                    top_n=min(8, len(legal_moves)),  # Increased from 5 to 8
                    min_visits=2
                )
                
                # If continuations with enough data found
                if best_continuations and best_continuations[0]['visits'] >= 3:
                    # Improved probabilistic selection
                    valid_continuations = [c for c in best_continuations if c['move'].value in legal_moves]
                    
                    if valid_continuations:
                        # Apply temperature to control exploration/exploitation
                        temperature = 1.0 if self._opening_phase else 0.5
                        
                        # Calculate scores with temperature
                        scores = [
                            (c['win_rate'] * (c['visits'] ** (1/3))) ** (1/temperature)  
                            for c in valid_continuations
                        ]
                        
                        # Normalize scores to probabilities
                        total = sum(scores)
                        if total > 0:
                            probs = [s / total for s in scores]
                            
                            # Sample move based on probabilities
                            selected_idx = np.random.choice(len(valid_continuations), p=probs)
                            return valid_continuations[selected_idx]['move'].value
                    
                    # Fallback: use best move if legal
                    if best_continuations[0]['move'].value in legal_moves:
                        return best_continuations[0]['move'].value
            except Exception:
                pass
        
        return None
    
    def _update_game_phase(self, board):
        """Update the game phase based on board state"""
        move_count = len(board.move_history)
        
        # Opening phase determination (first 12 steps)
        self._opening_phase = move_count <= 12
        
        # Endgame phase determination (board filled over 60% or critical threats present)
        threshold = (self.board_size ** 2) * 0.6
        self._endgame_phase = move_count >= threshold
        
        # Also consider endgame if critical threat patterns detected
        if not self._endgame_phase and move_count > 30:
            evaluation = board.evaluate_position()
            # If either player has advanced threats, consider it endgame
            for player in [1, 2]:
                if (evaluation[player]["patterns"].get("five", 0) > 0 or
                    evaluation[player]["patterns"].get("open_four", 0) > 0 or
                    evaluation[player]["patterns"].get("four", 0) > 1):
                    self._endgame_phase = True
                    break
    
    def _get_book_move(self, board):
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
    
    def _threat_space_search(self, board):
        """Threat space search for forcing moves"""
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
                
        except Exception:
            pass
        
        # No forcing move found
        return None
    
    def _evaluate_threat_potential(self, board, row: int, col: int, player: int, depth: int) -> float:
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
                
        return threat_score
    
    def _select_move_by_patterns(self, board, legal_moves):
        """Select move based on pattern recognition and heuristic evaluation"""
        best_move = None
        best_score = float('-inf')
        opponent = 3 - self.player_id
        
        # Try each move and evaluate resulting position
        for move in legal_moves:
            row, col = move // board.size, move % board.size
            
            try:
                # Use board's scoring method to quickly evaluate
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
                
                # Add some randomness for variety (reduced from 5 to 3)
                combined_score += random.random() * 3
                
                # Update best move
                if combined_score > best_score:
                    best_score = combined_score
                    best_move = move
            except Exception:
                continue
        
        # If a good move was found, return it
        if best_move is not None:
            return best_move
        
        # Fallback: choose a random legal move
        return random.choice(legal_moves) if legal_moves else -1
    
    def _parallel_mcts(self, board, legal_moves, time_limit):
        """
        Run MCTS in parallel using root parallelization
        
        Args:
            board: Current board state
            legal_moves: List of legal moves
            time_limit: Time limit for search
            
        Returns:
            int: Best move
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
        
        # Calculate simulations per thread
        num_threads = min(self.num_threads, len(legal_moves))
        simulations_per_thread = self.mcts_config['num_simulations'] // num_threads
        c_puct = self.mcts_config['c_puct']
        dirichlet_noise = self.mcts_config['dirichlet_noise']
        
        # Function to run MCTS on a single thread
        def run_mcts_thread(thread_id, num_simulations):
            # Create a copy of the board
            thread_board = type(board)(board.size)
            thread_board.board = np.copy(board.board)
            thread_board.current_player = board.current_player
            
            # Create root node
            root = MCTSNode()
            root.visit_count = 1  # Avoid division by zero
            
            # Initialize children with priors
            for move in legal_moves:
                # Initialize with prior probability
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
                    except Exception:
                        pass
                
                # Create child node
                root.children[move] = MCTSNode(prior=prior_prob)
            
            # Add Dirichlet noise specific to this thread
            if legal_moves:
                noise = np.random.dirichlet([dirichlet_noise] * len(legal_moves))
                for i, move in enumerate(legal_moves):
                    # More weight to prior (0.95) for stability
                    root.children[move].prior = 0.95 * root.children[move].prior + 0.05 * noise[i]
            
            # Run simulations
            for _ in range(num_simulations):
                # Selection phase
                node = root
                search_path = [node]
                sim_board = type(board)(board.size)
                sim_board.board = np.copy(thread_board.board)
                sim_board.current_player = thread_board.current_player
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
                    moves = sim_board.get_focused_moves(distance=2)
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
            
            # Return results for this thread
            results = {}
            for move, child in root.children.items():
                if child.visit_count > 0:
                    results[move] = (child.visit_count, child.value_sum / child.visit_count)
            
            return results, thread_id
        
        # Launch threads
        start_time = time.time()
        results_by_thread = [None] * num_threads
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_thread = {
                executor.submit(run_mcts_thread, i, simulations_per_thread): i 
                for i in range(num_threads)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_thread):
                thread_results, thread_id = future.result()
                results_by_thread[thread_id] = thread_results
                
                # Check time limit - early stopping if needed
                if time.time() - start_time >= time_limit:
                    break
        
        # Combine results from all threads
        combined_visits = defaultdict(int)
        combined_values = defaultdict(float)
        
        for thread_results in results_by_thread:
            if thread_results:
                for move, (visits, value) in thread_results.items():
                    combined_visits[move] += visits
                    # Weighted average based on visit count
                    combined_values[move] += visits * value
        
        # Calculate final values
        final_values = {}
        for move in combined_visits:
            if combined_visits[move] > 0:
                final_values[move] = combined_values[move] / combined_visits[move]
        
        # Select move with highest visit count
        best_move = None
        most_visits = -1
        
        for move, visits in combined_visits.items():
            if visits > most_visits:
                most_visits = visits
                best_move = move
        
        # If no move found, use fallback
        if best_move is None and legal_moves:
            best_move = legal_moves[0]
        
        # Record in FiberTree
        if self.tree and best_move is not None:
            try:
                self.tree.start_path()
                for move in self._move_history:
                    self.tree.add_move(move)
                self.tree.add_move(Move(best_move))
                self.tree.record_outcome('draw')  # Temporary outcome
                self.tree.end_path()
            except Exception:
                pass
        
        return best_move

    def _fast_rollout(self, board):
        """Perform fast random rollout to estimate position value"""
        try:
            sim_board = type(board)(board.size)
            sim_board.board = np.copy(board.board)
            sim_board.current_player = board.current_player
            depth = 0
            rollout_depth = 6  # Reduced from 8 to 6 for faster simulation
            
            # Use light rollout strategy for speed
            while not sim_board.game_over and depth < rollout_depth:
                # First check for immediate winning moves
                immediate_move = sim_board.detect_immediate_threat()
                if immediate_move is not None:
                    row, col = immediate_move // sim_board.size, immediate_move % sim_board.size
                    sim_board.make_move(row, col)
                else:
                    # Get focused moves to reduce search space
                    moves = sim_board.get_focused_moves(distance=2)
                    if not moves:
                        moves = sim_board.get_legal_moves()
                    
                    if not moves:
                        break
                    
                    # Lightweight evaluation to improve rollouts
                    # Only score a sample of moves if there are many
                    if len(moves) > 6:
                        moves_to_score = random.sample(moves, 6)
                    else:
                        moves_to_score = moves
                        
                    scores = []
                    for move in moves_to_score:
                        row, col = move // sim_board.size, move % sim_board.size
                        # Use a simplified scoring function
                        score = sim_board.get_score_for_move(row, col, sim_board.current_player)
                        scores.append(max(1.0, score))  # Ensure positive scores
                    
                    # Select move with probability proportional to score
                    total = sum(scores)
                    # Select from scored moves
                    if total > 0 and random.random() < 0.8:  # 80% chance of using scored move
                        probs = [s / total for s in scores]
                        move_idx = np.random.choice(len(scores), p=probs)
                        selected_move = moves_to_score[move_idx]
                    else:
                        # 20% chance of random move
                        selected_move = random.choice(moves)
                    
                    row, col = selected_move // sim_board.size, selected_move % sim_board.size
                    sim_board.make_move(row, col)
                
                depth += 1
            
            # If game ended during rollout
            if sim_board.game_over:
                if sim_board.winner == 0:  # Draw
                    return 0.0
                return 1.0 if sim_board.winner == self.player_id else -1.0
            
            # Use simplified heuristic for unfinished games
            # Just count pattern occurrences weighted by importance
            eval_result = sim_board.evaluate_position()
            my_patterns = eval_result[self.player_id]["patterns"]
            opp_patterns = eval_result[3 - self.player_id]["patterns"]
            
            my_score = (
                my_patterns.get("open_four", 0) * 10 +
                my_patterns.get("four", 0) * 8 +
                my_patterns.get("open_three", 0) * 5 +
                my_patterns.get("three", 0) * 2 +
                my_patterns.get("open_two", 0) * 0.5
            )
            
            opp_score = (
                opp_patterns.get("open_four", 0) * 10 +
                opp_patterns.get("four", 0) * 8 +
                opp_patterns.get("open_three", 0) * 5 +
                opp_patterns.get("three", 0) * 2 +
                opp_patterns.get("open_two", 0) * 0.5
            )
            
            # Normalize to -1 to 1 range
            return np.tanh((my_score - opp_score) / 20.0)
            
        except Exception:
            return 0.0  # Neutral evaluation as fallback
    
    def learn_from_game(self, board, winner: int):
        """Update FiberTree from completed game"""
        if not board.game_over:
            return
        
        if not self.tree:
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
            if board.move_history:
                first_move = board.move_history[0]
                if first_move not in self.opening_stats:
                    self.opening_stats[first_move] = {"wins": 0, "games": 0}
                
                self.opening_stats[first_move]["games"] += 1
                if (self.player_id == 1 and winner == 1) or (self.player_id == 2 and winner == 2):
                    self.opening_stats[first_move]["wins"] += 1
                
                # Save opening statistics
                try:
                    with open('opening_stats.pkl', 'wb') as f:
                        pickle.dump(self.opening_stats, f)
                except Exception:
                    pass
                
            # Learn from symmetries if board is not too filled
            if len(board.move_history) < 20:
                self._learn_from_symmetries(board, move_sequence, outcome)
                
        except Exception:
            pass
    
    def _learn_from_symmetries(self, board, move_sequence, outcome: str):
        """Learn from symmetric board positions"""
        if not self.tree:
            return
            
        try:
            last_state = None
            for i, move in enumerate(move_sequence):
                # Only process a subset of positions for efficiency
                if i < 20 and i % 2 == 0:  # Process every other move up to 20
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
        except Exception:
            pass
    
    def save_knowledge(self, file_path: str, compress: bool = True):
        """Save FiberTree to file"""
        if not self.tree:
            return
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            self.tree.save(file_path, compress=compress)
            print(f"Knowledge saved to {file_path}")
            
            # Save opening statistics
            try:
                opening_stats_path = os.path.splitext(file_path)[0] + '_openings.pkl'
                with open(opening_stats_path, 'wb') as f:
                    pickle.dump(self.opening_stats, f)
            except Exception:
                pass
        except Exception:
            pass
    
    def load_knowledge(self, file_path: str):
        """Load FiberTree from file"""
        try:
            self.tree = load_tree(file_path)
            print(f"Knowledge loaded from {file_path}")
            
            # Try to load opening statistics
            try:
                opening_stats_path = os.path.splitext(file_path)[0] + '_openings.pkl'
                if os.path.exists(opening_stats_path):
                    with open(opening_stats_path, 'rb') as f:
                        self.opening_stats = pickle.load(f)
            except Exception:
                pass
        except Exception:
            pass
    
    def prune_knowledge(self, min_visits: int = 3) -> int:
        """Prune FiberTree, removing rarely visited paths"""
        if not self.tree:
            return 0
            
        try:
            pruned = self.tree.prune_tree(min_visits=min_visits)
            print(f"Pruned {pruned} paths from knowledge tree")
            return pruned
        except Exception:
            return 0