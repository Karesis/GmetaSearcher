"""
GomokuGame: Gomoku game management and AI training system
Provides complete game logic, player interaction, and enhanced training methods
"""

import os
import time
import random
import pickle
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gomoku_game.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GomokuGame")

# Import board and AI modules with graceful error handling
try:
    from board import GomokuBoard
    from ai import GomokuAI
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please make sure board.py and ai.py are in your PYTHONPATH")
    raise

# Import FiberTree with error handling
try:
    from fbtree import load_tree
except ImportError as e:
    logger.error(f"Failed to import FiberTree: {e}")
    
    # Define minimal function for compatibility
    def load_tree(*args, **kwargs): return None

class GomokuGame:
    """
    Complete Gomoku game supporting human players, AI players and advanced training methods
    """
    
    def __init__(self, board_size: int = 15, config: Optional[Dict] = None):
        """
        Initialize Gomoku game
        
        Args:
            board_size: Board size
            config: Game configuration options
        """
        self.board_size = board_size
        self.board = GomokuBoard(board_size)
        self.players = {
            1: None,  # Black
            2: None   # White
        }
        
        # Default configuration
        self.config = {
            'display_mode': 'text',  # 'text' or 'gui'
            'save_games': True,      # Whether to save game records
            'games_dir': 'saved_games',  # Game records save directory
            'thinking_time': 3.0,    # AI thinking time (seconds)
            'difficulty': 'normal',  # 'easy', 'normal', 'hard', 'master'
            'collect_stats': True,   # Collect game statistics
            'auto_save': True,       # Auto-save AI knowledge
            'knowledge_dir': 'ai_knowledge',  # AI knowledge directory
            'tournament_mode': False,  # Tournament mode
            'verbose': True,         # Verbose output
        }
        
        # Update custom configuration
        if config:
            self.config.update(config)
        
        # Create necessary directories with error handling
        self._create_directories()
        
        # Game statistics
        self.stats = {
            'games_played': 0,
            'black_wins': 0,
            'white_wins': 0,
            'draws': 0,
            'avg_game_length': 0,
            'total_time': 0,
            'opening_stats': defaultdict(lambda: {'played': 0, 'black_wins': 0, 'white_wins': 0}),
            'move_history': []  # Store game history to analyze hotspots
        }
        
        # Current game state
        self.current_game_id = None
        self.game_record = []
        self.game_start_time = None
        
        logger.info(f"GomokuGame initialized with board size {board_size}")
    
    def _create_directories(self):
        """Create necessary directories with error handling"""
        try:
            # Create directories using Path for platform independence
            directories = []
            
            if self.config['save_games']:
                directories.append(Path(self.config['games_dir']))
            
            if self.config['auto_save']:
                directories.append(Path(self.config['knowledge_dir']))
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                
            logger.info(f"Created necessary directories: {', '.join(str(d) for d in directories)}")
                
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            # Continue without crashing - the program will attempt to create these again when needed
    
    def set_player(self, player_id: int, ai: Optional[GomokuAI] = None, **kwargs):
        """
        Set player as human or AI
        
        Args:
            player_id: Player ID (1=black, 2=white)
            ai: AI for this player, None means human
            **kwargs: AI configuration options
        """
        if player_id not in [1, 2]:
            logger.error(f"Invalid player ID: {player_id}. Must be 1 (black) or 2 (white)")
            raise ValueError("Player ID must be 1 (black) or 2 (white)")
        
        # If AI provided but needs configuration
        if ai and kwargs:
            for key, value in kwargs.items():
                if hasattr(ai, key):
                    setattr(ai, key, value)
        
        # Set AI thinking time
        if ai and 'time_limit' not in kwargs:
            ai.time_limit = self.config['thinking_time']
        
        # Register player
        self.players[player_id] = ai
        if ai:
            ai.start_game(player_id)
        
        player_type = "AI" if ai else "Human"
        logger.info(f"Set player {player_id} as {player_type}")
    
    def create_ai(self, player_id: int, difficulty: str = None) -> GomokuAI:
        """
        Create AI player based on difficulty
        
        Args:
            player_id: Player ID (1=black, 2=white)
            difficulty: Difficulty level, None means use default config
            
        Returns:
            GomokuAI: Created AI instance
        """
        if difficulty is None:
            difficulty = self.config['difficulty']
        
        logger.info(f"Creating AI player {player_id} with difficulty {difficulty}")
        
        # Set parameters based on difficulty level
        if difficulty == 'easy':
            ai = GomokuAI(
                board_size=self.board_size,
                exploration_factor=2.0,  # More exploration, less exploitation
                max_depth=4,             # Shallow search depth
                use_patterns=True,
                use_opening_book=False,  # Don't use opening book
                time_limit=1.0           # Shorter thinking time
            )
        elif difficulty == 'normal':
            ai = GomokuAI(
                board_size=self.board_size,
                exploration_factor=1.2,
                max_depth=6,
                use_patterns=True,
                use_opening_book=True,
                time_limit=2.0
            )
        elif difficulty == 'hard':
            ai = GomokuAI(
                board_size=self.board_size,
                exploration_factor=1.0,
                max_depth=8,
                use_patterns=True,
                use_opening_book=True,
                time_limit=3.0
            )
        elif difficulty == 'master':
            ai = GomokuAI(
                board_size=self.board_size,
                exploration_factor=0.8,  # Less exploration, more exploitation
                max_depth=10,
                use_patterns=True,
                use_opening_book=True,
                time_limit=5.0,          # Longer thinking time
                adaptive_exploration=True
            )
        else:
            logger.error(f"Unknown difficulty level: {difficulty}")
            raise ValueError(f"Unknown difficulty level: {difficulty}")
        
        # Try to load existing knowledge
        knowledge_path = self._get_knowledge_path(player_id, difficulty)
        
        if os.path.exists(knowledge_path):
            try:
                ai.load_knowledge(knowledge_path)
                if self.config['verbose']:
                    print(f"Loaded AI knowledge: {knowledge_path}")
                logger.info(f"Loaded AI knowledge from {knowledge_path}")
            except Exception as e:
                logger.error(f"Error loading AI knowledge: {e}")
                print(f"Error loading AI knowledge: {e}")
        
        ai.start_game(player_id)
        return ai
    
    def _get_knowledge_path(self, player_id: int, difficulty: str) -> str:
        """Get standardized knowledge file path"""
        # Use Path for platform independence
        return str(Path(self.config['knowledge_dir']) / f"ai_knowledge_p{player_id}_{difficulty}.bin")
    
    def _get_game_record_path(self, game_id: str) -> str:
        """Get standardized game record file path"""
        return str(Path(self.config['games_dir']) / f"{game_id}.json")
    
    def play_game(self, max_moves: int = 225) -> int:
        """
        Play a complete game
        
        Args:
            max_moves: Maximum number of moves before forced end
            
        Returns:
            int: Game winner (0=draw, 1=black, 2=white, -1=aborted)
        """
        # Initialize game
        self.board.reset()
        move_count = 0
        self.game_record = []
        self.game_start_time = time.time()
        
        # Generate game ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_game_id = f"game_{timestamp}_{random.randint(1000, 9999)}"
        
        if self.config['verbose']:
            print(f"\n=== Starting new game ID: {self.current_game_id} ===")
            self.board.print_board()
        
        logger.info(f"Starting new game: {self.current_game_id}")
        
        # Game main loop
        while not self.board.game_over and move_count < max_moves:
            current_player = self.board.current_player
            move_start_time = time.time()
            
            # Get player move
            if self.players[current_player]:  # AI player
                try:
                    ai_move = self.players[current_player].select_move(self.board)
                    if ai_move >= 0:
                        row, col = ai_move // self.board.size, ai_move % self.board.size
                        
                        if self.config['verbose']:
                            thinking_time = time.time() - move_start_time
                            print(f"Player {current_player} (AI) thinking time: {thinking_time:.2f}s")
                            print(f"Player {current_player} (AI) plays at: ({row}, {col})")
                        
                        self.board.make_move(row, col)
                        
                        # Record move
                        self.game_record.append({
                            'player': current_player,
                            'move': (row, col),
                            'time': time.time() - move_start_time,
                            'move_number': move_count + 1
                        })
                    else:
                        logger.error("AI returned invalid move, game terminated")
                        print("AI returned invalid move, game terminated")
                        return -1
                except Exception as e:
                    logger.error(f"Error during AI move: {e}")
                    print(f"Error during AI move: {e}")
                    return -1
            else:  # Human player
                valid_move = False
                while not valid_move:
                    try:
                        move_input = input(f"Player {current_player} move (row,col): ")
                        # Check special commands
                        if move_input.lower() in ['quit', 'exit', 'q']:
                            logger.info("Player quit the game")
                            print("Player quit the game")
                            return -1
                        elif move_input.lower() in ['undo', 'u']:
                            # Implement undo functionality
                            if len(self.board.move_history) >= 2:  # Undo two steps (own and opponent's)
                                for _ in range(2):
                                    if self.game_record:
                                        self.game_record.pop()
                                # Rebuild board
                                self.board.reset()
                                for record in self.game_record:
                                    r, c = record['move']
                                    self.board.make_move(r, c)
                                move_count = len(self.game_record)
                                self.board.print_board()
                                logger.info("Undid last two moves")
                                continue
                            else:
                                print("Cannot undo any more steps")
                                continue
                        
                        # Normal move input
                        row, col = map(int, move_input.split(','))
                        thinking_time = time.time() - move_start_time
                        
                        valid_move = self.board.make_move(row, col)
                        if valid_move:
                            # Record move
                            self.game_record.append({
                                'player': current_player,
                                'move': (row, col),
                                'time': thinking_time,
                                'move_number': move_count + 1
                            })
                        else:
                            print("Invalid move. Please try again.")
                    except (ValueError, IndexError) as e:
                        print(f"Invalid input: {e}. Use format: row,col (e.g.: 7,7)")
            
            move_count += 1
            
            # Update and display board
            if self.config['verbose']:
                self.board.print_board()
        
        # Game over, display result
        game_duration = time.time() - self.game_start_time
        
        if self.config['verbose']:
            if self.board.winner == 0:
                print("Game ended in a draw")
            else:
                winner_name = "Black" if self.board.winner == 1 else "White"
                print(f"Game over. Winner: {winner_name}")
            
            print(f"Game duration: {game_duration:.2f} seconds, {move_count} moves")
        
        logger.info(f"Game {self.current_game_id} finished. Winner: {self.board.winner}, Moves: {move_count}")
        
        # Update AI knowledge
        for player_id, ai in self.players.items():
            if ai:
                try:
                    ai.learn_from_game(self.board, self.board.winner)
                    
                    # Auto-save AI knowledge
                    if self.config['auto_save']:
                        knowledge_path = self._get_knowledge_path(player_id, self.config['difficulty'])
                        ai.save_knowledge(knowledge_path)
                        
                        if self.config['verbose']:
                            print(f"Saved AI knowledge: {knowledge_path}")
                except Exception as e:
                    logger.error(f"Error updating AI knowledge: {e}")
        
        # Collect game statistics
        if self.config['collect_stats']:
            self._update_game_stats(move_count, game_duration)
        
        # Save game record
        if self.config['save_games']:
            self._save_game_record()
        
        return self.board.winner
    
    def _update_game_stats(self, move_count: int, game_duration: float):
        """Update game statistics"""
        try:
            self.stats['games_played'] += 1
            
            if self.board.winner == 1:
                self.stats['black_wins'] += 1
            elif self.board.winner == 2:
                self.stats['white_wins'] += 1
            else:
                self.stats['draws'] += 1
            
            # Update average game length
            prev_total = self.stats['avg_game_length'] * (self.stats['games_played'] - 1)
            self.stats['avg_game_length'] = (prev_total + move_count) / self.stats['games_played']
            
            # Update total time
            self.stats['total_time'] += game_duration
            
            # Update opening statistics
            if self.game_record:
                first_move = self.game_record[0]['move']
                move_key = first_move[0] * self.board.size + first_move[1]
                
                self.stats['opening_stats'][move_key]['played'] += 1
                if self.board.winner == 1:
                    self.stats['opening_stats'][move_key]['black_wins'] += 1
                elif self.board.winner == 2:
                    self.stats['opening_stats'][move_key]['white_wins'] += 1
            
            # Record move history for heatmap
            for record in self.game_record:
                r, c = record['move']
                pos = r * self.board.size + c
                self.stats['move_history'].append({
                    'position': pos,
                    'player': record['player'],
                    'game_id': self.current_game_id,
                    'result': self.board.winner
                })
        except Exception as e:
            logger.error(f"Error updating game statistics: {e}")
    
    def _save_game_record(self):
        """Save game record to file with error handling"""
        if not self.game_record:
            return
        
        try:
            # Prepare game data
            game_data = {
                'game_id': self.current_game_id,
                'date': datetime.datetime.now().isoformat(),
                'board_size': self.board.size,
                'moves': self.game_record,
                'result': self.board.winner,
                'black_player': 'AI' if self.players[1] else 'Human',
                'white_player': 'AI' if self.players[2] else 'Human',
                'duration': time.time() - self.game_start_time
            }
            
            # Save as JSON file using standardized path
            file_path = self._get_game_record_path(self.current_game_id)
            
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(game_data, f, indent=2)
            
            if self.config['verbose']:
                print(f"Game record saved: {file_path}")
                
            logger.info(f"Game record saved: {file_path}")
                
        except Exception as e:
            logger.error(f"Error saving game record: {e}")
    
    def self_play(self, num_games: int = 10, parallel: bool = False, cores: int = None) -> Dict[str, Any]:
        """
        Run self-play games to train AI
        
        Args:
            num_games: Number of games
            parallel: Whether to run in parallel (multi-core)
            cores: Number of CPU cores to use, None means use all available
            
        Returns:
            Dict: Game statistics
        """
        if not (self.players[1] and self.players[2]):
            logger.error("Self-play requires two AI players")
            raise ValueError("Self-play requires two AI players")
        
        stats = {
            "black_wins": 0,
            "white_wins": 0,
            "draws": 0,
            "games": []
        }
        
        # Use parallel mode
        if parallel and num_games > 1:
            return self._parallel_self_play(num_games, cores)
        
        # Serial mode
        logger.info(f"Starting serial self-play for {num_games} games")
        
        for i in range(num_games):
            if self.config['verbose']:
                print(f"\nStarting self-play game {i+1}/{num_games}")
            
            # Alternate first player for every other game
            if i % 2 == 1 and not self.config['tournament_mode']:
                # Swap AI roles
                ai1, ai2 = self.players[1], self.players[2]
                self.set_player(1, ai2)
                self.set_player(2, ai1)
            
            # Play one game
            winner = self.play_game()
            
            # Record result
            if winner >= 0:  # Not aborted
                game_info = {
                    'game_id': self.current_game_id,
                    'winner': winner,
                    'moves': len(self.game_record)
                }
                stats['games'].append(game_info)
                
                if winner == 1:
                    stats["black_wins"] += 1
                elif winner == 2:
                    stats["white_wins"] += 1
                else:
                    stats["draws"] += 1
                    
                if self.config['verbose']:
                    print(f"Current stats: Black wins: {stats['black_wins']}, "
                         f"White wins: {stats['white_wins']}, Draws: {stats['draws']}")
            
            # Restore original AI role assignment
            if i % 2 == 1 and not self.config['tournament_mode']:
                ai1, ai2 = self.players[1], self.players[2]
                self.set_player(1, ai2)
                self.set_player(2, ai1)
            
            # Every 10 games save AI knowledge
            if (i + 1) % 10 == 0 and self.config['auto_save']:
                self._save_ai_knowledge()
                
                # Prune knowledge tree periodically to avoid excessive growth
                if (i + 1) % self.config.get('prune_frequency', 50) == 0:
                    self._prune_ai_knowledge()
        
        # Final knowledge save
        if self.config['auto_save']:
            self._save_ai_knowledge()
        
        return stats
    
    def _save_ai_knowledge(self):
        """Save AI knowledge for all AI players"""
        for player_id, ai in self.players.items():
            if ai:
                try:
                    difficulty = self.config['difficulty']
                    knowledge_path = self._get_knowledge_path(player_id, difficulty)
                    ai.save_knowledge(knowledge_path)
                    logger.info(f"Saved AI {player_id} knowledge to {knowledge_path}")
                except Exception as e:
                    logger.error(f"Error saving AI {player_id} knowledge: {e}")
    
    def _prune_ai_knowledge(self):
        """Prune AI knowledge trees to manage size"""
        for player_id, ai in self.players.items():
            if ai:
                try:
                    pruned = ai.prune_knowledge(min_visits=2)
                    if self.config['verbose']:
                        print(f"Pruned AI {player_id}'s knowledge tree, removed {pruned} paths")
                    logger.info(f"Pruned AI {player_id}'s knowledge tree, removed {pruned} paths")
                except Exception as e:
                    logger.error(f"Error pruning AI {player_id} knowledge: {e}")
    
    def _parallel_self_play(self, num_games: int, cores: Optional[int] = None) -> Dict[str, Any]:
        """
        Run self-play in parallel using multiple processes
        
        Args:
            num_games: Number of games
            cores: Number of CPU cores, None means use all available
            
        Returns:
            Dict: Game statistics
        """
        # Determine core count
        if cores is None:
            cores = mp.cpu_count()
        
        cores = min(cores, mp.cpu_count(), num_games)
        
        if self.config['verbose']:
            print(f"Using {cores} CPU cores to run {num_games} self-play games in parallel...")
        
        logger.info(f"Starting parallel self-play with {cores} cores for {num_games} games")
        
        # Save current AI knowledge for worker processes to load
        temp_knowledge_files = []
        for player_id, ai in self.players.items():
            if ai:
                try:
                    # Use temporary file with random suffix to avoid conflicts
                    temp_file = f"temp_knowledge_p{player_id}_{random.randint(1000, 9999)}.bin"
                    ai.save_knowledge(temp_file)
                    temp_knowledge_files.append((player_id, temp_file))
                except Exception as e:
                    logger.error(f"Error saving temporary knowledge for player {player_id}: {e}")
        
        # Prepare worker process parameters
        games_per_process = [num_games // cores + (1 if i < num_games % cores else 0) 
                           for i in range(cores)]
        
        args_list = []
        for i, games in enumerate(games_per_process):
            # Use different game save directory for each process to avoid conflicts
            process_games_dir = str(Path(self.config['games_dir']) / f"process_{i}")
            
            # Create process config
            process_config = self.config.copy()
            process_config['games_dir'] = process_games_dir
            process_config['verbose'] = False  # Disable verbose output to avoid confusion
            
            args_list.append({
                'process_id': i,
                'num_games': games,
                'board_size': self.board_size,
                'config': process_config,
                'temp_knowledge_files': temp_knowledge_files,
                'difficulty': self.config['difficulty'],
                'tournament_mode': self.config['tournament_mode']
            })
        
        # Use process pool to run games in parallel
        try:
            with mp.Pool(cores) as pool:
                results = pool.map(self._self_play_worker, args_list)
            
            # Combine results
            stats = {
                "black_wins": 0,
                "white_wins": 0,
                "draws": 0,
                "games": []
            }
            
            for result in results:
                stats["black_wins"] += result["black_wins"]
                stats["white_wins"] += result["white_wins"]
                stats["draws"] += result["draws"]
                stats["games"].extend(result["games"])
                
        except Exception as e:
            logger.error(f"Error in parallel self-play: {e}")
            stats = {"error": str(e)}
        finally:
            # Clean up temporary files
            for _, file_path in temp_knowledge_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.error(f"Error removing temporary file {file_path}: {e}")
        
        # Merge all process game records
        if self.config['save_games']:
            self._merge_process_game_records(cores)
        
        # Merge all process AI knowledge
        if self.config['auto_save']:
            self._merge_process_ai_knowledge(cores)
        
        if self.config['verbose']:
            print(f"Parallel self-play complete: Black wins: {stats['black_wins']}, "
                 f"White wins: {stats['white_wins']}, Draws: {stats['draws']}")
        
        logger.info(f"Completed parallel self-play: {len(stats['games'])} games")
        
        return stats
    
    @staticmethod
    def _self_play_worker(args):
        """
        Worker function for parallel self-play
        
        Args:
            args: Dictionary with game parameters
            
        Returns:
            Dict: Game statistics
        """
        process_id = args['process_id']
        num_games = args['num_games']
        board_size = args['board_size']
        config = args['config']
        temp_knowledge_files = args['temp_knowledge_files']
        difficulty = args['difficulty']
        tournament_mode = args['tournament_mode']
        
        # Configure process-specific logger
        process_logger = logging.getLogger(f"GomokuGame_Process{process_id}")
        
        try:
            # Create new game instance
            game = GomokuGame(board_size=board_size, config=config)
            
            # Create and load AI
            ai1 = game.create_ai(1, difficulty)
            ai2 = game.create_ai(2, difficulty)
            
            # Load temporary knowledge files
            for player_id, file_path in temp_knowledge_files:
                if os.path.exists(file_path):
                    if player_id == 1:
                        ai1.load_knowledge(file_path)
                    elif player_id == 2:
                        ai2.load_knowledge(file_path)
            
            # Set AI players
            game.set_player(1, ai1)
            game.set_player(2, ai2)
            
            # Run self-play
            stats = {
                "black_wins": 0,
                "white_wins": 0,
                "draws": 0,
                "games": []
            }
            
            for i in range(num_games):
                # Alternate first player for every other game
                if i % 2 == 1 and not tournament_mode:
                    # Swap AI roles
                    game.set_player(1, ai2)
                    game.set_player(2, ai1)
                else:
                    game.set_player(1, ai1)
                    game.set_player(2, ai2)
                
                # Play one game
                winner = game.play_game()
                
                # Record result
                if winner >= 0:  # Not aborted
                    game_info = {
                        'game_id': game.current_game_id,
                        'winner': winner,
                        'moves': len(game.game_record),
                        'process_id': process_id
                    }
                    stats['games'].append(game_info)
                    
                    if winner == 1:
                        stats["black_wins"] += 1
                    elif winner == 2:
                        stats["white_wins"] += 1
                    else:
                        stats["draws"] += 1
                
                # Periodically save AI knowledge
                if (i + 1) % 10 == 0 and config['auto_save']:
                    # Ensure process knowledge directory exists
                    knowledge_dir = Path(config['knowledge_dir']) / f"process_{process_id}"
                    knowledge_dir.mkdir(parents=True, exist_ok=True)
                    
                    for player_id, ai in [(1, ai1), (2, ai2)]:
                        knowledge_path = str(knowledge_dir / f"ai_knowledge_p{player_id}_{difficulty}.bin")
                        ai.save_knowledge(knowledge_path)
            
            # Final knowledge save
            if config['auto_save']:
                # Ensure process knowledge directory exists
                knowledge_dir = Path(config['knowledge_dir']) / f"process_{process_id}"
                knowledge_dir.mkdir(parents=True, exist_ok=True)
                
                for player_id, ai in [(1, ai1), (2, ai2)]:
                    knowledge_path = str(knowledge_dir / f"ai_knowledge_p{player_id}_{difficulty}.bin")
                    ai.save_knowledge(knowledge_path)
            
            process_logger.info(f"Process {process_id} completed {num_games} games")
            return stats
            
        except Exception as e:
            process_logger.error(f"Error in process {process_id}: {e}")
            # Return partial results if available, otherwise empty stats
            if 'stats' in locals():
                return stats
            else:
                return {"black_wins": 0, "white_wins": 0, "draws": 0, "games": []}
    
    def _merge_process_game_records(self, cores: int):
        """Merge game records from all processes"""
        try:
            for i in range(cores):
                process_games_dir = Path(self.config['games_dir']) / f"process_{i}"
                if process_games_dir.exists():
                    for filename in process_games_dir.glob('*.json'):
                        src = filename
                        dst = Path(self.config['games_dir']) / filename.name
                        # Use unique filename to avoid conflicts
                        if dst.exists():
                            base, ext = dst.stem, dst.suffix
                            dst = Path(self.config['games_dir']) / f"{base}_{random.randint(1000, 9999)}{ext}"
                        
                        # Move file
                        src.rename(dst)
                        
                    # Try to remove process directory if empty
                    try:
                        if not any(process_games_dir.iterdir()):
                            process_games_dir.rmdir()
                    except:
                        pass
                        
            logger.info(f"Merged game records from {cores} processes")
                
        except Exception as e:
            logger.error(f"Error merging process game records: {e}")
    
    def _merge_process_ai_knowledge(self, cores: int):
        """Merge AI knowledge from all processes"""
        try:
            for player_id in [1, 2]:
                if not self.players[player_id]:
                    continue
                    
                # Load each process's knowledge and merge
                for process_id in range(cores):
                    process_knowledge_dir = Path(self.config['knowledge_dir']) / f"process_{process_id}"
                    knowledge_path = process_knowledge_dir / f"ai_knowledge_p{player_id}_{self.config['difficulty']}.bin"
                    
                    if knowledge_path.exists():
                        try:
                            # Load this process's knowledge
                            process_tree = load_tree(str(knowledge_path))
                            
                            # Merge to main AI
                            if self.players[player_id] and self.players[player_id].tree and process_tree:
                                self.players[player_id].tree.merge(process_tree)
                                
                                # Save merged knowledge
                                merged_path = self._get_knowledge_path(player_id, self.config['difficulty'])
                                self.players[player_id].save_knowledge(merged_path)
                                
                                logger.info(f"Merged knowledge for player {player_id} from process {process_id}")
                        except Exception as e:
                            logger.error(f"Error merging knowledge from process {process_id} for player {player_id}: {e}")
            
            # Prune merged knowledge
            self._prune_ai_knowledge()
            
        except Exception as e:
            logger.error(f"Error in knowledge merge: {e}")
    
    def run_tournament(self, 
                     participants: List[Dict], 
                     games_per_match: int = 10, 
                     parallel: bool = False) -> Dict:
        """
        Run tournament between AIs
        
        Args:
            participants: List of participants, each a dict with 'id' and 'ai' keys
            games_per_match: Number of games per match pair
            parallel: Whether to use parallel computation
            
        Returns:
            Dict: Tournament results
        """
        if len(participants) < 2:
            logger.error("Tournament requires at least two participants")
            raise ValueError("Tournament requires at least two participants")
        
        # Configure tournament mode
        tournament_config = self.config.copy()
        tournament_config['tournament_mode'] = True
        tournament_config['verbose'] = False
        self.config.update(tournament_config)
        
        # Tournament results
        results = {
            'matches': [],
            'standings': {p['id']: {'wins': 0, 'losses': 0, 'draws': 0, 'points': 0, 'played': 0}
                        for p in participants}
        }
        
        # Print tournament info
        print(f"\n=== Starting Tournament ({len(participants)} participants, {games_per_match} games per match) ===")
        for p in participants:
            print(f"Participant: {p['id']}")
        
        logger.info(f"Starting tournament with {len(participants)} participants, {games_per_match} games per match")
        
        # All possible match combinations
        matches = []
        for i in range(len(participants)):
            for j in range(i+1, len(participants)):
                matches.append((participants[i], participants[j]))
        
        # Run matches
        for match_idx, (p1, p2) in enumerate(matches):
            print(f"\nMatch {match_idx+1}/{len(matches)}: {p1['id']} vs {p2['id']}")
            logger.info(f"Starting match: {p1['id']} vs {p2['id']}")
            
            # Set AI players with error handling
            try:
                self.set_player(1, p1['ai'])
                self.set_player(2, p2['ai'])
            except Exception as e:
                logger.error(f"Error setting players for match {p1['id']} vs {p2['id']}: {e}")
                print(f"Error: {e}")
                continue
            
            # Run match
            try:
                match_stats = self.self_play(num_games=games_per_match, parallel=parallel)
            except Exception as e:
                logger.error(f"Error running match {p1['id']} vs {p2['id']}: {e}")
                print(f"Error: {e}")
                continue
            
            # Record match result
            match_result = {
                'player1': p1['id'],
                'player2': p2['id'],
                'p1_wins': match_stats['black_wins'],
                'p2_wins': match_stats['white_wins'],
                'draws': match_stats['draws'],
                'games': match_stats['games']
            }
            results['matches'].append(match_result)
            
            # Update standings
            self._update_tournament_standings(results['standings'], p1['id'], p2['id'], match_stats)
            
            # Print match result
            print(f"Result: {p1['id']} {match_stats['black_wins']} - {match_stats['white_wins']} {p2['id']} (Draws: {match_stats['draws']})")
        
        # Calculate final standings
        standings = sorted(
            results['standings'].items(), 
            key=lambda x: (x[1]['points'], x[1]['wins']), 
            reverse=True
        )
        
        # Print tournament results
        print("\n=== Tournament Results ===")
        print(f"{'Player':15} {'Points':5} {'Wins':5} {'Losses':5} {'Draws':5} {'Games':5}")
        print("-" * 45)
        
        for i, (participant_id, stats) in enumerate(standings):
            print(f"{i+1}. {participant_id:12} {stats['points']:5} {stats['wins']:5} {stats['losses']:5} {stats['draws']:5} {stats['played']:5}")
        
        # Save tournament results
        self._save_tournament_results(results)
        
        # Restore original config
        self.config['tournament_mode'] = False
        self.config['verbose'] = True
        
        logger.info("Tournament completed")
        
        return results
    
    def _update_tournament_standings(self, standings, p1_id, p2_id, match_stats):
        """Update tournament standings with match results"""
        try:
            # Update played count
            games_count = match_stats['black_wins'] + match_stats['white_wins'] + match_stats['draws']
            standings[p1_id]['played'] += games_count
            standings[p2_id]['played'] += games_count
            
            # Update wins/losses/draws
            standings[p1_id]['wins'] += match_stats['black_wins']
            standings[p1_id]['losses'] += match_stats['white_wins']
            standings[p1_id]['draws'] += match_stats['draws']
            
            standings[p2_id]['wins'] += match_stats['white_wins']
            standings[p2_id]['losses'] += match_stats['black_wins']
            standings[p2_id]['draws'] += match_stats['draws']
            
            # Calculate points (win=3, draw=1)
            standings[p1_id]['points'] = (
                standings[p1_id]['wins'] * 3 + 
                standings[p1_id]['draws']
            )
            
            standings[p2_id]['points'] = (
                standings[p2_id]['wins'] * 3 + 
                standings[p2_id]['draws']
            )
        except Exception as e:
            logger.error(f"Error updating tournament standings: {e}")
    
    def _save_tournament_results(self, results):
        """Save tournament results to file"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            tournament_file = Path(self.config['games_dir']) / f"tournament_{timestamp}.json"
            
            # Ensure directory exists
            tournament_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(tournament_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nTournament results saved: {tournament_file}")
            logger.info(f"Tournament results saved: {tournament_file}")
            
        except Exception as e:
            logger.error(f"Error saving tournament results: {e}")
    
    def analyze_games(self, num_games: int = None) -> Dict[str, Any]:
        """
        Analyze saved game records
        
        Args:
            num_games: Number of games to analyze, None means all games
            
        Returns:
            Dict: Analysis results
        """
        games_dir = Path(self.config['games_dir'])
        if not games_dir.exists():
            logger.warning(f"Games directory does not exist: {games_dir}")
            return {"error": "Games directory does not exist"}
        
        # Collect game files
        game_files = list(games_dir.glob('game_*.json'))
        
        if not game_files:
            logger.warning("No game records found")
            return {"error": "No game records found"}
        
        if num_games and num_games < len(game_files):
            # Randomly select game files
            game_files = random.sample(game_files, num_games)
        
        print(f"Analyzing {len(game_files)} game records...")
        logger.info(f"Analyzing {len(game_files)} game records")
        
        # Analysis result
        analysis = {
            'total_games': len(game_files),
            'black_wins': 0,
            'white_wins': 0,
            'draws': 0,
            'avg_game_length': 0,
            'opening_stats': defaultdict(lambda: {'count': 0, 'black_wins': 0, 'white_wins': 0}),
            'position_heatmap': np.zeros((self.board_size, self.board_size)),
            'win_rate_by_position': {},
            'move_timing': {'early': [], 'mid': [], 'late': []},
            'common_patterns': []
        }
        
        # Analyze each game
        total_moves = 0
        move_data = []
        
        for file_path in game_files:
            try:
                with open(file_path, 'r') as f:
                    game_data = json.load(f)
                
                # Basic statistics
                result = game_data.get('result', 0)
                if result == 1:
                    analysis['black_wins'] += 1
                elif result == 2:
                    analysis['white_wins'] += 1
                else:
                    analysis['draws'] += 1
                
                moves = game_data.get('moves', [])
                total_moves += len(moves)
                
                # Record opening statistics
                if moves:
                    first_move = tuple(moves[0]['move'])
                    first_move_key = first_move[0] * self.board_size + first_move[1]
                    analysis['opening_stats'][first_move_key]['count'] += 1
                    
                    if result == 1:
                        analysis['opening_stats'][first_move_key]['black_wins'] += 1
                    elif result == 2:
                        analysis['opening_stats'][first_move_key]['white_wins'] += 1
                
                # Update heatmap
                for move in moves:
                    r, c = move['move']
                    analysis['position_heatmap'][r, c] += 1
                    
                    # Record move position and result
                    move_data.append({
                        'position': (r, c),
                        'player': move['player'],
                        'result': result,
                        'move_number': move.get('move_number', 0)
                    })
                    
                    # Record timing statistics
                    move_time = move.get('time', 0)
                    move_num = move.get('move_number', 0)
                    game_phase = 'early' if move_num <= 10 else ('mid' if move_num <= 30 else 'late')
                    analysis['move_timing'][game_phase].append(move_time)
                    
            except Exception as e:
                logger.error(f"Error analyzing game {file_path.name}: {e}")
        
        # Calculate average game length
        if analysis['total_games'] > 0:
            analysis['avg_game_length'] = total_moves / analysis['total_games']
        
        # Analyze win rate by position
        self._analyze_position_win_rates(move_data, analysis)
        
        # Find common opening patterns
        self._find_common_patterns(analysis)
        
        # Visualize heatmap
        self._visualize_heatmap(analysis['position_heatmap'])
        
        logger.info(f"Analysis complete: {analysis['black_wins']} black wins, {analysis['white_wins']} white wins")
        
        return analysis
    
    def _analyze_position_win_rates(self, move_data, analysis):
        """Analyze win rates by board position"""
        try:
            # Track results by position
            position_results = defaultdict(lambda: {'black_wins': 0, 'white_wins': 0, 'total': 0})
            
            for move in move_data:
                pos = (move['position'][0], move['position'][1])
                position_results[pos]['total'] += 1
                
                if move['result'] == 1:
                    position_results[pos]['black_wins'] += 1
                elif move['result'] == 2:
                    position_results[pos]['white_wins'] += 1
            
            # Calculate win rate for each position
            for pos, stats in position_results.items():
                if stats['total'] >= 5:  # Only consider positions with enough samples
                    black_win_rate = stats['black_wins'] / stats['total'] if stats['total'] > 0 else 0
                    white_win_rate = stats['white_wins'] / stats['total'] if stats['total'] > 0 else 0
                    analysis['win_rate_by_position'][pos] = {
                        'black_win_rate': black_win_rate,
                        'white_win_rate': white_win_rate,
                        'total_games': stats['total']
                    }
        except Exception as e:
            logger.error(f"Error analyzing position win rates: {e}")
    
    def _find_common_patterns(self, analysis):
        """Find common opening patterns"""
        try:
            # Sort openings by frequency
            top_openings = sorted(
                analysis['opening_stats'].items(), 
                key=lambda x: x[1]['count'], 
                reverse=True
            )[:5]
            
            # Format for the analysis result
            analysis['common_patterns'] = [
                {
                    'position': (key // self.board_size, key % self.board_size),
                    'count': stats['count'],
                    'black_win_rate': stats['black_wins'] / stats['count'] if stats['count'] > 0 else 0,
                    'white_win_rate': stats['white_wins'] / stats['count'] if stats['count'] > 0 else 0
                }
                for key, stats in top_openings
            ]
        except Exception as e:
            logger.error(f"Error finding common patterns: {e}")
    
    def _visualize_heatmap(self, heatmap: np.ndarray):
        """Visualize board heatmap"""
        try:
            plt.figure(figsize=(10, 10))
            
            # Use board color scheme
            plt.imshow(heatmap, cmap='YlOrRd')
            
            # Add grid
            plt.grid(color='black', linestyle='-', linewidth=0.5)
            
            # Set ticks
            plt.xticks(np.arange(self.board_size))
            plt.yticks(np.arange(self.board_size))
            
            # Add title and labels
            plt.title('Move Heatmap Analysis')
            plt.colorbar(label='Move Frequency')
            
            # Save image
            heatmap_path = 'move_heatmap.png'
            plt.savefig(heatmap_path)
            
            if self.config['verbose']:
                print(f"Heatmap saved as '{heatmap_path}'")
                
            logger.info(f"Heatmap saved to {heatmap_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing heatmap: {e}")
    
    def train_curriculum(self, 
                       phases: List[Dict], 
                       save_interval: int = 50,
                       base_dir: str = "curriculum_training") -> Dict:
        """
        Execute curriculum learning training, gradually increasing difficulty
        
        Args:
            phases: List of training phases, each a dict with parameters
            save_interval: Auto-save interval (games)
            base_dir: Training data base directory
            
        Returns:
            Dict: Training results
        """
        # Create base directory
        base_path = Path(base_dir)
        try:
            base_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating training directory: {e}")
            raise
        
        # Training results
        training_results = {
            'phases': [],
            'overall_stats': {
                'total_games': 0,
                'training_time': 0
            }
        }
        
        # Implement curriculum learning
        training_start_time = time.time()
        
        for phase_idx, phase in enumerate(phases):
            phase_name = phase.get('name', f"Phase {phase_idx+1}")
            games_count = phase.get('games', 100)
            difficulty = phase.get('difficulty', 'normal')
            parallel = phase.get('parallel', False)
            
            print(f"\n=== Starting Training Phase: {phase_name} ===")
            print(f"Games: {games_count}, Difficulty: {difficulty}, Parallel: {parallel}")
            
            logger.info(f"Starting training phase {phase_name}: {games_count} games, {difficulty} difficulty")
            
            # Set phase-specific configuration
            self._configure_training_phase(phase, phase_idx, base_path)
            
            # Create or update AIs
            ai1, ai2 = self._setup_training_ais(phase, difficulty)
            
            # Set players
            self.set_player(1, ai1)
            self.set_player(2, ai2)
            
            # Run self-play training
            phase_start_time = time.time()
            try:
                phase_stats = self.self_play(num_games=games_count, parallel=parallel)
                phase_duration = time.time() - phase_start_time
            except Exception as e:
                logger.error(f"Error during phase {phase_name}: {e}")
                phase_stats = {"black_wins": 0, "white_wins": 0, "draws": 0, "games": []}
                phase_duration = time.time() - phase_start_time
            
            # Save phase AI knowledge
            self._save_phase_knowledge(phase_idx, ai1, ai2)
            
            # Record phase result
            phase_result = self._record_phase_results(
                phase_name, games_count, difficulty, 
                phase_stats, phase_duration, phase_idx
            )
            
            training_results['phases'].append(phase_result)
            training_results['overall_stats']['total_games'] += games_count
            
            # Print phase results
            self._print_phase_results(phase_name, phase_stats, games_count, phase_duration)
            
            # Analyze phase data
            self._analyze_phase_data(phase_idx, games_count, phase_result)
        
        # Calculate total training time
        training_results['overall_stats']['training_time'] = time.time() - training_start_time
        
        # Save total training results
        training_results_path = base_path / 'training_results.json'
        self._save_training_results(training_results, training_results_path)
        
        print(f"\n=== Curriculum Learning Training Complete ===")
        print(f"Total games: {training_results['overall_stats']['total_games']}")
        print(f"Total time: {training_results['overall_stats']['training_time']:.2f} seconds")
        print(f"Training results saved: {training_results_path}")
        
        logger.info(f"Curriculum training completed: {training_results['overall_stats']['total_games']} games")
        
        return training_results
    
    def _configure_training_phase(self, phase, phase_idx, base_path):
        """Configure phase-specific settings"""
        try:
            # Set phase-specific config
            phase_config = self.config.copy()
            phase_config.update(phase.get('config', {}))
            phase_config['difficulty'] = phase.get('difficulty', 'normal')
            
            # Create phase directory
            phase_dir = base_path / f"phase_{phase_idx+1}"
            phase_dir.mkdir(parents=True, exist_ok=True)
            
            # Update game and knowledge directories
            phase_config['games_dir'] = str(phase_dir / 'games')
            phase_config['knowledge_dir'] = str(phase_dir / 'knowledge')
            
            # Create subdirectories
            Path(phase_config['games_dir']).mkdir(parents=True, exist_ok=True)
            Path(phase_config['knowledge_dir']).mkdir(parents=True, exist_ok=True)
            
            # Update game configuration
            self.config.update(phase_config)
            
            logger.info(f"Configured phase {phase_idx+1} with difficulty {phase_config['difficulty']}")
            
        except Exception as e:
            logger.error(f"Error configuring training phase {phase_idx+1}: {e}")
            raise
    
    def _setup_training_ais(self, phase, difficulty):
        """Create or update AIs for training phase"""
        ai1 = None
        ai2 = None
        
        try:
            # If AI1 knowledge path specified, load it
            if 'ai1_knowledge' in phase:
                ai1 = GomokuAI(board_size=self.board_size)
                ai1.load_knowledge(phase['ai1_knowledge'])
            else:
                # Use previous phase AI (if any)
                if self.players[1]:
                    ai1 = self.players[1]
                else:
                    ai1 = self.create_ai(1, difficulty)
            
            # Set AI configuration
            if 'ai1_config' in phase:
                for key, value in phase['ai1_config'].items():
                    if hasattr(ai1, key):
                        setattr(ai1, key, value)
            
            # Similarly handle AI2
            if 'ai2_knowledge' in phase:
                ai2 = GomokuAI(board_size=self.board_size)
                ai2.load_knowledge(phase['ai2_knowledge'])
            else:
                if self.players[2]:
                    ai2 = self.players[2]
                else:
                    ai2 = self.create_ai(2, difficulty)
            
            if 'ai2_config' in phase:
                for key, value in phase['ai2_config'].items():
                    if hasattr(ai2, key):
                        setattr(ai2, key, value)
                        
            return ai1, ai2
            
        except Exception as e:
            logger.error(f"Error setting up training AIs: {e}")
            # Create new AIs as fallback
            if not ai1:
                ai1 = self.create_ai(1, difficulty)
            if not ai2:
                ai2 = self.create_ai(2, difficulty)
            return ai1, ai2
    
    def _save_phase_knowledge(self, phase_idx, ai1, ai2):
        """Save phase AI knowledge"""
        try:
            knowledge_dir = Path(self.config['knowledge_dir'])
            ai1_path = knowledge_dir / f"ai1_phase_{phase_idx+1}.bin"
            ai2_path = knowledge_dir / f"ai2_phase_{phase_idx+1}.bin"
            
            ai1.save_knowledge(str(ai1_path))
            ai2.save_knowledge(str(ai2_path))
            
            logger.info(f"Saved phase {phase_idx+1} AI knowledge")
            
        except Exception as e:
            logger.error(f"Error saving phase knowledge: {e}")
    
    def _record_phase_results(self, phase_name, games_count, difficulty, 
                            phase_stats, phase_duration, phase_idx):
        """Record phase training results"""
        try:
            return {
                'name': phase_name,
                'games': games_count,
                'difficulty': difficulty,
                'black_wins': phase_stats['black_wins'],
                'white_wins': phase_stats['white_wins'],
                'draws': phase_stats['draws'],
                'win_rate_black': phase_stats['black_wins'] / games_count if games_count > 0 else 0,
                'win_rate_white': phase_stats['white_wins'] / games_count if games_count > 0 else 0,
                'duration': phase_duration,
                'ai1_knowledge_path': str(Path(self.config['knowledge_dir']) / f"ai1_phase_{phase_idx+1}.bin"),
                'ai2_knowledge_path': str(Path(self.config['knowledge_dir']) / f"ai2_phase_{phase_idx+1}.bin")
            }
        except Exception as e:
            logger.error(f"Error recording phase results: {e}")
            # Return minimal results
            return {
                'name': phase_name,
                'error': str(e)
            }
    
    def _print_phase_results(self, phase_name, phase_stats, games_count, phase_duration):
        """Print phase results"""
        try:
            black_win_rate = phase_stats['black_wins'] / games_count if games_count > 0 else 0
            white_win_rate = phase_stats['white_wins'] / games_count if games_count > 0 else 0
            draw_rate = phase_stats['draws'] / games_count if games_count > 0 else 0
            
            print(f"\nPhase {phase_name} complete:")
            print(f"Black wins: {phase_stats['black_wins']}/{games_count} ({black_win_rate:.2%})")
            print(f"White wins: {phase_stats['white_wins']}/{games_count} ({white_win_rate:.2%})")
            print(f"Draws: {phase_stats['draws']}/{games_count} ({draw_rate:.2%})")
            print(f"Time: {phase_duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error printing phase results: {e}")
    
    def _analyze_phase_data(self, phase_idx, games_count, phase_result):
        """Analyze phase data and save info"""
        try:
            # Analyze up to 100 games
            phase_analysis = self.analyze_games(min(100, games_count))
            
            # Save phase information
            phase_info = {
                'phase_result': phase_result,
                'phase_analysis': phase_analysis
            }
            
            phase_info_path = Path(self.config['knowledge_dir']).parent / 'phase_info.json'
            
            # Convert numpy arrays for JSON serialization
            self._save_json_with_numpy(phase_info, phase_info_path)
            
            logger.info(f"Saved phase {phase_idx+1} analysis")
            
        except Exception as e:
            logger.error(f"Error analyzing phase data: {e}")
    
    def _save_json_with_numpy(self, data, file_path):
        """Save JSON with numpy array conversion"""
        try:
            # Convert numpy arrays for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy(item) for item in obj)
                elif isinstance(obj, set):
                    return {convert_numpy(item) for item in obj}
                else:
                    return obj
            
            with open(file_path, 'w') as f:
                json.dump(convert_numpy(data), f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving JSON with numpy data: {e}")
    
    def _save_training_results(self, training_results, file_path):
        """Save overall training results"""
        try:
            with open(file_path, 'w') as f:
                # Convert any numpy types
                self._save_json_with_numpy(training_results, file_path)
                
            logger.info(f"Saved training results to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")


# Example usage
if __name__ == "__main__":
    # Create game
    game = GomokuGame(board_size=15)
    
    # Create two AIs
    ai_black = game.create_ai(1, difficulty='normal')
    ai_white = game.create_ai(2, difficulty='normal')
    
    # Set players
    game.set_player(1, ai_black)  # Black AI
    game.set_player(2, ai_white)  # White AI
    
    # Run 10 self-play games
    stats = game.self_play(num_games=10, parallel=True)
    print(f"Self-play results: {stats}")
    
    # Analyze games
    analysis = game.analyze_games()
    print(f"Game analysis completed")
    
    # Create a human vs AI game
    game.set_player(1, ai_black)  # Black AI
    game.set_player(2, None)      # White Human
    
    print("\nHuman vs AI game starting (you play White):")
    game.play_game()