"""
GomokuGame: Optimized Gomoku game management with parallel self-play
"""

import os
import time
import random
import datetime
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from fbtree import load_tree
# Import board and AI modules
from board import GomokuBoard
from ai import GomokuAI

class GomokuGame:
    """Streamlined Gomoku game implementation with parallelization support"""
    
    def __init__(self, board_size: int = 15, num_threads: int = 4):
        """Initialize Gomoku game with parallel processing capability"""
        self.board_size = board_size
        self.board = GomokuBoard(board_size)
        self.players = {
            1: None,  # Black
            2: None   # White
        }
        
        # Configuration with reasonable defaults
        self.save_games = True
        self.games_dir = 'saved_games'
        self.knowledge_dir = 'ai_knowledge'
        self.num_threads = min(num_threads, os.cpu_count() or 4)  # Limit to available cores
        
        # Ensure directories exist
        os.makedirs(self.games_dir, exist_ok=True)
        os.makedirs(self.knowledge_dir, exist_ok=True)
        
        # Game state tracking
        self.current_game_id = None
        self.game_record = []
        self.game_start_time = None
        self.stats = {
            'games_played': 0,
            'black_wins': 0,
            'white_wins': 0,
            'draws': 0,
        }
    
    def set_player(self, player_id: int, ai: Optional[GomokuAI] = None):
        """
        Set player as human or AI
        
        Args:
            player_id: Player ID (1=black, 2=white)
            ai: AI for this player, None means human
        """
        if player_id not in [1, 2]:
            raise ValueError("Player ID must be 1 (black) or 2 (white)")
        
        # Register player
        self.players[player_id] = ai
        if ai:
            ai.start_game(player_id)
        
        player_type = "AI" if ai else "Human"
        print(f"Set player {player_id} as {player_type}")
    
    def create_ai(self, player_id: int, num_threads: int = None) -> GomokuAI:
        """
        Create AI player with multi-threading support
        
        Args:
            player_id: Player ID (1=black, 2=white)
            num_threads: Number of threads for this AI to use
            
        Returns:
            GomokuAI: Created AI instance
        """
        # Default to using the game's thread count
        if num_threads is None:
            num_threads = self.num_threads
            
        # Create powerful AI with parallel processing
        ai = GomokuAI(board_size=self.board_size, num_threads=num_threads)
        
        # Try to load existing knowledge
        knowledge_path = os.path.join(self.knowledge_dir, f"ai_knowledge_p{player_id}.bin")
        
        if os.path.exists(knowledge_path):
            try:
                ai.load_knowledge(knowledge_path)
                print(f"Loaded AI knowledge: {knowledge_path}")
            except Exception as e:
                print(f"Error loading AI knowledge: {e}")
        
        ai.start_game(player_id)
        return ai
    
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
        
        print(f"\n=== Starting new game ID: {self.current_game_id} ===")
        self.board.print_board()
        
        # Game main loop
        with tqdm(total=max_moves, desc="Game Progress", unit="move") as pbar:
            while not self.board.game_over and move_count < max_moves:
                current_player = self.board.current_player
                move_start_time = time.time()
                
                # Get player move
                if self.players[current_player]:  # AI player
                    try:
                        ai_move = self.players[current_player].select_move(self.board)
                        if ai_move >= 0:
                            row, col = ai_move // self.board.size, ai_move % self.board.size
                            
                            thinking_time = time.time() - move_start_time
                            
                            self.board.make_move(row, col)
                            
                            # Record move
                            self.game_record.append({
                                'player': current_player,
                                'move': (row, col),
                                'time': thinking_time,
                                'move_number': move_count + 1
                            })
                        else:
                            print("AI returned invalid move, game terminated")
                            return -1
                    except Exception as e:
                        print(f"Error during AI move: {e}")
                        return -1
                else:  # Human player
                    valid_move = False
                    while not valid_move:
                        try:
                            move_input = input(f"Player {current_player} move (row,col): ")
                            # Check special commands
                            if move_input.lower() in ['quit', 'exit', 'q']:
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
                                    print("Undid last two moves")
                                    # Update progress bar
                                    pbar.n = move_count
                                    pbar.refresh()
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
                
                # Update progress bar
                pbar.update(1)
                
                # Update and display board
                self.board.print_board()
        
        # Game over, display result
        game_duration = time.time() - self.game_start_time
        
        if self.board.winner == 0:
            print("Game ended in a draw")
        else:
            winner_name = "Black" if self.board.winner == 1 else "White"
            print(f"Game over. Winner: {winner_name}")
        
        print(f"Game duration: {game_duration:.2f} seconds, {move_count} moves")
        
        # Update AI knowledge
        for player_id, ai in self.players.items():
            if ai:
                try:
                    ai.learn_from_game(self.board, self.board.winner)
                    
                    # Auto-save AI knowledge
                    knowledge_path = os.path.join(self.knowledge_dir, f"ai_knowledge_p{player_id}.bin")
                    ai.save_knowledge(knowledge_path)
                except Exception as e:
                    print(f"Error updating AI knowledge: {e}")
        
        # Update statistics
        self._update_game_stats()
        
        # Save game record
        if self.save_games:
            self._save_game_record()
        
        return self.board.winner
    
    def _update_game_stats(self):
        """Update game statistics"""
        self.stats['games_played'] += 1
        
        if self.board.winner == 1:
            self.stats['black_wins'] += 1
        elif self.board.winner == 2:
            self.stats['white_wins'] += 1
        else:
            self.stats['draws'] += 1
    
    def _save_game_record(self):
        """Save game record to file"""
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
            
            # Save as JSON file
            file_path = os.path.join(self.games_dir, f"{self.current_game_id}.json")
            
            with open(file_path, 'w') as f:
                json.dump(game_data, f, indent=2)
            
            print(f"Game record saved: {file_path}")
                
        except Exception as e:
            print(f"Error saving game record: {e}")
    
    def self_play(self, num_games: int = 10, parallel: bool = True) -> Dict[str, Any]:
        """
        Run self-play games to train AI with parallel execution
        
        Args:
            num_games: Number of games
            parallel: Whether to use parallel execution
            
        Returns:
            Dict: Game statistics
        """
        if not (self.players[1] and self.players[2]):
            raise ValueError("Self-play requires two AI players")
        
        stats = {
            "black_wins": 0,
            "white_wins": 0,
            "draws": 0,
            "games": []
        }
        
        # Use parallel mode if requested and more than 1 game
        if parallel and num_games > 1:
            return self._parallel_self_play(num_games)
        
        # Serial mode
        print(f"\nStarting self-play training for {num_games} games")
        
        with tqdm(total=num_games, desc="Self-play Progress") as pbar:
            for i in range(num_games):
                print(f"\nStarting self-play game {i+1}/{num_games}")
                
                # Alternate first player for every other game
                if i % 2 == 1:
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
                    
                # Restore original AI role assignment
                if i % 2 == 1:
                    ai1, ai2 = self.players[1], self.players[2]
                    self.set_player(1, ai2)
                    self.set_player(2, ai1)
                
                # Every 10 games prune knowledge tree to avoid excessive growth
                if (i + 1) % 10 == 0:
                    self._prune_ai_knowledge()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'Black': stats['black_wins'], 
                    'White': stats['white_wins'], 
                    'Draws': stats['draws']
                })
        
        print(f"\nSelf-play complete: Black wins: {stats['black_wins']}, "
             f"White wins: {stats['white_wins']}, Draws: {stats['draws']}")
        
        return stats
    
    def _parallel_self_play(self, num_games: int) -> Dict[str, Any]:
        """
        Run self-play games in parallel using multiple processes
        
        Args:
            num_games: Number of games to play
            
        Returns:
            Dict: Game statistics
        """
        # Determine process count
        num_processes = min(self.num_threads, num_games)
        print(f"\nStarting parallel self-play with {num_processes} processes for {num_games} games")
        
        # Create temporary directories for each process
        temp_dirs = []
        for i in range(num_processes):
            process_dir = f"process_{i}"
            process_games_dir = os.path.join(self.games_dir, process_dir)
            process_knowledge_dir = os.path.join(self.knowledge_dir, process_dir)
            
            os.makedirs(process_games_dir, exist_ok=True)
            os.makedirs(process_knowledge_dir, exist_ok=True)
            
            temp_dirs.append((process_games_dir, process_knowledge_dir))
        
        # Prepare AI knowledge files for worker processes
        shared_knowledge = {}
        for player_id, ai in self.players.items():
            if ai and ai.tree:
                # Save current knowledge to be loaded by worker processes
                knowledge_path = os.path.join(self.knowledge_dir, f"shared_knowledge_p{player_id}.bin")
                ai.save_knowledge(knowledge_path)
                shared_knowledge[player_id] = knowledge_path
        
        # Distribute games across processes
        games_per_process = [num_games // num_processes + (1 if i < num_games % num_processes else 0) 
                           for i in range(num_processes)]
        
        # Prepare arguments for each process
        process_args = []
        for i in range(num_processes):
            process_args.append({
                'process_id': i,
                'games_count': games_per_process[i],
                'board_size': self.board_size,
                'games_dir': temp_dirs[i][0],
                'knowledge_dir': temp_dirs[i][1],
                'shared_knowledge': shared_knowledge
            })
        
        # Launch processes
        results = []
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Submit all tasks
            futures = [executor.submit(self._self_play_process, args) for args in process_args]
            
            # Track progress with tqdm
            with tqdm(total=num_games, desc="Self-play Progress") as pbar:
                completed = 0
                
                # Process results as they complete
                for future in as_completed(futures):
                    try:
                        process_results = future.result()
                        results.append(process_results)
                        
                        # Update progress bar
                        games_in_process = process_results['total_games']
                        completed += games_in_process
                        pbar.update(games_in_process)
                        
                        # Update statistics display
                        black_wins = sum(r['black_wins'] for r in results)
                        white_wins = sum(r['white_wins'] for r in results)
                        draws = sum(r['draws'] for r in results)
                        
                        pbar.set_postfix({
                            'Black': black_wins,
                            'White': white_wins,
                            'Draws': draws
                        })
                    except Exception as e:
                        print(f"Error in process: {e}")
        
        # Merge results
        merged_stats = {
            "black_wins": 0,
            "white_wins": 0,
            "draws": 0,
            "games": []
        }
        
        for result in results:
            merged_stats["black_wins"] += result["black_wins"]
            merged_stats["white_wins"] += result["white_wins"]
            merged_stats["draws"] += result["draws"]
            merged_stats["games"].extend(result["games"])
        
        # Merge knowledge from all processes
        self._merge_process_knowledge(num_processes)
        
        # Clean up temporary files
        self._cleanup_temp_files(shared_knowledge)
        
        print(f"\nParallel self-play complete: Black wins: {merged_stats['black_wins']}, "
             f"White wins: {merged_stats['white_wins']}, Draws: {merged_stats['draws']}")
        
        return merged_stats
    
    @staticmethod
    def _self_play_process(args):
        """
        Worker process function for parallel self-play
        
        Args:
            args: Process arguments
            
        Returns:
            Dict: Process statistics
        """
        process_id = args['process_id']
        games_count = args['games_count']
        board_size = args['board_size']
        games_dir = args['games_dir']
        knowledge_dir = args['knowledge_dir']
        shared_knowledge = args['shared_knowledge']
        
        try:
            # Create game instance for this process
            board = GomokuBoard(board_size)
            
            # Create AIs
            ai1 = GomokuAI(board_size=board_size)
            ai2 = GomokuAI(board_size=board_size)
            
            # Load shared knowledge
            if 1 in shared_knowledge:
                ai1.load_knowledge(shared_knowledge[1])
            if 2 in shared_knowledge:
                ai2.load_knowledge(shared_knowledge[2])
            
            # Initialize AIs
            ai1.start_game(1)
            ai2.start_game(2)
            
            # Process statistics
            stats = {
                "black_wins": 0,
                "white_wins": 0,
                "draws": 0,
                "games": [],
                "total_games": games_count
            }
            
            # Play games
            for i in range(games_count):
                # Alternate first player
                black_ai = ai1 if i % 2 == 0 else ai2
                white_ai = ai2 if i % 2 == 0 else ai1
                
                # Reset board
                board.reset()
                
                # Record moves
                game_record = []
                move_count = 0
                game_id = f"game_p{process_id}_{i}"
                
                # Play until game over or max moves reached
                max_moves = board_size * board_size
                
                while not board.game_over and move_count < max_moves:
                    current_player = board.current_player
                    ai = black_ai if current_player == 1 else white_ai
                    
                    # Get move from AI
                    try:
                        move = ai.select_move(board)
                        if move >= 0:
                            row, col = move // board.size, move % board.size
                            
                            # Make move
                            board.make_move(row, col)
                            
                            # Record move
                            game_record.append({
                                'player': current_player,
                                'move': (row, col),
                                'move_number': move_count + 1
                            })
                            
                            move_count += 1
                        else:
                            # Invalid move
                            break
                    except Exception:
                        # Error during move
                        break
                
                # Game completed - record result
                if board.game_over:
                    winner = board.winner
                    
                    # Update statistics
                    if winner == 1:
                        stats["black_wins"] += 1
                    elif winner == 2:
                        stats["white_wins"] += 1
                    else:
                        stats["draws"] += 1
                    
                    # Record game info
                    stats["games"].append({
                        'game_id': game_id,
                        'winner': winner,
                        'moves': move_count,
                        'process_id': process_id
                    })
                    
                    # Update AI knowledge
                    black_ai.learn_from_game(board, winner)
                    white_ai.learn_from_game(board, winner)
                    
                    # Save game record
                    game_data = {
                        'game_id': game_id,
                        'board_size': board_size,
                        'moves': game_record,
                        'result': winner
                    }
                    
                    try:
                        with open(os.path.join(games_dir, f"{game_id}.json"), 'w') as f:
                            json.dump(game_data, f)
                    except Exception:
                        pass
                
                # Prune AI knowledge every 10 games
                if (i + 1) % 10 == 0:
                    try:
                        ai1.prune_knowledge(min_visits=2)
                        ai2.prune_knowledge(min_visits=2)
                    except Exception:
                        pass
            
            # Save final knowledge
            try:
                ai1.save_knowledge(os.path.join(knowledge_dir, f"ai_knowledge_p1.bin"))
                ai2.save_knowledge(os.path.join(knowledge_dir, f"ai_knowledge_p2.bin"))
            except Exception:
                pass
            
            return stats
            
        except Exception as e:
            # Return partial results on error
            print(f"Error in process {process_id}: {e}")
            return {
                "black_wins": 0, 
                "white_wins": 0, 
                "draws": 0, 
                "games": [],
                "total_games": 0,
                "error": str(e)
            }
    
    def _merge_process_knowledge(self, num_processes):
        """Merge knowledge from all processes"""
        print("Merging knowledge from all processes...")
        
        for player_id, ai in self.players.items():
            if ai and ai.tree:
                # Load and merge knowledge from each process
                for i in range(num_processes):
                    process_knowledge_path = os.path.join(
                        self.knowledge_dir, f"process_{i}/ai_knowledge_p{player_id}.bin"
                    )
                    
                    if os.path.exists(process_knowledge_path):
                        try:
                            # Load process knowledge
                            process_tree = load_tree(process_knowledge_path)
                            
                            # Merge into main tree
                            if process_tree:
                                ai.tree.merge(process_tree)
                                print(f"Merged knowledge for player {player_id} from process {i}")
                        except Exception as e:
                            print(f"Error merging knowledge from process {i}: {e}")
                
                # Save merged knowledge
                knowledge_path = os.path.join(self.knowledge_dir, f"ai_knowledge_p{player_id}.bin")
                ai.save_knowledge(knowledge_path)
                
                # Prune merged knowledge
                ai.prune_knowledge(min_visits=2)
    
    def _cleanup_temp_files(self, shared_knowledge):
        """Clean up temporary files after parallel execution"""
        # Remove shared knowledge files
        for _, path in shared_knowledge.items():
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
    
    def _prune_ai_knowledge(self):
        """Prune AI knowledge trees to manage size"""
        for player_id, ai in self.players.items():
            if ai:
                try:
                    pruned = ai.prune_knowledge(min_visits=2)
                    if pruned > 0:
                        print(f"Pruned AI {player_id}'s knowledge tree: removed {pruned} paths")
                except Exception:
                    pass

# Simple example usage
if __name__ == "__main__":
    import os
    
    # Get number of available CPU cores
    num_cores = os.cpu_count() or 4
    print(f"Available CPU cores: {num_cores}")
    
    # Create game with parallel processing
    game = GomokuGame(board_size=15, num_threads=num_cores)
    
    # Simple menu
    while True:
        print("\n=== Gomoku Game Menu ===")
        print("1. Play as Black (vs AI)")
        print("2. Play as White (vs AI)")
        print("3. AI vs AI (single game)")
        print("4. AI vs AI self-play training (parallel)")
        print("5. Check system info")
        print("6. Exit")
        
        choice = input("Choose an option: ")
        
        if choice == "1":
            # Create AI with half the cores
            ai_white = game.create_ai(2, num_threads=max(1, num_cores//2))
            game.set_player(1, None)  # Human plays Black
            game.set_player(2, ai_white)
            game.play_game()
        elif choice == "2":
            # Create AI with half the cores
            ai_black = game.create_ai(1, num_threads=max(1, num_cores//2))
            game.set_player(1, ai_black)
            game.set_player(2, None)  # Human plays White
            game.play_game()
        elif choice == "3":
            # Create two AIs, each with half available cores
            ai_black = game.create_ai(1, num_threads=max(1, num_cores//2))
            ai_white = game.create_ai(2, num_threads=max(1, num_cores//2))
            game.set_player(1, ai_black)
            game.set_player(2, ai_white)
            game.play_game()
        elif choice == "4":
            # Self-play training
            ai_black = game.create_ai(1, num_threads=2)  # Use fewer threads per AI
            ai_white = game.create_ai(2, num_threads=2)  # so processes don't compete
            game.set_player(1, ai_black)
            game.set_player(2, ai_white)
            
            num_games = int(input("Number of games for training: "))
            use_parallel = input("Use parallel execution? (y/n): ").lower() == 'y'
            game.self_play(num_games=num_games, parallel=use_parallel)
        elif choice == "5":
            # Display system info
            print(f"\nSystem Information:")
            print(f"Available CPU cores: {num_cores}")
            print(f"Configured thread count: {game.num_threads}")
            
            # Check AI knowledge
            print("\nAI Knowledge Status:")
            for player_id in [1, 2]:
                knowledge_path = os.path.join(game.knowledge_dir, f"ai_knowledge_p{player_id}.bin")
                if os.path.exists(knowledge_path):
                    size_mb = os.path.getsize(knowledge_path) / (1024 * 1024)
                    print(f"Player {player_id}: {knowledge_path} ({size_mb:.2f} MB)")
                else:
                    print(f"Player {player_id}: No knowledge file found")
        elif choice == "6":
            break
        else:
            print("Invalid choice. Try again.")