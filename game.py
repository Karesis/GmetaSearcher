"""
GomokuGame: 五子棋游戏管理和AI训练系统
提供完整的游戏逻辑、玩家交互和增强训练方法
"""

import os
import time
import random
import pickle
import datetime
import json
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import multiprocessing as mp
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from fbtree import load_tree

from board import GomokuBoard
from ai import GomokuAI

class GomokuGame:
    """
    完整的五子棋游戏，支持人类玩家、AI玩家和高级训练方法
    """
    
    def __init__(self, board_size: int = 15, config: Optional[Dict] = None):
        """
        初始化五子棋游戏
        
        Args:
            board_size: 棋盘大小
            config: 游戏配置选项
        """
        self.board_size = board_size
        self.board = GomokuBoard(board_size)
        self.players = {
            1: None,  # 黑方
            2: None   # 白方
        }
        
        # 默认配置
        self.config = {
            'display_mode': 'text',  # 'text' 或 'gui'
            'save_games': True,      # 是否保存游戏记录
            'games_dir': 'saved_games',  # 游戏记录保存目录
            'thinking_time': 3.0,    # AI思考时间（秒）
            'difficulty': 'normal',  # 'easy', 'normal', 'hard', 'master'
            'collect_stats': True,   # 收集游戏统计数据
            'auto_save': True,       # 自动保存AI知识
            'knowledge_dir': 'ai_knowledge',  # AI知识库目录
            'tournament_mode': False,  # 锦标赛模式
            'verbose': True,         # 详细输出
        }
        
        # 更新自定义配置
        if config:
            self.config.update(config)
        
        # 创建必要的目录
        if self.config['save_games'] and not os.path.exists(self.config['games_dir']):
            os.makedirs(self.config['games_dir'])
        
        if self.config['auto_save'] and not os.path.exists(self.config['knowledge_dir']):
            os.makedirs(self.config['knowledge_dir'])
        
        # 游戏统计
        self.stats = {
            'games_played': 0,
            'black_wins': 0,
            'white_wins': 0,
            'draws': 0,
            'avg_game_length': 0,
            'total_time': 0,
            'opening_stats': defaultdict(lambda: {'played': 0, 'black_wins': 0, 'white_wins': 0}),
            'move_history': []  # 存储游戏历史以分析热点
        }
        
        # 当前游戏状态
        self.current_game_id = None
        self.game_record = []
        self.game_start_time = None
    
    def set_player(self, player_id: int, ai: Optional[GomokuAI] = None, **kwargs):
        """
        设置玩家为人类或AI
        
        Args:
            player_id: 玩家ID（1=黑方，2=白方）
            ai: 用于此玩家的AI，None表示人类
            **kwargs: AI配置选项
        """
        if player_id not in [1, 2]:
            raise ValueError("玩家ID必须为1（黑方）或2（白方）")
        
        # 如果提供了AI但需要配置
        if ai and kwargs:
            for key, value in kwargs.items():
                if hasattr(ai, key):
                    setattr(ai, key, value)
        
        # 设置AI的思考时间
        if ai and 'time_limit' not in kwargs:
            ai.time_limit = self.config['thinking_time']
        
        # 注册玩家
        self.players[player_id] = ai
        if ai:
            ai.start_game(player_id)
    
    def create_ai(self, player_id: int, difficulty: str = None) -> GomokuAI:
        """
        根据难度创建AI玩家
        
        Args:
            player_id: 玩家ID（1=黑方，2=白方）
            difficulty: 难度级别，None表示使用默认配置
            
        Returns:
            GomokuAI: 创建的AI实例
        """
        if difficulty is None:
            difficulty = self.config['difficulty']
        
        # 基于难度级别设置参数
        if difficulty == 'easy':
            ai = GomokuAI(
                board_size=self.board_size,
                exploration_factor=2.0,  # 更多探索，更少利用
                max_depth=4,             # 浅搜索深度
                use_patterns=True,
                use_opening_book=False,  # 不使用开局库
                time_limit=1.0           # 更短的思考时间
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
                exploration_factor=0.8,  # 更少探索，更多利用
                max_depth=10,
                use_patterns=True,
                use_opening_book=True,
                time_limit=5.0,          # 更长的思考时间
                adaptive_exploration=True
            )
        else:
            raise ValueError(f"未知难度级别: {difficulty}")
        
        # 尝试加载已有知识
        knowledge_path = os.path.join(
            self.config['knowledge_dir'], 
            f"ai_knowledge_p{player_id}_{difficulty}.bin"
        )
        
        if os.path.exists(knowledge_path):
            try:
                ai.load_knowledge(knowledge_path)
                if self.config['verbose']:
                    print(f"已加载AI知识: {knowledge_path}")
            except Exception as e:
                print(f"加载AI知识时出错: {e}")
        
        ai.start_game(player_id)
        return ai
    
    def play_game(self, max_moves: int = 225) -> int:
        """
        进行一次完整的游戏
        
        Args:
            max_moves: 在强制结束前的最大移动次数
            
        Returns:
            int: 游戏胜者（0=平局，1=黑方，2=白方）
        """
        # 初始化游戏
        self.board.reset()
        move_count = 0
        self.game_record = []
        self.game_start_time = time.time()
        
        # 生成游戏ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_game_id = f"game_{timestamp}_{random.randint(1000, 9999)}"
        
        if self.config['verbose']:
            print(f"\n=== 开始新游戏 ID: {self.current_game_id} ===")
            self.board.print_board()
        
        # 游戏主循环
        while not self.board.game_over and move_count < max_moves:
            current_player = self.board.current_player
            move_start_time = time.time()
            
            # 获取玩家移动
            if self.players[current_player]:  # AI玩家
                ai_move = self.players[current_player].select_move(self.board)
                if ai_move >= 0:
                    row, col = ai_move // self.board.size, ai_move % self.board.size
                    
                    if self.config['verbose']:
                        thinking_time = time.time() - move_start_time
                        print(f"玩家 {current_player} (AI) 思考时间: {thinking_time:.2f}秒")
                        print(f"玩家 {current_player} (AI) 落子位置: ({row}, {col})")
                    
                    self.board.make_move(row, col)
                    
                    # 记录移动
                    self.game_record.append({
                        'player': current_player,
                        'move': (row, col),
                        'time': thinking_time if 'thinking_time' in locals() else 0,
                        'move_number': move_count + 1
                    })
                else:
                    print("AI返回了无效移动，游戏终止")
                    break
            else:  # 人类玩家
                valid_move = False
                while not valid_move:
                    try:
                        move_input = input(f"玩家 {current_player} 落子 (行,列): ")
                        # 检查特殊命令
                        if move_input.lower() in ['quit', 'exit', 'q']:
                            print("玩家退出游戏")
                            return -1
                        elif move_input.lower() in ['undo', 'u']:
                            # 实现撤销功能
                            if len(self.board.move_history) >= 2:  # 撤销两步（自己的和对手的）
                                for _ in range(2):
                                    if self.game_record:
                                        self.game_record.pop()
                                # 重建棋盘
                                self.board.reset()
                                for record in self.game_record:
                                    r, c = record['move']
                                    self.board.make_move(r, c)
                                move_count = len(self.game_record)
                                self.board.print_board()
                                continue
                            else:
                                print("无法撤销更多步骤")
                                continue
                        
                        # 正常移动输入
                        row, col = map(int, move_input.split(','))
                        thinking_time = time.time() - move_start_time
                        
                        valid_move = self.board.make_move(row, col)
                        if valid_move:
                            # 记录移动
                            self.game_record.append({
                                'player': current_player,
                                'move': (row, col),
                                'time': thinking_time,
                                'move_number': move_count + 1
                            })
                        else:
                            print("无效的落子。请重试。")
                    except (ValueError, IndexError):
                        print("输入无效。使用格式: 行,列 (例如: 7,7)")
            
            move_count += 1
            
            # 更新并显示棋盘
            if self.config['verbose']:
                self.board.print_board()
        
        # 游戏结束，显示结果
        game_duration = time.time() - self.game_start_time
        
        if self.config['verbose']:
            if self.board.winner == 0:
                print("游戏以平局结束")
            else:
                winner_name = "黑方" if self.board.winner == 1 else "白方"
                print(f"游戏结束。胜者: {winner_name}")
            
            print(f"游戏用时: {game_duration:.2f}秒，共 {move_count} 步")
        
        # 更新AI知识
        for player_id, ai in self.players.items():
            if ai:
                ai.learn_from_game(self.board, self.board.winner)
                
                # 自动保存AI知识
                if self.config['auto_save']:
                    difficulty = self.config['difficulty']
                    knowledge_path = os.path.join(
                        self.config['knowledge_dir'], 
                        f"ai_knowledge_p{player_id}_{difficulty}.bin"
                    )
                    ai.save_knowledge(knowledge_path)
                    
                    if self.config['verbose']:
                        print(f"已保存AI知识: {knowledge_path}")
        
        # 收集游戏统计
        if self.config['collect_stats']:
            self._update_game_stats(move_count, game_duration)
        
        # 保存游戏记录
        if self.config['save_games']:
            self._save_game_record()
        
        return self.board.winner
    
    def _update_game_stats(self, move_count: int, game_duration: float):
        """更新游戏统计数据"""
        self.stats['games_played'] += 1
        
        if self.board.winner == 1:
            self.stats['black_wins'] += 1
        elif self.board.winner == 2:
            self.stats['white_wins'] += 1
        else:
            self.stats['draws'] += 1
        
        # 更新平均游戏长度
        prev_total = self.stats['avg_game_length'] * (self.stats['games_played'] - 1)
        self.stats['avg_game_length'] = (prev_total + move_count) / self.stats['games_played']
        
        # 更新总时间
        self.stats['total_time'] += game_duration
        
        # 更新开局统计
        if len(self.game_record) > 0:
            first_move = self.game_record[0]['move']
            move_key = first_move[0] * self.board.size + first_move[1]
            
            self.stats['opening_stats'][move_key]['played'] += 1
            if self.board.winner == 1:
                self.stats['opening_stats'][move_key]['black_wins'] += 1
            elif self.board.winner == 2:
                self.stats['opening_stats'][move_key]['white_wins'] += 1
        
        # 记录移动历史用于热图
        for record in self.game_record:
            r, c = record['move']
            pos = r * self.board.size + c
            self.stats['move_history'].append({
                'position': pos,
                'player': record['player'],
                'game_id': self.current_game_id,
                'result': self.board.winner
            })
    
    def _save_game_record(self):
        """保存游戏记录到文件"""
        if not self.game_record:
            return
        
        # 准备游戏数据
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
        
        # 保存为JSON文件
        file_path = os.path.join(self.config['games_dir'], f"{self.current_game_id}.json")
        with open(file_path, 'w') as f:
            json.dump(game_data, f, indent=2)
        
        if self.config['verbose']:
            print(f"游戏记录已保存: {file_path}")
    
    def self_play(self, num_games: int = 10, parallel: bool = False, cores: int = None) -> Dict[str, int]:
        """
        运行自我对弈游戏来训练AI
        
        Args:
            num_games: 游戏数量
            parallel: 是否并行执行（多核）
            cores: 使用的CPU核心数，None表示使用所有可用核心
            
        Returns:
            Dict: 游戏统计
        """
        if not (self.players[1] and self.players[2]):
            raise ValueError("自我对弈需要两个AI玩家")
        
        stats = {
            "black_wins": 0,
            "white_wins": 0,
            "draws": 0,
            "games": []
        }
        
        # 使用并行模式
        if parallel and num_games > 1:
            return self._parallel_self_play(num_games, cores)
        
        # 串行模式
        for i in range(num_games):
            if self.config['verbose']:
                print(f"\n开始自我对弈游戏 {i+1}/{num_games}")
            
            # 为每局游戏交替先手
            if i % 2 == 1 and not self.config['tournament_mode']:
                # 交换AI角色
                ai1, ai2 = self.players[1], self.players[2]
                self.set_player(1, ai2)
                self.set_player(2, ai1)
            
            # 进行一局游戏
            winner = self.play_game()
            
            # 记录结果
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
                print(f"当前统计: 黑方胜: {stats['black_wins']}, "
                     f"白方胜: {stats['white_wins']}, 平局: {stats['draws']}")
            
            # 恢复原始AI角色分配
            if i % 2 == 1 and not self.config['tournament_mode']:
                ai1, ai2 = self.players[1], self.players[2]
                self.set_player(1, ai2)
                self.set_player(2, ai1)
            
            # 每10局游戏保存一次AI知识
            if (i + 1) % 10 == 0 and self.config['auto_save']:
                for player_id, ai in self.players.items():
                    if ai:
                        difficulty = self.config['difficulty']
                        knowledge_path = os.path.join(
                            self.config['knowledge_dir'], 
                            f"ai_knowledge_p{player_id}_{difficulty}.bin"
                        )
                        ai.save_knowledge(knowledge_path)
                        
                        # 每隔一定局数修剪知识树，避免过度增长
                        if (i + 1) % 50 == 0:
                            pruned = ai.prune_knowledge(min_visits=2)
                            if self.config['verbose']:
                                print(f"已修剪AI {player_id}的知识树，移除了{pruned}个路径")
        
        # 最终知识保存
        if self.config['auto_save']:
            for player_id, ai in self.players.items():
                if ai:
                    difficulty = self.config['difficulty']
                    knowledge_path = os.path.join(
                        self.config['knowledge_dir'], 
                        f"ai_knowledge_p{player_id}_{difficulty}.bin"
                    )
                    ai.save_knowledge(knowledge_path)
        
        return stats
    
    def _parallel_self_play(self, num_games: int, cores: Optional[int] = None) -> Dict[str, int]:
        """
        使用多进程并行执行自我对弈
        
        Args:
            num_games: 游戏数量
            cores: 使用的CPU核心数，None表示使用所有可用核心
            
        Returns:
            Dict: 游戏统计
        """
        if cores is None:
            cores = mp.cpu_count()
        
        cores = min(cores, mp.cpu_count(), num_games)
        
        if self.config['verbose']:
            print(f"使用 {cores} 个CPU核心并行进行 {num_games} 局自我对弈...")
        
        # 保存当前AI知识以便工作进程加载
        temp_knowledge_files = []
        for player_id, ai in self.players.items():
            if ai:
                temp_file = f"temp_knowledge_p{player_id}_{random.randint(1000, 9999)}.bin"
                ai.save_knowledge(temp_file)
                temp_knowledge_files.append((player_id, temp_file))
        
        # 准备工作进程的参数
        games_per_process = [num_games // cores + (1 if i < num_games % cores else 0) 
                           for i in range(cores)]
        
        args_list = []
        for i, games in enumerate(games_per_process):
            # 对每个进程使用不同的游戏保存目录，避免冲突
            process_games_dir = f"{self.config['games_dir']}_process_{i}"
            if not os.path.exists(process_games_dir):
                os.makedirs(process_games_dir)
            
            # 创建进程配置
            process_config = self.config.copy()
            process_config['games_dir'] = process_games_dir
            process_config['verbose'] = False  # 禁用详细输出，避免混乱
            
            args_list.append({
                'process_id': i,
                'num_games': games,
                'board_size': self.board_size,
                'config': process_config,
                'temp_knowledge_files': temp_knowledge_files,
                'difficulty': self.config['difficulty'],
                'tournament_mode': self.config['tournament_mode']
            })
        
        # 使用进程池并行执行游戏
        with mp.Pool(cores) as pool:
            results = pool.map(self._self_play_worker, args_list)
        
        # 汇总结果
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
        
        # 清理临时文件
        for _, file_path in temp_knowledge_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # 合并所有进程的游戏记录
        if self.config['save_games']:
            for i in range(cores):
                process_games_dir = f"{self.config['games_dir']}_process_{i}"
                if os.path.exists(process_games_dir):
                    for filename in os.listdir(process_games_dir):
                        if filename.endswith('.json'):
                            src = os.path.join(process_games_dir, filename)
                            dst = os.path.join(self.config['games_dir'], filename)
                            # 使用唯一文件名避免冲突
                            if os.path.exists(dst):
                                base, ext = os.path.splitext(dst)
                                dst = f"{base}_{random.randint(1000, 9999)}{ext}"
                            os.rename(src, dst)
        
        # 合并所有进程的AI知识
        # 这需要加载每个进程的AI知识并合并
        if self.config['auto_save']:
            for process_id in range(cores):
                for player_id in [1, 2]:
                    knowledge_path = os.path.join(
                        f"{self.config['knowledge_dir']}_process_{process_id}", 
                        f"ai_knowledge_p{player_id}_{self.config['difficulty']}.bin"
                    )
                    
                    if os.path.exists(knowledge_path):
                        # 加载此进程的知识
                        process_tree = load_tree(knowledge_path)
                        
                        # 合并到主AI
                        if self.players[player_id]:
                            self.players[player_id].tree.merge(process_tree)
                            
                            # 保存合并后的知识
                            merged_path = os.path.join(
                                self.config['knowledge_dir'], 
                                f"ai_knowledge_p{player_id}_{self.config['difficulty']}.bin"
                            )
                            self.players[player_id].save_knowledge(merged_path)
        
        if self.config['verbose']:
            print(f"并行自我对弈完成: 黑方胜: {stats['black_wins']}, "
                 f"白方胜: {stats['white_wins']}, 平局: {stats['draws']}")
        
        return stats
    
    @staticmethod
    def _self_play_worker(args):
        """
        用于并行自我对弈的工作函数
        
        Args:
            args: 包含游戏参数的字典
            
        Returns:
            Dict: 游戏统计
        """
        process_id = args['process_id']
        num_games = args['num_games']
        board_size = args['board_size']
        config = args['config']
        temp_knowledge_files = args['temp_knowledge_files']
        difficulty = args['difficulty']
        tournament_mode = args['tournament_mode']
        
        # 创建新的游戏实例
        game = GomokuGame(board_size=board_size, config=config)
        
        # 创建并加载AI
        ai1 = game.create_ai(1, difficulty)
        ai2 = game.create_ai(2, difficulty)
        
        # 加载临时知识文件
        for player_id, file_path in temp_knowledge_files:
            if os.path.exists(file_path):
                if player_id == 1:
                    ai1.load_knowledge(file_path)
                elif player_id == 2:
                    ai2.load_knowledge(file_path)
        
        # 设置AI玩家
        game.set_player(1, ai1)
        game.set_player(2, ai2)
        
        # 进行自我对弈
        stats = {
            "black_wins": 0,
            "white_wins": 0,
            "draws": 0,
            "games": []
        }
        
        for i in range(num_games):
            # 为每局游戏交替先手
            if i % 2 == 1 and not tournament_mode:
                # 交换AI角色
                game.set_player(1, ai2)
                game.set_player(2, ai1)
            else:
                game.set_player(1, ai1)
                game.set_player(2, ai2)
            
            # 进行一局游戏
            winner = game.play_game()
            
            # 记录结果
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
            
            # 每10局游戏保存一次AI知识
            if (i + 1) % 10 == 0 and config['auto_save']:
                # 确保进程的知识目录存在
                knowledge_dir = f"{config['knowledge_dir']}_process_{process_id}"
                if not os.path.exists(knowledge_dir):
                    os.makedirs(knowledge_dir)
                
                for player_id, ai in [(1, ai1), (2, ai2)]:
                    knowledge_path = os.path.join(
                        knowledge_dir, 
                        f"ai_knowledge_p{player_id}_{difficulty}.bin"
                    )
                    ai.save_knowledge(knowledge_path)
        
        # 最终知识保存
        if config['auto_save']:
            # 确保进程的知识目录存在
            knowledge_dir = f"{config['knowledge_dir']}_process_{process_id}"
            if not os.path.exists(knowledge_dir):
                os.makedirs(knowledge_dir)
            
            for player_id, ai in [(1, ai1), (2, ai2)]:
                knowledge_path = os.path.join(
                    knowledge_dir, 
                    f"ai_knowledge_p{player_id}_{difficulty}.bin"
                )
                ai.save_knowledge(knowledge_path)
        
        return stats
    
    def run_tournament(self, 
                     participants: List[Dict], 
                     games_per_match: int = 10, 
                     parallel: bool = False) -> Dict:
        """
        运行AI之间的锦标赛
        
        Args:
            participants: 参赛者列表，每个参赛者是带有'id'和'ai'键的字典
            games_per_match: 每对选手之间的比赛次数
            parallel: 是否使用并行计算
            
        Returns:
            Dict: 锦标赛结果
        """
        if len(participants) < 2:
            raise ValueError("锦标赛需要至少两名参与者")
        
        # 配置锦标赛模式
        tournament_config = self.config.copy()
        tournament_config['tournament_mode'] = True
        tournament_config['verbose'] = False
        self.config.update(tournament_config)
        
        # 锦标赛结果
        results = {
            'matches': [],
            'standings': {p['id']: {'wins': 0, 'losses': 0, 'draws': 0, 'points': 0, 'played': 0}
                        for p in participants}
        }
        
        # 打印锦标赛信息
        print(f"\n=== 开始锦标赛 ({len(participants)}名参与者, 每对{games_per_match}局) ===")
        for p in participants:
            print(f"参与者: {p['id']}")
        
        # 所有可能的对战组合
        matches = []
        for i in range(len(participants)):
            for j in range(i+1, len(participants)):
                matches.append((participants[i], participants[j]))
        
        # 进行比赛
        for match_idx, (p1, p2) in enumerate(matches):
            print(f"\n比赛 {match_idx+1}/{len(matches)}: {p1['id']} vs {p2['id']}")
            
            # 设置AI玩家
            self.set_player(1, p1['ai'])
            self.set_player(2, p2['ai'])
            
            # 进行比赛
            match_stats = self.self_play(num_games=games_per_match, parallel=parallel)
            
            # 记录比赛结果
            match_result = {
                'player1': p1['id'],
                'player2': p2['id'],
                'p1_wins': match_stats['black_wins'],
                'p2_wins': match_stats['white_wins'],
                'draws': match_stats['draws'],
                'games': match_stats['games']
            }
            results['matches'].append(match_result)
            
            # 更新排名
            results['standings'][p1['id']]['played'] += games_per_match
            results['standings'][p2['id']]['played'] += games_per_match
            
            results['standings'][p1['id']]['wins'] += match_stats['black_wins']
            results['standings'][p1['id']]['losses'] += match_stats['white_wins']
            results['standings'][p1['id']]['draws'] += match_stats['draws']
            
            results['standings'][p2['id']]['wins'] += match_stats['white_wins']
            results['standings'][p2['id']]['losses'] += match_stats['black_wins']
            results['standings'][p2['id']]['draws'] += match_stats['draws']
            
            # 计算积分（胜=3分，平=1分）
            results['standings'][p1['id']]['points'] = (
                results['standings'][p1['id']]['wins'] * 3 + 
                results['standings'][p1['id']]['draws']
            )
            
            results['standings'][p2['id']]['points'] = (
                results['standings'][p2['id']]['wins'] * 3 + 
                results['standings'][p2['id']]['draws']
            )
            
            # 打印比赛结果
            print(f"结果: {p1['id']} {match_stats['black_wins']} - {match_stats['white_wins']} {p2['id']} (平局: {match_stats['draws']})")
        
        # 计算最终排名
        standings = sorted(
            results['standings'].items(), 
            key=lambda x: (x[1]['points'], x[1]['wins']), 
            reverse=True
        )
        
        # 打印锦标赛结果
        print("\n=== 锦标赛结果 ===")
        print(f"{'选手':15} {'积分':5} {'胜':5} {'负':5} {'平':5} {'总局数':5}")
        print("-" * 45)
        
        for i, (participant_id, stats) in enumerate(standings):
            print(f"{i+1}. {participant_id:12} {stats['points']:5} {stats['wins']:5} {stats['losses']:5} {stats['draws']:5} {stats['played']:5}")
        
        # 保存锦标赛结果
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tournament_file = os.path.join(self.config['games_dir'], f"tournament_{timestamp}.json")
        
        with open(tournament_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n锦标赛结果已保存: {tournament_file}")
        
        # 恢复原始配置
        self.config['tournament_mode'] = False
        self.config['verbose'] = True
        
        return results
    
    def analyze_games(self, num_games: int = None) -> Dict[str, Any]:
        """
        分析已保存的游戏记录
        
        Args:
            num_games: 分析的游戏数量，None表示所有游戏
            
        Returns:
            Dict: 分析结果
        """
        if not os.path.exists(self.config['games_dir']):
            return {"error": "游戏目录不存在"}
        
        # 收集游戏文件
        game_files = [f for f in os.listdir(self.config['games_dir']) 
                     if f.endswith('.json') and f.startswith('game_')]
        
        if not game_files:
            return {"error": "没有找到游戏记录"}
        
        if num_games and num_games < len(game_files):
            # 随机选择游戏文件
            game_files = random.sample(game_files, num_games)
        
        print(f"分析 {len(game_files)} 局游戏记录...")
        
        # 分析结果
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
        
        # 分析每局游戏
        total_moves = 0
        move_data = []
        
        for file_name in game_files:
            file_path = os.path.join(self.config['games_dir'], file_name)
            try:
                with open(file_path, 'r') as f:
                    game_data = json.load(f)
                
                # 基本统计
                result = game_data.get('result', 0)
                if result == 1:
                    analysis['black_wins'] += 1
                elif result == 2:
                    analysis['white_wins'] += 1
                else:
                    analysis['draws'] += 1
                
                moves = game_data.get('moves', [])
                total_moves += len(moves)
                
                # 记录开局统计
                if moves:
                    first_move = tuple(moves[0]['move'])
                    first_move_key = first_move[0] * self.board_size + first_move[1]
                    analysis['opening_stats'][first_move_key]['count'] += 1
                    
                    if result == 1:
                        analysis['opening_stats'][first_move_key]['black_wins'] += 1
                    elif result == 2:
                        analysis['opening_stats'][first_move_key]['white_wins'] += 1
                
                # 更新热图
                for move in moves:
                    r, c = move['move']
                    analysis['position_heatmap'][r, c] += 1
                    
                    # 记录落子位置与结果
                    move_data.append({
                        'position': (r, c),
                        'player': move['player'],
                        'result': result,
                        'move_number': move.get('move_number', 0)
                    })
                    
                    # 记录时间统计
                    move_time = move.get('time', 0)
                    move_num = move.get('move_number', 0)
                    game_phase = 'early' if move_num <= 10 else ('mid' if move_num <= 30 else 'late')
                    analysis['move_timing'][game_phase].append(move_time)
                    
            except Exception as e:
                print(f"分析游戏 {file_name} 时出错: {e}")
        
        # 计算平均游戏长度
        if analysis['total_games'] > 0:
            analysis['avg_game_length'] = total_moves / analysis['total_games']
        
        # 按位置分析胜率
        position_results = defaultdict(lambda: {'black_wins': 0, 'white_wins': 0, 'total': 0})
        
        for move in move_data:
            pos = (move['position'][0], move['position'][1])
            position_results[pos]['total'] += 1
            
            if move['result'] == 1:
                position_results[pos]['black_wins'] += 1
            elif move['result'] == 2:
                position_results[pos]['white_wins'] += 1
        
        # 计算每个位置的胜率
        for pos, stats in position_results.items():
            if stats['total'] >= 5:  # 只考虑有足够样本的位置
                black_win_rate = stats['black_wins'] / stats['total'] if stats['total'] > 0 else 0
                white_win_rate = stats['white_wins'] / stats['total'] if stats['total'] > 0 else 0
                analysis['win_rate_by_position'][pos] = {
                    'black_win_rate': black_win_rate,
                    'white_win_rate': white_win_rate,
                    'total_games': stats['total']
                }
        
        # 寻找常见开局模式
        top_openings = sorted(
            analysis['opening_stats'].items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )[:5]
        
        analysis['common_patterns'] = [
            {
                'position': (key // self.board_size, key % self.board_size),
                'count': stats['count'],
                'black_win_rate': stats['black_wins'] / stats['count'] if stats['count'] > 0 else 0,
                'white_win_rate': stats['white_wins'] / stats['count'] if stats['count'] > 0 else 0
            }
            for key, stats in top_openings
        ]
        
        # 可视化热图
        self._visualize_heatmap(analysis['position_heatmap'])
        
        return analysis
    
    def _visualize_heatmap(self, heatmap: np.ndarray):
        """可视化棋盘热图"""
        plt.figure(figsize=(10, 10))
        
        # 使用棋盘颜色
        plt.imshow(heatmap, cmap='YlOrRd')
        
        # 添加网格
        plt.grid(color='black', linestyle='-', linewidth=0.5)
        
        # 设置刻度
        plt.xticks(np.arange(self.board_size))
        plt.yticks(np.arange(self.board_size))
        
        # 添加标题和标签
        plt.title('走子热图分析')
        plt.colorbar(label='落子频率')
        
        # 保存图像
        plt.savefig('move_heatmap.png')
        
        if self.config['verbose']:
            print("热图已保存为 'move_heatmap.png'")
    
    def train_curriculum(self, 
                       phases: List[Dict], 
                       save_interval: int = 50,
                       base_dir: str = "curriculum_training") -> Dict:
        """
        执行课程学习训练，逐步增加难度
        
        Args:
            phases: 训练阶段列表，每个阶段是带有参数的字典
            save_interval: 自动保存间隔（局数）
            base_dir: 训练数据基础目录
            
        Returns:
            Dict: 训练结果
        """
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # 训练结果
        training_results = {
            'phases': [],
            'overall_stats': {
                'total_games': 0,
                'training_time': 0
            }
        }
        
        # 实现课程学习
        training_start_time = time.time()
        
        for phase_idx, phase in enumerate(phases):
            phase_name = phase.get('name', f"Phase {phase_idx+1}")
            games_count = phase.get('games', 100)
            difficulty = phase.get('difficulty', 'normal')
            parallel = phase.get('parallel', False)
            
            print(f"\n=== 开始训练阶段: {phase_name} ===")
            print(f"游戏数: {games_count}, 难度: {difficulty}, 并行: {parallel}")
            
            # 设置阶段特定配置
            phase_config = self.config.copy()
            phase_config.update(phase.get('config', {}))
            phase_config['difficulty'] = difficulty
            
            # 创建阶段目录
            phase_dir = os.path.join(base_dir, f"phase_{phase_idx+1}")
            if not os.path.exists(phase_dir):
                os.makedirs(phase_dir)
            
            # 更新游戏和知识库目录
            phase_config['games_dir'] = os.path.join(phase_dir, 'games')
            phase_config['knowledge_dir'] = os.path.join(phase_dir, 'knowledge')
            
            if not os.path.exists(phase_config['games_dir']):
                os.makedirs(phase_config['games_dir'])
            if not os.path.exists(phase_config['knowledge_dir']):
                os.makedirs(phase_config['knowledge_dir'])
            
            # 更新游戏配置
            self.config.update(phase_config)
            
            # 创建或更新AI
            ai1 = None
            ai2 = None
            
            # 如果指定了AI1的知识库路径，加载它
            if 'ai1_knowledge' in phase:
                ai1 = GomokuAI(board_size=self.board_size)
                ai1.load_knowledge(phase['ai1_knowledge'])
            else:
                # 使用上一阶段的AI（如果有）
                if self.players[1]:
                    ai1 = self.players[1]
                else:
                    ai1 = self.create_ai(1, difficulty)
            
            # 设置AI配置
            if 'ai1_config' in phase:
                for key, value in phase['ai1_config'].items():
                    if hasattr(ai1, key):
                        setattr(ai1, key, value)
            
            # 同样处理AI2
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
            
            # 设置玩家
            self.set_player(1, ai1)
            self.set_player(2, ai2)
            
            # 进行自我对弈训练
            phase_start_time = time.time()
            phase_stats = self.self_play(num_games=games_count, parallel=parallel)
            phase_duration = time.time() - phase_start_time
            
            # 保存阶段AI知识
            ai1_path = os.path.join(phase_config['knowledge_dir'], f"ai1_phase_{phase_idx+1}.bin")
            ai2_path = os.path.join(phase_config['knowledge_dir'], f"ai2_phase_{phase_idx+1}.bin")
            
            ai1.save_knowledge(ai1_path)
            ai2.save_knowledge(ai2_path)
            
            # 记录阶段结果
            phase_result = {
                'name': phase_name,
                'games': games_count,
                'difficulty': difficulty,
                'black_wins': phase_stats['black_wins'],
                'white_wins': phase_stats['white_wins'],
                'draws': phase_stats['draws'],
                'win_rate_black': phase_stats['black_wins'] / games_count if games_count > 0 else 0,
                'win_rate_white': phase_stats['white_wins'] / games_count if games_count > 0 else 0,
                'duration': phase_duration,
                'ai1_knowledge_path': ai1_path,
                'ai2_knowledge_path': ai2_path
            }
            
            training_results['phases'].append(phase_result)
            training_results['overall_stats']['total_games'] += games_count
            
            # 打印阶段结果
            print(f"\n阶段 {phase_name} 完成:")
            print(f"黑方胜: {phase_stats['black_wins']}/{games_count} ({phase_result['win_rate_black']:.2%})")
            print(f"白方胜: {phase_stats['white_wins']}/{games_count} ({phase_result['win_rate_white']:.2%})")
            print(f"平局: {phase_stats['draws']}/{games_count} ({phase_stats['draws']/games_count:.2%})")
            print(f"用时: {phase_duration:.2f}秒")
            
            # 分析阶段数据
            phase_analysis = self.analyze_games(min(100, games_count))
            
            # 保存阶段信息
            phase_info = {
                'phase_result': phase_result,
                'phase_analysis': phase_analysis
            }
            
            phase_info_path = os.path.join(phase_dir, 'phase_info.json')
            with open(phase_info_path, 'w') as f:
                # 由于Numpy数组不能直接序列化为JSON，需要转换
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
                
                json.dump(convert_numpy(phase_info), f, indent=2)
        
        # 计算总训练时间
        training_results['overall_stats']['training_time'] = time.time() - training_start_time
        
        # 保存总训练结果
        training_results_path = os.path.join(base_dir, 'training_results.json')
        with open(training_results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        print(f"\n=== 课程学习训练完成 ===")
        print(f"总游戏数: {training_results['overall_stats']['total_games']}")
        print(f"总用时: {training_results['overall_stats']['training_time']:.2f}秒")
        print(f"训练结果已保存: {training_results_path}")
        
        return training_results


# 使用示例
if __name__ == "__main__":
    # 创建游戏
    game = GomokuGame(board_size=15)
    
    # 创建两个AI
    ai_black = game.create_ai(1, difficulty='normal')
    ai_white = game.create_ai(2, difficulty='normal')
    
    # 设置玩家
    game.set_player(1, ai_black)  # 黑方为AI
    game.set_player(2, ai_white)  # 白方为AI
    
    # 进行10局自我对弈
    stats = game.self_play(num_games=10, parallel=True)
    print(f"自我对弈结果: {stats}")
    
    # 分析游戏
    analysis = game.analyze_games()
    print(f"游戏分析: {analysis}")
    
    # 创建一个人类对AI的游戏
    game.set_player(1, ai_black)  # 黑方为AI
    game.set_player(2, None)      # 白方为人类
    
    print("\n人类vs AI游戏开始（你是白方）:")
    game.play_game()