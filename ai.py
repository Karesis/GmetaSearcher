"""
GomokuAI: 高级五子棋AI决策引擎
结合FiberTree学习、模式识别、威胁搜索和蒙特卡洛方法
"""

import random
import numpy as np
import time
import pickle
import os
from typing import List, Tuple, Dict, Set, Optional, Any
from collections import defaultdict
from fbtree import create_tree, Move, FiberTree, load_tree

class GomokuAI:
    """
    高级五子棋AI，使用多种策略和学习方法
    """
    
    # 开局库
    OPENING_BOOK = {
        # 中心开局变种
        0: [112, 98, 126, 140, 96, 128, 82, 142, 68, 156],  # 第一手在中心位置的变种
        
        # 常见开局序列 - 示例: [第一手, 第二手, 常见的下几步...]
        112: [98, 126, 97, 127, 113, 111],  # 从中心点开始的序列
        98: [112, 84, 126, 97, 127, 82],    # 左上偏移开局
        96: [112, 80, 128, 97, 95, 111],    # 左偏移开局
    }
    
    # 递归深度限制
    MAX_THREAT_SEARCH_DEPTH = 4
    
    def __init__(self, 
                board_size: int = 15, 
                tree: Optional[FiberTree] = None,
                exploration_factor: float = 1.0,
                max_depth: int = 10,
                use_patterns: bool = True,
                use_opening_book: bool = True, 
                storage_type: str = 'memory',
                db_path: Optional[str] = None,
                time_limit: float = 5.0):
        """
        初始化五子棋AI
        
        Args:
            board_size: 棋盘大小
            tree: 现有FiberTree，未提供则创建新的
            exploration_factor: 控制探索与利用的平衡（更高=更多探索）
            max_depth: 搜索最大深度
            use_patterns: 是否使用基于模式的评估
            use_opening_book: 是否使用开局库
            storage_type: 'memory'或'sqlite'用于FiberTree
            db_path: 使用sqlite存储时的数据库路径
            time_limit: 每步棋的时间限制（秒）
        """
        self.board_size = board_size
        self.tree = tree if tree else create_tree(storage_type=storage_type, db_path=db_path)
        self.exploration_factor = exploration_factor
        self.max_depth = max_depth
        self.use_patterns = use_patterns
        self.use_opening_book = use_opening_book
        self.player_id = None  # 在开始游戏时设置
        self.time_limit = time_limit
        
        # 高级配置
        self.rollout_depth = 8  # 蒙特卡洛rollout深度
        self.cache_size = 100000  # 置换表大小
        self.adaptive_exploration = True  # 根据游戏阶段自适应调整探索因子
        self.use_symmetry = True  # 利用棋盘对称性减少状态空间
        
        # 蒙特卡洛树搜索配置
        self.mcts_config = {
            'c_puct': 5.0,  # PUCT常数（探索权重）
            'num_simulations': 800,  # 默认模拟次数
            'dirichlet_noise': 0.03,  # Dirichlet噪声量（促进探索）
            'value_weight': 0.15  # 价值网络权重
        }
        
        # 缓存与状态
        self._transposition_table = {}  # 状态->评估的缓存
        self._move_history = []  # 记录移动历史
        self._state_visits = defaultdict(int)  # 记录状态访问次数
        self._opening_phase = True  # 是否在开局阶段
        self._endgame_phase = False  # 是否在残局阶段
        
        # 初始化威胁度评估分数
        self.threat_weights = {
            'win': 100000,  # 能直接获胜的步骤
            'block_win': 80000,  # 阻止对手获胜
            'fork': 20000,  # 产生双威胁（如双活四）
            'block_fork': 15000,  # 阻止对手产生双威胁
            'connect4': 5000,  # 形成连四
            'block_connect4': 4000,  # 阻止对手连四
            'connect3': 1000,  # 形成连三
            'block_connect3': 800,  # 阻止对手连三
        }
        
        # 开局历史
        self.opening_stats = {}  # 记录开局效果统计
        
        # 尝试加载开局统计数据
        try:
            if os.path.exists('opening_stats.pkl'):
                with open('opening_stats.pkl', 'rb') as f:
                    self.opening_stats = pickle.load(f)
        except Exception as e:
            print(f"加载开局统计数据时出错: {e}")
    
    def start_game(self, player_id: int):
        """
        设置AI的玩家ID（1为黑，2为白）
        
        Args:
            player_id: AI将扮演的玩家ID
        """
        self.player_id = player_id
        self._move_history = []
        self._transposition_table = {}
        self._opening_phase = True
        self._endgame_phase = False
    
    def select_move(self, board) -> int:
        """
        为当前棋盘状态选择最佳移动
        
        Args:
            board: 当前棋盘状态
            
        Returns:
            int: 选择的移动（一维格式）
        """
        # 计时开始
        start_time = time.time()
        
        if board.game_over:
            return -1
        
        # 获取到目前为止的移动历史
        move_history = [Move(pos) for pos in board.move_history]
        self._move_history = move_history
        
        # 检查棋盘状态，确定游戏阶段
        self._update_game_phase(board)
        
        # 自适应调整探索因子
        if self.adaptive_exploration:
            self._adjust_exploration_factor(board)
        
        # 1. 检查紧急威胁
        immediate_threat = board.detect_immediate_threat()
        if immediate_threat is not None:
            return immediate_threat
        
        # 2. 如果在开局并且使用开局库，查找预设移动
        if self.use_opening_book and self._opening_phase:
            book_move = self._get_book_move(board)
            if book_move is not None:
                return book_move
        
        # 3. 获取合法移动
        legal_moves = board.get_focused_moves(distance=3, consider_threats=True)
        
        # 如果没有专注的移动或游戏初期，获取所有合法移动
        if not legal_moves or len(move_history) < 4:
            legal_moves = board.get_legal_moves()
        
        # 如果只有一个合法移动，直接返回
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # 4. 使用威胁搜索查找强制胜手
        if not self._endgame_phase and len(move_history) > 6:
            forcing_move = self._threat_space_search(board)
            if forcing_move is not None:
                return forcing_move
        
        # 5. 尝试从FiberTree获取最佳后续
        if len(move_history) > 8:  # 当有足够历史时使用FiberTree
            best_continuations = self.tree.get_best_continuation(
                move_history, 
                top_n=min(5, len(legal_moves)),
                min_visits=2
            )
            
            # 如果找到了具有足够数据的后续
            if best_continuations and best_continuations[0]['visits'] >= 3:
                # 根据胜率和访问次数概率性选择
                total_score = sum(c['win_rate'] * (c['visits'] ** 0.5) for c in best_continuations)
                if total_score > 0:
                    # 引入随机性，但偏好高胜率
                    rand_val = random.random() * total_score
                    cumulative = 0
                    for cont in best_continuations:
                        cumulative += cont['win_rate'] * (cont['visits'] ** 0.5)
                        if cumulative >= rand_val:
                            # 验证移动是否合法
                            if cont['move'].value in legal_moves:
                                return cont['move'].value
                
                # 后备：如果最佳移动合法，则使用它
                if best_continuations[0]['move'].value in legal_moves:
                    return best_continuations[0]['move'].value
        
        # 6. 根据游戏阶段选择决策方法
        if self._endgame_phase or (time.time() - start_time) > self.time_limit * 0.3:
            # 残局或时间有限时使用模式评估
            return self._select_move_by_patterns(board, legal_moves)
        else:
            # 中局和开局后期使用Monte Carlo树搜索
            remaining_time = max(0.1, self.time_limit - (time.time() - start_time))
            return self._select_move_by_mcts(board, legal_moves, remaining_time)
    
    def _update_game_phase(self, board):
        """根据棋盘状态更新游戏阶段"""
        move_count = len(board.move_history)
        
        # 开局阶段判断（前12步）
        self._opening_phase = move_count <= 12
        
        # 残局阶段判断（棋盘填充超过60%或者出现关键威胁）
        threshold = (self.board_size ** 2) * 0.6
        self._endgame_phase = move_count >= threshold
        
        # 如果检测到关键威胁模式，也视为进入残局
        if not self._endgame_phase and move_count > 30:
            evaluation = board.evaluate_position()
            # 如果任一方有高级威胁，视为残局
            for player in [1, 2]:
                if (evaluation[player]["patterns"].get("five", 0) > 0 or
                    evaluation[player]["patterns"].get("open_four", 0) > 0 or
                    evaluation[player]["patterns"].get("four", 0) > 1):
                    self._endgame_phase = True
                    break
    
    def _adjust_exploration_factor(self, board):
        """根据游戏阶段自适应调整探索因子"""
        move_count = len(board.move_history)
        board_capacity = self.board_size ** 2
        
        if self._opening_phase:
            # 开局阶段：鼓励探索
            self.exploration_factor = 1.4
        elif self._endgame_phase:
            # 残局阶段：减少探索，专注利用
            self.exploration_factor = 0.6
        else:
            # 中盘：线性从1.2减少到0.8
            mid_game_progress = min(1.0, (move_count - 12) / (board_capacity * 0.6 - 12))
            self.exploration_factor = 1.2 - mid_game_progress * 0.4
    
    def _get_book_move(self, board) -> Optional[int]:
        """从开局库中获取走法"""
        move_history = board.move_history
        
        # 第一步特殊处理
        if not move_history:
            if 0 in self.OPENING_BOOK:
                # 随机选择一个开局变体
                candidate_moves = self.OPENING_BOOK[0]
                # 从开局统计中选择表现最好的开局
                if len(self.opening_stats) > 0:
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
                    
                    if best_moves and best_win_rate > 0.4:  # 只有胜率足够时才使用统计
                        return random.choice(best_moves)
                
                return random.choice(candidate_moves)
        
        # 查找当前历史是否匹配开局序列
        if move_history:
            last_move = move_history[-1]
            if last_move in self.OPENING_BOOK:
                responses = self.OPENING_BOOK[last_move]
                for response in responses:
                    row, col = response // board.size, response % board.size
                    if board.board[row, col] == 0:  # 确保位置未被占用
                        return response
        
        return None
    
    def _threat_space_search(self, board) -> Optional[int]:
        """
        威胁空间搜索，寻找强制胜手
        
        Args:
            board: 当前棋盘
            
        Returns:
            Optional[int]: 找到的强制胜手，无则返回None
        """
        # 为当前玩家找到所有威胁位置
        eval_result = board.evaluate_position()
        threats_current = eval_result[self.player_id]["threats"]
        threats_opponent = eval_result[3 - self.player_id]["threats"]
        
        # 阶段1: 检查直接威胁
        if threats_current:
            # 检查是否有能赢的威胁
            for r, c in threats_current:
                # 模拟落子
                board.board[r, c] = self.player_id
                if board._check_win(r, c):
                    # 如果这步能赢，返回它
                    board.board[r, c] = 0  # 恢复
                    return r * board.size + c
                board.board[r, c] = 0  # 恢复
        
        # 阶段2: 威胁拓展搜索
        # 我们搜索产生重叠威胁的位置（例如双四、三四等）
        best_move = None
        best_score = -1
        
        # 获取专注的走法
        focused_moves = board.get_focused_moves(distance=2)
        
        for move in focused_moves:
            row, col = move // board.size, move % board.size
            
            # 模拟走这步棋
            board.board[row, col] = self.player_id
            
            # 计算这步棋后的威胁能力
            threat_score = self._evaluate_threat_potential(board, row, col, self.player_id, 1)
            
            # 恢复棋盘
            board.board[row, col] = 0
            
            # 更新最佳走法
            if threat_score > best_score:
                best_score = threat_score
                best_move = move
        
        # 如果找到了有足够威胁潜力的走法，返回它
        if best_score >= self.threat_weights['connect4']:
            return best_move
        
        # 阶段3: 阻止对手威胁
        if threats_opponent:
            return threats_opponent[0][0] * board.size + threats_opponent[0][1]
        
        # 没有找到强制性走法
        return None
    
    def _evaluate_threat_potential(self, board, row, col, player, depth):
        """
        评估一步棋的威胁潜力
        
        Args:
            board: 棋盘
            row, col: 走法位置
            player: 当前玩家
            depth: 当前搜索深度
            
        Returns:
            float: 威胁分数
        """
        if depth > self.MAX_THREAT_SEARCH_DEPTH:
            return 0
        
        opponent = 3 - player
        threat_score = 0
        
        # 检查是否获胜
        if board._check_win(row, col):
            return self.threat_weights['win'] / depth  # 更快获胜的价值更高
        
        # 评估这步棋创建的威胁
        eval_current = board.evaluate_position(player)
        
        # 计算威胁分数
        patterns = eval_current[player]["patterns"]
        
        # 计算直接威胁
        if patterns.get("open_four", 0) > 0:
            threat_score += self.threat_weights['connect4']
        if patterns.get("four", 0) > 0:
            threat_score += self.threat_weights['connect4'] * 0.8
        
        # 双重威胁检测
        # 例如: 双三、双四或者三四组合
        threat_count = (patterns.get("open_three", 0) + 
                       patterns.get("three", 0) * 0.5 + 
                       patterns.get("open_four", 0) * 2 + 
                       patterns.get("four", 0))
        
        if threat_count >= 2:
            threat_score += self.threat_weights['fork'] / depth
        
        # 递归检查对手的对策，但限制深度
        if depth < self.MAX_THREAT_SEARCH_DEPTH and threat_score > 0:
            # 获取对手的最佳对策
            best_counter_threat = 0
            
            # 获取对手的可能对策（专注在威胁区域）
            opponent_moves = []
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = row + dr, col + dc
                    if 0 <= r < board.size and 0 <= c < board.size and board.board[r, c] == 0:
                        opponent_moves.append((r, c))
            
            # 如果有太多可能的对策，只考虑有威胁的位置
            if len(opponent_moves) > 6:
                eval_opponent = board.evaluate_position(opponent)
                opponent_threats = eval_opponent[opponent]["threats"]
                if opponent_threats:
                    opponent_moves = [(r, c) for r, c in opponent_threats if board.board[r, c] == 0]
            
            # 评估对手的每个对策
            for r, c in opponent_moves:
                board.board[r, c] = opponent
                counter_threat = self._evaluate_threat_potential(board, r, c, opponent, depth + 1)
                board.board[r, c] = 0
                
                best_counter_threat = max(best_counter_threat, counter_threat)
            
            # 如果对手有强力对策，降低我们威胁的价值
            if best_counter_threat > self.threat_weights['connect3']:
                threat_score *= 0.5
        
        return threat_score
    
    def _select_move_by_patterns(self, board, legal_moves: List[int]) -> int:
        """
        基于模式识别和启发式评估选择走法
        
        Args:
            board: 当前棋盘
            legal_moves: 合法走法列表
            
        Returns:
            int: 选择的走法
        """
        best_move = None
        best_score = float('-inf')
        opponent = 3 - self.player_id
        
        # 尝试每个走法并评估结果位置
        for move in legal_moves:
            row, col = move // board.size, move % board.size
            
            # 使用棋盘的评分方法快速评估（避免创建临时棋盘）
            my_score = board.get_score_for_move(row, col, self.player_id)
            opponent_score = board.get_score_for_move(row, col, opponent)
            
            # 综合考虑进攻与防守
            # 给予阻止对手走法更高的权重（1.2倍）
            combined_score = my_score - opponent_score * 1.2
            
            # 如果是开局，加入位置偏好
            if len(board.move_history) < 10:
                center = board.size // 2
                distance_from_center = abs(row - center) + abs(col - center)
                combined_score -= distance_from_center * 5
            
            # 加入一些随机性以增加变化
            combined_score += random.random() * 5
            
            # 更新最佳走法
            if combined_score > best_score:
                best_score = combined_score
                best_move = move
        
        # 如果找到了好的走法，返回它
        if best_move is not None:
            return best_move
        
        # 后备策略：选择一个随机合法走法
        return random.choice(legal_moves) if legal_moves else -1
    
    def _select_move_by_mcts(self, board, legal_moves: List[int], time_limit: float) -> int:
        """
        使用蒙特卡洛树搜索选择走法
        
        Args:
            board: 当前棋盘
            legal_moves: 合法走法列表
            time_limit: 搜索时间限制（秒）
            
        Returns:
            int: 选择的走法
        """
        class MCTSNode:
            def __init__(self, prior=0):
                self.visit_count = 0
                self.value_sum = 0
                self.children = {}
                self.prior = prior
            
            def get_value(self, parent_visit_count):
                # UCB公式: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
                if self.visit_count == 0:
                    return float('inf') if self.prior > 0 else 0
                
                # 探索项
                exploration = (c_puct * self.prior * 
                              (parent_visit_count ** 0.5) / 
                              (1 + self.visit_count))
                
                # 价值项（价值的平均值）
                value = self.value_sum / self.visit_count
                
                return value + exploration
        
        # 从当前棋盘状态创建一个临时棋盘用于MCTS搜索
        mcts_board = type(board)(board.size)
        mcts_board.board = np.copy(board.board)
        mcts_board.current_player = board.current_player
        
        # 搜索配置
        c_puct = self.mcts_config['c_puct']
        num_simulations = self.mcts_config['num_simulations']
        
        # 根节点
        root = MCTSNode()
        root.visit_count = 1  # Avoid division by zero
        
        # 初始化根节点的子节点
        for move in legal_moves:
            # 使用简单先验概率初始化
            prior_prob = 1.0 / len(legal_moves)
            
            # 如果有FiberTree统计，可以改进先验
            move_obj = Move(move)
            fiber_id = self.tree.find_path(self._move_history + [move_obj])
            if fiber_id:
                stats = self.tree.get_statistics(fiber_id)
                if stats['visit_count'] > 0:
                    # 使用过去的胜率作为先验
                    win_rate = stats['win_count'] / stats['visit_count']
                    prior_prob = max(prior_prob, win_rate)
            
            # 创建子节点
            root.children[move] = MCTSNode(prior=prior_prob)
        
        # 添加Dirichlet噪声以增加探索
        if len(legal_moves) > 0:
            noise = np.random.dirichlet([self.mcts_config['dirichlet_noise']] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                # 将95%的权重给先验，5%给噪声
                root.children[move].prior = 0.95 * root.children[move].prior + 0.05 * noise[i]
        
        # MCTS主循环
        start_time = time.time()
        num_simulations_completed = 0
        
        while (time.time() - start_time < time_limit and 
               num_simulations_completed < num_simulations):
            # 选择阶段：选择最有价值的路径
            node = root
            search_path = [node]
            sim_board = type(board)(board.size)
            sim_board.board = np.copy(board.board)
            sim_board.current_player = board.current_player
            current_moves = []
            
            # 选择阶段（Selection）
            while node.children and not sim_board.game_over:
                # 选择UCB值最高的动作
                max_value = float('-inf')
                best_move = None
                
                for move, child in node.children.items():
                    ucb_value = child.get_value(node.visit_count)
                    if ucb_value > max_value:
                        max_value = ucb_value
                        best_move = move
                
                if best_move is None:
                    break
                
                # 执行最佳动作
                node = node.children[best_move]
                search_path.append(node)
                
                row, col = best_move // sim_board.size, best_move % sim_board.size
                sim_board.make_move(row, col)
                current_moves.append(best_move)
            
            # 扩展阶段（Expansion）
            if not sim_board.game_over:
                # 扩展这个节点
                moves = sim_board.get_focused_moves()
                if not moves:
                    moves = sim_board.get_legal_moves()
                
                # 初始化每个可能的动作
                for move in moves:
                    if move not in node.children:
                        node.children[move] = MCTSNode(prior=1.0/len(moves))
            
            # 模拟阶段（Simulation/Rollout）
            if sim_board.game_over:
                # 游戏已结束，使用确定的结果
                if sim_board.winner == self.player_id:
                    value = 1.0
                elif sim_board.winner == 0:  # 平局
                    value = 0.0
                else:  # 对手赢
                    value = -1.0
            else:
                # 执行快速rollout
                value = self._fast_rollout(sim_board)
            
            # 反向传播阶段（Backpropagation）
            for node in search_path:
                node.visit_count += 1
                node.value_sum += value
                value = -value  # 交替视角
            
            num_simulations_completed += 1
        
        # 选择访问次数最多的动作
        max_visits = -1
        best_move = legal_moves[0] if legal_moves else -1
        
        for move, child in root.children.items():
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_move = move
        
        # 将搜索结果添加到FiberTree
        self.tree.start_path()
        for move in self._move_history:
            self.tree.add_move(move)
        self.tree.add_move(Move(best_move))
        
        # 增加轻微的先验，表示我们选择了这个动作但还不知道结果
        self.tree.record_outcome('draw')  # 暂时记录为平局
        self.tree.end_path()
        
        return best_move
    
    def _fast_rollout(self, board) -> float:
        """
        执行快速随机rollout以估计局面价值
        
        Args:
            board: 当前棋盘
            
        Returns:
            float: 估计的价值（从当前玩家视角）
        """
        sim_board = type(board)(board.size)
        sim_board.board = np.copy(board.board)
        sim_board.current_player = board.current_player
        depth = 0
        
        # 使用较先进的rollout策略而不是纯随机
        while not sim_board.game_over and depth < self.rollout_depth:
            # 首先检查获胜或防守走法
            immediate_move = sim_board.detect_immediate_threat()
            if immediate_move is not None:
                row, col = immediate_move // sim_board.size, immediate_move % sim_board.size
                sim_board.make_move(row, col)
            else:
                # 获取合法走法
                moves = sim_board.get_focused_moves(distance=2)
                if not moves:
                    moves = sim_board.get_legal_moves()
                
                if not moves:
                    break
                
                # 评估所有走法并概率性选择（而不是纯随机）
                scores = []
                for move in moves:
                    row, col = move // sim_board.size, move % sim_board.size
                    score = sim_board.get_score_for_move(row, col, sim_board.current_player)
                    scores.append(max(1.0, score))  # 确保所有走法至少有一定概率
                
                # 随机选择，但偏好高分走法
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
        
        # 如果游戏在rollout中结束
        if sim_board.game_over:
            if sim_board.winner == 0:  # 平局
                return 0.0
            return 1.0 if sim_board.winner == self.player_id else -1.0
        
        # 使用简单的启发式评估未完成的游戏
        eval_result = sim_board.evaluate_position()
        my_score = eval_result[self.player_id]["score"]
        opp_score = eval_result[3 - self.player_id]["score"]
        
        # 归一化成-1到1之间的值
        normalized_score = np.tanh((my_score - opp_score) / 1000.0)
        
        return normalized_score
    
    def learn_from_game(self, board, winner: int):
        """
        从已完成的游戏中更新FiberTree
        
        Args:
            board: 最终棋盘状态
            winner: 游戏胜者（0=平局，1=黑方，2=白方）
        """
        if not board.game_over:
            return
        
        # 将移动历史转换为FiberTree移动
        move_sequence = [Move(pos) for pos in board.move_history]
        
        # 记录结果
        outcome = 'draw'
        if winner == self.player_id:
            outcome = 'win'
        elif winner != 0:  # 不是平局
            outcome = 'loss'
        
        # 在FiberTree中存储路径
        self.tree.start_path()
        for move in move_sequence:
            self.tree.add_move(move)
        self.tree.record_outcome(outcome)
        self.tree.end_path()
        
        # 更新开局统计
        if self.use_opening_book and len(board.move_history) > 0:
            first_move = board.move_history[0]
            if first_move not in self.opening_stats:
                self.opening_stats[first_move] = {"wins": 0, "games": 0}
            
            self.opening_stats[first_move]["games"] += 1
            if (self.player_id == 1 and winner == 1) or (self.player_id == 2 and winner == 2):
                self.opening_stats[first_move]["wins"] += 1
            
            # 保存开局统计
            try:
                with open('opening_stats.pkl', 'wb') as f:
                    pickle.dump(self.opening_stats, f)
            except Exception as e:
                print(f"保存开局统计时出错: {e}")
        
        # 如果利用对称性，还存储等价变换
        if self.use_symmetry:
            last_state = None
            for i, move in enumerate(move_sequence):
                # 为前20步棋存储对称变换，后面的步骤对称变换意义不大
                if i < 20:
                    # 重建棋盘到这一步
                    if last_state is None:
                        temp_board = type(board)(board.size)
                        for j in range(i):
                            prev_move = move_sequence[j].value
                            r, c = prev_move // board.size, prev_move % board.size
                            temp_board.make_move(r, c)
                        last_state = temp_board
                    else:
                        # 使用上一状态继续
                        r, c = move_sequence[i-1].value // board.size, move_sequence[i-1].value % board.size
                        last_state.make_move(r, c)
                    
                    # 获取当前移动的对称变换
                    symmetries = last_state.get_symmetries(move.value)
                    
                    # 存储每个对称变换
                    for sym_board, sym_move in symmetries:
                        # 跳过原始状态
                        if sym_move == move.value:
                            continue
                            
                        # 创建新路径
                        self.tree.start_path()
                        
                        # 添加此对称变换的移动历史
                        for j in range(i):
                            prev_move = move_sequence[j].value
                            sym_prev_move = None
                            
                            # 确定这一步在对称变换中的对应位置
                            r, c = prev_move // board.size, prev_move % board.size
                            for _, test_move in last_state.get_symmetries(prev_move):
                                if test_move != prev_move:
                                    sym_prev_move = test_move
                                    break
                            
                            if sym_prev_move is not None:
                                self.tree.add_move(Move(sym_prev_move))
                            else:
                                self.tree.add_move(Move(prev_move))
                        
                        # 添加当前对称移动
                        self.tree.add_move(Move(sym_move))
                        
                        # 记录结果并结束路径
                        self.tree.record_outcome(outcome)
                        self.tree.end_path()
    
    def save_knowledge(self, file_path: str, compress: bool = True):
        """
        将FiberTree保存到文件
        
        Args:
            file_path: 保存知识的路径
            compress: 是否压缩输出
        """
        self.tree.save(file_path, compress=compress)
        
        # 保存开局统计
        try:
            opening_stats_path = os.path.splitext(file_path)[0] + '_openings.pkl'
            with open(opening_stats_path, 'wb') as f:
                pickle.dump(self.opening_stats, f)
        except Exception as e:
            print(f"保存开局统计时出错: {e}")
    
    def load_knowledge(self, file_path: str):
        """
        从文件加载FiberTree
        
        Args:
            file_path: 知识文件路径
        """
        self.tree = load_tree(file_path)
        
        # 尝试加载开局统计
        try:
            opening_stats_path = os.path.splitext(file_path)[0] + '_openings.pkl'
            if os.path.exists(opening_stats_path):
                with open(opening_stats_path, 'rb') as f:
                    self.opening_stats = pickle.load(f)
        except Exception as e:
            print(f"加载开局统计时出错: {e}")
    
    def prune_knowledge(self, min_visits: int = 3):
        """
        修剪FiberTree，移除很少访问的路径
        
        Args:
            min_visits: 路径保留所需的最小访问次数
        """
        return self.tree.prune_tree(min_visits=min_visits)
    
    def analyze_knowledge(self) -> Dict[str, Any]:
        """
        分析FiberTree中的知识
        
        Returns:
            Dict: 关于FiberTree的分析
        """
        analysis = self.tree.analyze_path_diversity()
        
        # 添加开局统计分析
        if self.opening_stats:
            top_openings = []
            for move, stats in self.opening_stats.items():
                if stats["games"] >= 5:
                    win_rate = stats["wins"] / stats["games"]
                    top_openings.append((move, win_rate, stats["games"]))
            
            top_openings.sort(key=lambda x: x[1], reverse=True)
            analysis["top_openings"] = top_openings[:5]
        
        return analysis
    
    def get_best_first_moves(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        获取基于统计的最佳开局走法
        
        Args:
            top_n: 返回的顶级走法数量
            
        Returns:
            List[Dict]: 关于最佳开局走法的信息
        """
        # 检查所有可能的第一步走法
        first_moves = []
        for pos in range(self.board_size * self.board_size):
            # 检查FiberTree统计
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
            
            # 合并开局库统计
            if pos in self.opening_stats:
                book_stats = self.opening_stats[pos]
                if book_stats["games"] > 0:
                    book_win_rate = book_stats["wins"] / book_stats["games"]
                    
                    # 查找是否已经在列表中
                    found = False
                    for move in first_moves:
                        if move['position'] == pos:
                            # 结合两种统计
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
        
        # 按胜率和访问次数排序
        sorted_moves = sorted(first_moves, 
                             key=lambda x: (x['win_rate'], x['visits']), 
                             reverse=True)
        
        return sorted_moves[:top_n]