"""
GomokuBoard: 增强版五子棋棋盘模块
包含棋盘状态、规则执行和高效模式评估
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Any
import itertools


class GomokuBoard:
    """
    表示五子棋游戏棋盘，具有增强的模式评估和高效状态管理
    """
    
    # 定义棋型及其对应的威胁等级
    PATTERNS = {
        # 棋型名称: (模式, 开放端数, 分数)
        'five': ('11111', 0, 100000),  # 五连
        'open_four': ('011110', 2, 10000),  # 活四
        'four': ('011112', 1, 1000),  # 冲四
        'four_b': ('211110', 1, 1000),  # 冲四变体
        'four_c': ('11101', 1, 1000),  # 跳冲四
        'four_d': ('10111', 1, 1000),  # 跳冲四变体
        'open_three': ('01110', 2, 1000),  # 活三
        'three': ('211100', 1, 100),  # 眠三
        'three_b': ('001112', 1, 100),  # 眠三变体
        'three_c': ('11100', 1, 100),  # 眠三变体
        'three_d': ('00111', 1, 100),  # 眠三变体
        'open_two': ('00110', 2, 10),  # 活二
        'open_two_b': ('01100', 2, 10),  # 活二变体
        'two': ('11000', 1, 6),  # 眠二
        'two_b': ('00011', 1, 6),  # 眠二变体
    }
    
    # 分数标准化因子
    SCORE_FACTOR = 1.0
    
    def __init__(self, size: int = 15):
        """
        初始化棋盘
        
        Args:
            size: 棋盘大小，默认为15x15
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)  # 0: 空, 1: 黑, 2: 白
        self.current_player = 1  # 黑方先行
        self.move_history = []
        self.last_move = None
        self.game_over = False
        self.winner = None
        
        # 模式评估的缓存和增量计算
        self._pattern_cache = {}  # 缓存每个位置每个方向的模式
        self._move_scores = {}  # 缓存每个可能移动的分数
        self._line_strings = {}  # 缓存每个方向的行字符串
        self._threat_positions = set()  # 需要特别注意的威胁位置
        
        # 预计算棋盘位置的重要性（距离中心的曼哈顿距离）
        self._position_importance = self._calculate_position_importance()
    
    def _calculate_position_importance(self) -> np.ndarray:
        """计算棋盘位置的重要性（基于到中心的距离）"""
        importance = np.zeros((self.size, self.size), dtype=np.float32)
        center = self.size // 2
        
        for r in range(self.size):
            for c in range(self.size):
                # 计算到中心的曼哈顿距离，并转换为重要性分数（越近越高）
                dist = abs(r - center) + abs(c - center)
                max_dist = 2 * center
                importance[r, c] = 1.0 - (dist / max_dist) * 0.8  # 保留0.2作为基础值
        
        return importance
    
    def reset(self):
        """重置棋盘为初始状态"""
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1
        self.move_history = []
        self.last_move = None
        self.game_over = False
        self.winner = None
        
        # 重置缓存
        self._pattern_cache = {}
        self._move_scores = {}
        self._line_strings = {}
        self._threat_positions = set()
    
    def make_move(self, row: int, col: int) -> bool:
        """
        在指定位置落子
        
        Args:
            row: 行索引（从0开始）
            col: 列索引（从0开始）
            
        Returns:
            bool: 如果移动有效并成功则为True，否则为False
        """
        # 检查游戏是否已结束
        if self.game_over:
            return False
            
        # 检查位置是否在边界内
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
            
        # 检查位置是否为空
        if self.board[row, col] != 0:
            return False
        
        # 落子
        self.board[row, col] = self.current_player
        pos = row * self.size + col  # 转换为一维位置
        self.move_history.append(pos)
        self.last_move = (row, col)
        
        # 清除受此移动影响的缓存
        self._invalidate_caches(row, col)
        
        # 检查是否获胜
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        # 检查是否和棋
        elif len(self.move_history) == self.size * self.size:
            self.game_over = True
            self.winner = 0  # 和棋
        
        # 切换玩家
        self.current_player = 3 - self.current_player  # 在1和2之间切换
        return True
    
    def make_move_1d(self, pos: int) -> bool:
        """
        使用一维位置落子（0到size²-1）
        
        Args:
            pos: 一维位置（row * size + column）
            
        Returns:
            bool: 如果移动有效并成功则为True
        """
        row, col = pos // self.size, pos % self.size
        return self.make_move(row, col)
    
    def _invalidate_caches(self, row: int, col: int):
        """使受特定移动影响的缓存无效"""
        # 清除受影响的模式缓存
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            # 检查影响范围（通常是4个位置）
            for offset in range(-4, 5):
                r, c = row + dr * offset, col + dc * offset
                if 0 <= r < self.size and 0 <= c < self.size:
                    key = (r, c, dr, dc)
                    if key in self._pattern_cache:
                        del self._pattern_cache[key]
        
        # 清除移动分数缓存
        self._move_scores = {}
        
        # 清除受影响的行字符串缓存
        for dr, dc in directions:
            # 计算该方向上的行起点
            start_r, start_c = row, col
            while 0 <= start_r - dr < self.size and 0 <= start_c - dc < self.size:
                start_r -= dr
                start_c -= dc
            key = (start_r, start_c, dr, dc)
            if key in self._line_strings:
                del self._line_strings[key]
    
    def get_legal_moves(self) -> List[int]:
        """
        获取所有合法移动（一维格式）
        
        Returns:
            List[int]: 合法位置列表，以一维索引表示
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
        获取位于现有棋子周围一定距离内的合法移动
        
        Args:
            distance: 考虑的最大距离
            consider_threats: 是否特别考虑威胁位置
            
        Returns:
            List[int]: 合法移动列表，专注于相关位置
        """
        if not self.move_history:
            # 如果棋盘为空，返回中心点和附近位置
            mid = self.size // 2
            result = [(mid * self.size + mid)]  # 中心点
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)]:
                r, c = mid + dr, mid + dc
                if 0 <= r < self.size and 0 <= c < self.size:
                    result.append(r * self.size + c)
            return result
        
        if self.game_over:
            return []
        
        # 找出所有需要考虑的位置
        candidates = set()
        
        # 获取所有有棋子的位置
        occupied = set()
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    occupied.add((r, c))
        
        # 找出距离任何棋子在指定范围内的空位置
        for r, c in occupied:
            for dr in range(-distance, distance + 1):
                for dc in range(-distance, distance + 1):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.size and 0 <= nc < self.size and 
                        self.board[nr, nc] == 0 and 
                        (dr != 0 or dc != 0)):  # 排除棋子本身
                        candidates.add((nr, nc))
        
        # 如果需要考虑威胁位置，将它们添加到候选集中
        if consider_threats and self._threat_positions:
            for r, c in self._threat_positions:
                if 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == 0:
                    candidates.add((r, c))
        
        # 转换为一维位置
        return [r * self.size + c for r, c in candidates]
    
    def _check_win(self, row: int, col: int) -> bool:
        """
        检查最后一步落子是否创建了获胜线
        
        Args:
            row: 最后落子的行
            col: 最后落子的列
            
        Returns:
            bool: 如果该移动创建了获胜线则为True
        """
        player = self.board[row, col]
        
        # 定义四个方向：水平、垂直、对角线、反对角线
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1  # 计算连续棋子数量（包括当前棋子）
            
            # 检查正方向
            r, c = row + dr, col + dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            
            # 检查负方向
            r, c = row - dr, col - dc
            while 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            
            # 如果连续5个或更多则获胜
            if count >= 5:
                return True
                
        return False
    
    def evaluate_position(self, player_perspective: Optional[int] = None) -> Dict[str, Any]:
        """
        评估当前棋盘位置，计算双方的棋型计数和总分
        
        Args:
            player_perspective: 从哪个玩家的角度评估(1=黑，2=白，None=当前玩家)
            
        Returns:
            Dict: 包含棋型计数和总分的评估结果
        """
        if player_perspective is None:
            player_perspective = self.current_player
        
        opponent = 3 - player_perspective
        
        # 初始化结果
        result = {
            player_perspective: {"patterns": {}, "score": 0, "threats": []},
            opponent: {"patterns": {}, "score": 0, "threats": []}
        }
        
        # 初始化棋型计数
        for player in [player_perspective, opponent]:
            for pattern_name in self.PATTERNS.keys():
                result[player]["patterns"][pattern_name] = 0
        
        # 扫描整个棋盘的所有行、列和对角线
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        # 记录检测到的威胁位置
        threats = set()
        
        # 扫描行
        for start_row in range(self.size):
            for start_col in range(self.size):
                for dr, dc in directions:
                    # 确保这是一个新行的起点
                    if (start_row > 0 and dr != 0) or (start_col > 0 and dc != 0):
                        continue
                        
                    line_key = (start_row, start_col, dr, dc)
                    if line_key in self._line_strings:
                        line_str = self._line_strings[line_key]
                    else:
                        # 提取行
                        line = []
                        r, c = start_row, start_col
                        while 0 <= r < self.size and 0 <= c < self.size:
                            line.append(self.board[r, c])
                            r += dr
                            c += dc
                        
                        # 如果行太短则跳过
                        if len(line) < 5:
                            continue
                            
                        # 转换为字符串以便模式匹配
                        line_str = ''.join(map(str, line))
                        self._line_strings[line_key] = line_str
                    
                    # 对双方玩家匹配模式
                    for player in [player_perspective, opponent]:
                        # 将对手棋子替换为'2'，将我方棋子替换为'1'，空位替换为'0'
                        if player == player_perspective:
                            player_str = str(player)
                            opponent_str = str(opponent)
                        else:
                            player_str = str(opponent)
                            opponent_str = str(player_perspective)
                            
                        pattern_str = line_str.replace(player_str, '1').replace(opponent_str, '2').replace('0', '0')
                        
                        # 检查是否有匹配的棋型
                        for pattern_name, (pattern, _, score) in self.PATTERNS.items():
                            # 计算为这个玩家匹配的模式数量
                            matches = self._find_pattern_matches(pattern_str, pattern)
                            if matches > 0:
                                result[player]["patterns"][pattern_name] += matches
                                result[player]["score"] += matches * score
                                
                                # 如果是威胁性棋型，标记威胁位置
                                if pattern_name in ['five', 'open_four', 'four', 'four_b', 'four_c', 'four_d']:
                                    # 找出威胁位置（下一步可能获胜的位置）
                                    threat_pos = self._find_threat_positions(pattern_str, pattern, start_row, start_col, dr, dc)
                                    for thr in threat_pos:
                                        threats.add(thr)
                                        if player == self.current_player:  # 如果是当前玩家的威胁
                                            result[player]["threats"].append(thr)
        
        # 更新威胁位置集合
        self._threat_positions = threats
        
        # 最终评分计算
        # 我们不仅考虑棋型得分，还考虑位置重要性
        for player in [player_perspective, opponent]:
            # 添加位置重要性得分
            position_score = 0
            for i in range(self.size):
                for j in range(self.size):
                    if self.board[i, j] == player:
                        position_score += self._position_importance[i, j] * 10  # 根据位置重要性加权
            
            result[player]["score"] += position_score
            
            # 规范化得分
            result[player]["score"] *= self.SCORE_FACTOR
        
        return result
    
    def _find_pattern_matches(self, line_str: str, pattern: str) -> int:
        """计算一个模式在行中出现的次数"""
        count = 0
        for i in range(len(line_str) - len(pattern) + 1):
            if line_str[i:i+len(pattern)] == pattern:
                count += 1
        return count
    
    def _find_threat_positions(self, line_str: str, pattern: str, 
                              start_row: int, start_col: int, 
                              dr: int, dc: int) -> List[Tuple[int, int]]:
        """找出威胁位置（需要防守或可以进攻的位置）"""
        threat_positions = []
        
        for i in range(len(line_str) - len(pattern) + 1):
            if line_str[i:i+len(pattern)] == pattern:
                # 对于威胁，我们要找出能形成五连的空位
                # 例如，对于 "011110"，第0和第5位是威胁位置
                for j in range(len(pattern)):
                    if pattern[j] == '0':  # 空位
                        r = start_row + (i + j) * dr
                        c = start_col + (i + j) * dc
                        if 0 <= r < self.size and 0 <= c < self.size:
                            threat_positions.append((r, c))
        
        return threat_positions
    
    def get_score_for_move(self, row: int, col: int, player: int) -> float:
        """
        计算特定位置的移动分数
        使用增量评估而不是完全重新计算
        
        Args:
            row, col: 移动位置
            player: 玩家ID（1或2）
            
        Returns:
            float: 该位置的移动分数
        """
        # 检查位置是否合法
        if not (0 <= row < self.size and 0 <= col < self.size) or self.board[row, col] != 0:
            return float('-inf')
            
        # 检查缓存
        cache_key = (row, col, player)
        if cache_key in self._move_scores:
            return self._move_scores[cache_key]
        
        # 创建临时棋盘状态
        original_value = self.board[row, col]
        self.board[row, col] = player
        
        # 评估受这步影响的4个方向
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        total_score = 0
        
        for dr, dc in directions:
            # 在该方向上扫描11个位置（最多可能影响的范围）
            line = []
            for offset in range(-5, 6):
                r, c = row + dr * offset, col + dc * offset
                if 0 <= r < self.size and 0 <= c < self.size:
                    line.append(str(self.board[r, c]))
                else:
                    line.append('X')  # 超出边界
            
            line_str = ''.join(line)
            
            # 将对手棋子替换为'2'，将我方棋子替换为'1'，空位替换为'0'
            opponent = 3 - player
            pattern_str = line_str.replace(str(player), '1').replace(str(opponent), '2').replace('0', '0').replace('X', 'X')
            
            # 检查所有可能的棋型
            for pattern_name, (pattern, openness, score) in self.PATTERNS.items():
                matches = self._find_pattern_matches(pattern_str, pattern)
                if matches > 0:
                    # 根据开放度和玩家调整分数
                    adjusted_score = score
                    if player == self.current_player:
                        adjusted_score *= 1.1  # 略微偏好进攻
                    total_score += matches * adjusted_score
        
        # 额外考虑位置价值
        total_score += self._position_importance[row, col] * 5
        
        # 恢复原始棋盘状态
        self.board[row, col] = original_value
        
        # 缓存结果
        self._move_scores[cache_key] = total_score
        return total_score
    
    def detect_immediate_threat(self) -> Optional[int]:
        """
        检测紧急威胁，返回防守或获胜的位置
        
        Returns:
            Optional[int]: 一维威胁位置，如果没有紧急威胁则为None
        """
        # 先检查我方能否一步获胜
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 0:
                    # 临时落子
                    self.board[r, c] = self.current_player
                    if self._check_win(r, c):
                        # 找到获胜位置
                        self.board[r, c] = 0  # 恢复
                        return r * self.size + c
                    self.board[r, c] = 0  # 恢复
        
        # 检查对手能否一步获胜
        opponent = 3 - self.current_player
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] == 0:
                    # 模拟对手落子
                    self.board[r, c] = opponent
                    if self._check_win(r, c):
                        # 找到需要防守的位置
                        self.board[r, c] = 0  # 恢复
                        return r * self.size + c
                    self.board[r, c] = 0  # 恢复
        
        # 检查双活三和活四等复杂威胁
        eval_result = self.evaluate_position()
        if eval_result[self.current_player]["threats"]:
            # 找到进攻威胁位置
            r, c = eval_result[self.current_player]["threats"][0]
            return r * self.size + c
        
        # 检查对手的威胁位置
        eval_result = self.evaluate_position(opponent)
        if eval_result[opponent]["threats"]:
            # 找到防守威胁位置
            r, c = eval_result[opponent]["threats"][0]
            return r * self.size + c
        
        return None
    
    def get_zobrist_hash(self) -> int:
        """
        计算棋盘的Zobrist哈希值，用于快速状态比较
        
        Returns:
            int: Zobrist哈希值
        """
        # 使用固定随机数作为种子，确保哈希值的一致性
        if not hasattr(self, '_zobrist_table'):
            # 初始化Zobrist表（首次调用时）
            np.random.seed(42)
            self._zobrist_table = np.random.randint(0, 2**64, (3, self.size, self.size), dtype=np.uint64)
        
        # 计算哈希值
        h = 0
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    h ^= self._zobrist_table[self.board[r, c], r, c]
        
        return h
    
    def print_board(self):
        """打印当前棋盘状态到控制台"""
        symbols = {0: ".", 1: "X", 2: "O"}
        
        # 打印列索引
        print("   " + " ".join(str(i % 10) for i in range(self.size)))
        
        # 打印行并带有行索引
        for i, row in enumerate(self.board):
            print(f"{i:2d} " + " ".join(symbols[cell] for cell in row))
            
        # 打印玩家回合
        player_name = "黑方" if self.current_player == 1 else "白方"
        print(f"\n当前玩家: {player_name} ({symbols[self.current_player]})")
        
        # 打印游戏状态
        if self.game_over:
            if self.winner == 0:
                print("游戏以平局结束")
            else:
                winner_name = "黑方" if self.winner == 1 else "白方"
                print(f"游戏结束。胜者: {winner_name}")
    
    def get_symmetries(self, move: int) -> List[Tuple[np.ndarray, int]]:
        """
        获取当前棋盘和移动的所有对称变换
        用于数据增强和减少状态空间
        
        Args:
            move: 一维移动位置
            
        Returns:
            List[Tuple[board, move]]: 对称变换列表
        """
        row, col = move // self.size, move % self.size
        symmetries = []
        
        # 原始棋盘
        board = np.copy(self.board)
        
        # 旋转90°、180°、270°
        for i in range(4):
            # 旋转棋盘
            rot_board = np.rot90(board, i)
            # 计算旋转后的移动位置
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
            
            # 水平翻转
            flip_board = np.fliplr(rot_board)
            flip_col = self.size - 1 - new_col
            flip_move = new_row * self.size + flip_col
            symmetries.append((flip_board, flip_move))
        
        return symmetries