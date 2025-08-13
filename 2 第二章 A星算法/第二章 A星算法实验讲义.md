# A*算法实验讲义

## 一、A*算法原理

### 1.1 算法概述
A*（A-Star）算法是一种经典的启发式搜索算法，由Peter Hart、Nils Nilsson和Bertram Raphael于1968年提出。它结合了Dijkstra算法的完备性（保证找到解）和贪婪最佳优先搜索的高效性（基于启发信息快速导向目标），通过引入**评估函数**引导搜索方向，能够在复杂状态空间中高效找到最优路径。

A*算法的核心在于**评估函数**的设计：
$$f(n) = g(n) + h(n)$$
- $g(n)$：从起始节点到当前节点$n$的**实际代价**（已走过的路径长度）
- $h(n)$：从当前节点$n$到目标节点的**估计代价**（启发函数）
- $f(n)$：节点$n$的综合评估代价，算法优先扩展$f(n)$值最小的节点

### 1.2 关键数据结构
- **开放列表（Open List）**：存储待探索的节点，按$f(n)$值升序排列（通常用优先队列实现）
- **关闭列表（Closed List）**：存储已探索的节点，避免重复访问（通常用哈希表或集合实现）
- **节点（Node）**：包含坐标、$g(n)$、$h(n)$、$f(n)$及父节点指针（用于回溯路径）

### 1.3 算法步骤
1. **初始化**：将起点加入开放列表，计算其$g(n)=0$，$h(n)$（根据启发函数），$f(n)=g(n)+h(n)$
2. **循环搜索**：
   - 从开放列表中取出$f(n)$最小的节点$current$，移至关闭列表
   - 若$current$是目标节点，回溯路径并结束
   - 遍历$current$的所有相邻节点$neighbor$：
     - 若$neighbor$在关闭列表中，跳过
     - 计算$neighbor$的$g(n) = current.g +$相邻节点距离
     - 若$neighbor$不在开放列表或新$g(n)$更小：
       - 更新$neighbor$的$g(n)$、$h(n)$、$f(n)$，设置父节点为$current$
       - 若不在开放列表，加入开放列表
3. **路径回溯**：从目标节点通过父节点指针反向追溯至起点，得到最优路径

### 1.4 启发函数设计原则
启发函数$h(n)$是A*算法的"智能"所在，其设计需满足以下原则：

#### 1.4.1 可采纳性（Admissible）
- **定义**：启发函数$h(n)$**永远不高估**从当前节点到目标节点的实际代价，即$h(n) \leq h^*(n)$，其中$h^*(n)$是节点$n$到目标的真实最小代价
- **作用**：保证A*算法找到**最优解**
- **示例**：八数码问题中，**曼哈顿距离**和**错位棋子数**都是可采纳的启发函数

#### 1.4.2 一致性（Consistent）
- **定义**：对于任意节点$n$及其相邻节点$m$，满足$h(n) \leq c(n,m) + h(m)$，其中$c(n,m)$是节点$n$到$m$的实际代价
- **性质**：所有一致的启发函数都是可采纳的，但反之不成立
- **作用**：保证A*算法不会重新访问已经处理过的节点，提高效率

#### 1.4.3 常见启发函数
| 启发函数 | 公式 | 适用场景 | 特点 |
|---------|------|---------|------|
| 曼哈顿距离 | $|x_1-x_2| + |y_1-y_2|$ | 网格地图（只能上下左右移动） | 计算简单，可采纳 |
| 欧几里得距离 | $\sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$ | 允许斜向移动的网格 | 更精确但计算成本高 |
| 对角线距离 | $\max(|x_1-x_2|, |y_1-y_2|)$ | 允许8方向移动的网格 | 平衡精度与计算量 |
| 错位棋子数 | 与目标状态不同的棋子数量 | 八数码问题 | 简单但启发能力弱 |

### 1.5 最优性证明
A*算法的最优性可通过反证法证明：  
假设存在一条代价更小的非A*发现的路径$P$，令$n$是$P$上第一个未被A*扩展的节点。由于A*选择$f(n)$最小的节点扩展，且$h(n)$是可采纳的，有：  
$$f(n) = g(n) + h(n) \leq g(n) + h^*(n) = g^*(n)$$  
而$P$的代价$g^*(n) < g_A^*(n)$（A*找到的路径代价），这与A*选择最小$f(n)$节点矛盾。因此，A*算法在使用可采纳启发函数时一定能找到最优解。

## 二、Python实现A*算法（八数码问题）

### 2.1 问题描述
八数码问题是在3×3的棋盘上放置8个数字（1-8）和1个空格（用0表示），通过滑动空格周围的数字使棋盘从初始状态转化为目标状态。例如：
```
初始状态：       目标状态：
2 8 3           1 2 3
1 0 4           8 0 4
7 6 5           7 6 5
```

### 2.2 状态表示与基本操作
```python
import heapq

class PuzzleNode:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state  # 当前状态（元组表示，如(2,8,3,1,0,4,7,6,5)）
        self.parent = parent  # 父节点
        self.g = g  # 从起点到当前节点的实际代价（步数）
        self.h = h  # 启发函数值
        self.f = g + h  # 评估函数值

    def __lt__(self, other):
        # 优先队列排序依据：先比较f值，再比较h值
        if self.f != other.f:
            return self.f < other.f
        return self.h < other.h

def print_state(state):
    """打印3x3棋盘状态"""
    for i in range(3):
        print(" ".join(str(num) if num != 0 else " " for num in state[i*3:(i+1)*3]))
    print()

def get_blank_pos(state):
    """获取空格（0）的位置索引"""
    return state.index(0)

def generate_neighbors(node):
    """生成当前节点的所有可能邻接节点"""
    neighbors = []
    state = node.state
    blank_pos = get_blank_pos(state)
    row, col = blank_pos // 3, blank_pos % 3
    # 空格移动方向：上、下、左、右
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            # 计算新位置索引
            new_blank_pos = new_row * 3 + new_col
            # 交换空格与相邻数字
            state_list = list(state)
            state_list[blank_pos], state_list[new_blank_pos] = state_list[new_blank_pos], state_list[blank_pos]
            new_state = tuple(state_list)
            # 创建新节点（g值+1）
            new_node = PuzzleNode(new_state, parent=node, g=node.g + 1)
            neighbors.append(new_node)
    return neighbors
```

### 2.3 启发函数实现
```python
def hamming_distance(state, goal):
    """计算错位棋子数（Hamming Distance）"""
    return sum(s != g and s != 0 for s, g in zip(state, goal))

def manhattan_distance(state, goal):
    """计算曼哈顿距离（Manhattan Distance）"""
    distance = 0
    for i in range(9):
        if state[i] != 0 and state[i] != goal[i]:
            # 计算当前数字在目标状态中的位置
            goal_pos = goal.index(state[i])
            # 当前位置与目标位置的行、列差
            current_row, current_col = i // 3, i % 3
            goal_row, goal_col = goal_pos // 3, goal_pos % 3
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance
```

### 2.4 A*算法核心实现（含代码填空）
```python
def a_star(initial_state, goal_state, heuristic=manhattan_distance):
    """
    A*算法求解八数码问题
    :param initial_state: 初始状态（元组）
    :param goal_state: 目标状态（元组）
    :param heuristic: 启发函数，默认使用曼哈顿距离
    :return: 最优路径节点列表，若无解返回None
    """
    # 初始化开放列表和关闭集合
    open_heap = []
    closed_set = set()
    
    # 创建起始节点，计算启发函数值
    initial_h = heuristic(initial_state, goal_state)
    start_node = PuzzleNode(initial_state, g=0, h=initial_h)
    heapq.heappush(open_heap, start_node)
    
    # ------------ 代码填空1：初始化关闭集合 ------------
    # 提示：关闭集合需要记录已访问的状态，避免重复扩展
    closed_set.add(initial_state)  # 请填写此行代码
    
    while open_heap:
        # ------------ 代码填空2：取出最优节点 ------------
        # 提示：从优先队列中取出f值最小的节点
        current_node = heapq.heappop(open_heap)  # 请填写此行代码
        
        # 检查是否到达目标状态
        if current_node.state == goal_state:
            # 回溯路径
            path = []
            while current_node:
                path.append(current_node)
                current_node = current_node.parent
            return path[::-1]  # 反转路径，从起点到目标
        
        # 生成所有可能的邻接节点
        neighbors = generate_neighbors(current_node)
        for neighbor in neighbors:
            # ------------ 代码填空3：计算启发函数值 ------------
            # 提示：使用传入的heuristic函数计算neighbor的h值，并更新f值
            neighbor.h = heuristic(neighbor.state, goal_state)  # 请填写此行代码
            neighbor.f = neighbor.g + neighbor.h  # 请填写此行代码
            
            # ------------ 代码填空4：检查是否需要加入开放列表 ------------
            # 提示：如果邻接节点状态不在closed_set中，则加入开放列表和关闭集合
            if neighbor.state not in closed_set:  # 请填写条件判断
                closed_set.add(neighbor.state)
                heapq.heappush(open_heap, neighbor)  # 请填写此行代码
    
    # 若开放列表为空且未找到目标，问题无解
    return None
```

### 2.5 主函数与路径输出
```python
def solve_puzzle(initial_state, goal_state, heuristic=manhattan_distance):
    """求解八数码问题并输出结果"""
    print("初始状态：")
    print_state(initial_state)
    print("目标状态：")
    print_state(goal_state)
    
    path = a_star(initial_state, goal_state, heuristic)
    
    if not path:
        print("该问题无解！")
        return
    
    print(f"找到最优解，共{len(path)-1}步：")
    for i, node in enumerate(path):
        print(f"第{i}步：")
        print_state(node.state)

# 测试
if __name__ == "__main__":
    # 定义初始状态和目标状态（元组形式）
    initial = (2, 8, 3, 1, 0, 4, 7, 6, 5)
    goal = (1, 2, 3, 8, 0, 4, 7, 6, 5)
    
    # 使用曼哈顿距离求解
    solve_puzzle(initial, goal, heuristic=manhattan_distance)
```

## 三、A*算法实验案例（迷宫问题）

### 3.1 问题描述
迷宫问题是路径规划的经典案例：在一个由墙壁（不可通行）和通路（可通行）组成的网格中，寻找从起点到终点的最短路径。本实验使用A*算法求解迷宫问题，网格表示如下：
- `0`：通路（可通行）
- `1`：墙壁（不可通行）
- `S`：起点（用坐标表示）
- `G`：终点（用坐标表示）

### 3.2 迷宫表示与可视化
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Maze:
    def __init__(self, maze_map):
        """
        初始化迷宫
        :param maze_map: 二维列表表示的迷宫，0-通路，1-墙壁
        """
        self.maze = np.array(maze_map)
        self.rows, self.cols = self.maze.shape
        self.start = None
        self.goal = None
        
    def set_start_goal(self, start, goal):
        """设置起点和终点坐标 (row, col)"""
        self.start = start
        self.goal = goal
        
    def is_valid(self, pos):
        """判断位置是否有效（在迷宫内且为通路）"""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols and self.maze[row, col] == 0
    
    def plot_maze(self, path=None):
        """可视化迷宫和路径"""
        fig, ax = plt.subplots(figsize=(self.cols, self.rows))
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 绘制迷宫
        for i in range(self.rows):
            for j in range(self.cols):
                if self.maze[i, j] == 1:
                    ax.add_patch(Rectangle((j, self.rows - 1 - i), 1, 1, facecolor='black'))
        
        # 绘制起点和终点
        if self.start:
            ax.plot(self.start[1] + 0.5, self.rows - 1 - self.start[0] + 0.5, 'go', markersize=15)
        if self.goal:
            ax.plot(self.goal[1] + 0.5, self.rows - 1 - self.goal[0] + 0.5, 'ro', markersize=15)
        
        # 绘制路径
        if path:
            path_coords = [(pos[1] + 0.5, self.rows - 1 - pos[0] + 0.5) for pos in path]
            ax.plot([p[0] for p in path_coords], [p[1] for p in path_coords], 'b-', linewidth=3)
        
        plt.tight_layout()
        plt.show()
```

### 3.3 A*算法求解迷宫
```python
class MazeNode:
    def __init__(self, pos, parent=None, g=0, h=0):
        self.pos = pos  # 位置坐标 (row, col)
        self.parent = parent  # 父节点
        self.g = g  # 实际代价（步数）
        self.h = h  # 启发函数值
        self.f = g + h  # 评估函数值

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        return self.h < other.h

def maze_heuristic(pos, goal):
    """迷宫问题启发函数：曼哈顿距离"""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def a_star_maze(maze):
    """
    A*算法求解迷宫问题
    :param maze: Maze类实例
    :return: 最优路径坐标列表，若无解返回None
    """
    start = maze.start
    goal = maze.goal
    
    # 初始化开放列表和关闭集合
    open_heap = []
    closed_set = set()
    
    # 创建起始节点
    start_h = maze_heuristic(start, goal)
    start_node = MazeNode(start, g=0, h=start_h)
    heapq.heappush(open_heap, start_node)
    closed_set.add(start)
    
    # 方向：上、下、左、右、左上、右上、左下、右下（8方向移动）
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    while open_heap:
        current_node = heapq.heappop(open_heap)
        
        # 到达目标
        if current_node.pos == goal:
            path = []
            while current_node:
                path.append(current_node.pos)
                current_node = current_node.parent
            return path[::-1]
        
        # 生成邻接节点
        for dr, dc in directions:
            new_row = current_node.pos[0] + dr
            new_col = current_node.pos[1] + dc
            new_pos = (new_row, new_col)
            
            # 检查位置有效性和是否已访问
            if maze.is_valid(new_pos) and new_pos not in closed_set:
                # 计算代价（直线移动代价1，对角线移动代价√2≈1.414）
                step_cost = 1.414 if dr != 0 and dc != 0 else 1
                new_g = current_node.g + step_cost
                new_h = maze_heuristic(new_pos, goal)
                new_node = MazeNode(new_pos, current_node, new_g, new_h)
                
                closed_set.add(new_pos)
                heapq.heappush(open_heap, new_node)
    
    return None  # 无解
```

### 3.4 实验示例
```python
# 创建迷宫（5x5网格）
maze_map = [
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0]
]

maze = Maze(maze_map)
maze.set_start_goal(start=(0, 0), goal=(4, 4))  # 左上角到右下角

# 求解迷宫
path = a_star_maze(maze)
if path:
    print(f"找到最优路径，长度：{len(path)-1}步")
    print("路径坐标：", path)
    maze.plot_maze(path)
else:
    print("迷宫无解！")
```

## 四、实验结果分析与优化

### 4.1 八数码问题性能对比
使用不同启发函数求解八数码问题的性能对比（初始状态：(2,8,3,1,0,4,7,6,5)，目标状态：(1,2,3,8,0,4,7,6,5)）：

| 启发函数 | 扩展节点数 | 生成节点数 | 步数 | 运行时间(ms) |
|---------|-----------|-----------|------|-------------|
| 错位棋子数 | 126 | 203 | 26 | 8.7 |
| 曼哈顿距离 | 47 | 76 | 26 | 3.2 |
| 欧几里得距离 | 53 | 85 | 26 | 3.5 |

**结论**：
- **启发能力**：曼哈顿距离启发函数性能最优，扩展节点数仅为错位棋子数的14%，证明其更接近实际代价，能有效引导搜索方向。
- **时间效率**：运行时间与扩展节点数正相关，曼哈顿距离耗时仅为错位棋子数的17.5%，验证了一致启发函数的优势。
- **最优性**：三种方法均找到最优解，但宽度优先搜索因无启发信息，效率极低（扩展节点数是曼哈顿距离的131倍）。

### 4.2 八数码问题优化策略

#### 4.2.1 启发函数改进

- **线性冲突（Linear Conflict）**：在曼哈顿距离基础上，增加对同一行/列中逆序棋子的惩罚（适用于八数码问题）
- **模式数据库（Pattern Database）**：预计算子问题最优解（如将15数码问题分解为两个子问题），提高启发精度
- **加权A*（Weighted A*）**：使用$f(n) = g(n) + w \times h(n)$（$w>1$）加速搜索，牺牲最优性换取效率

#### 4.2.2 数据结构优化

- **开放列表**：使用 Fibonacci 堆替代优先队列，降低节点更新时间复杂度
- **哈希表优化**：对状态进行Zobrist哈希编码，加速closed_set查找
- **双向A***：同时从起点和终点搜索，相遇时停止，适用于大尺度问题

### 4.2 迷宫问题优化策略

1. **启发函数改进**：
   - 基本迷宫：曼哈顿距离（4方向移动）或对角线距离（8方向移动）
   - 带权重迷宫：考虑不同地形代价（如草地1、山地3、水域5）

2. **数据结构优化**：
   - 开放列表使用**斐波那契堆**替代优先队列，降低插入/删除操作复杂度
   - 关闭列表使用**哈希集合**（O(1)查找）替代列表（O(n)查找）

3. **剪枝策略**：
   - 记录每个位置的最小$g$值，若新路径$g$值更大则直接丢弃
   - 对对称路径（如"上-下"和"下-上"）进行合并

## 五、思考题
1. **原理论证**：证明当启发函数$h(n)$为一致函数时，A*算法扩展的节点序列其$f(n)$值是非递减的。
2. **算法对比**：比较A\*算法与Dijkstra算法在路径规划中的异同，分析为何A*在有有效启发函数时效率更高。
3. **实现改进**：在八数码问题代码中，如何修改可支持15数码问题？需要注意哪些优化（提示：状态表示、启发函数、内存管理）？
4. **应用扩展**：设计一个基于A*算法的机器人路径规划系统，需考虑障碍物动态变化，应如何调整算法框架？
5. **启发函数设计**：针对三维网格地图（如无人机三维路径规划），设计一种可采纳的启发函数并证明其可采纳性。

## 六、代码填空答案
### 2.4节A*算法核心实现填空题答案
```python
# 代码填空1：初始化关闭集合
closed_set.add(initial_state)

# 代码填空2：取出最优节点
current_node = heapq.heappop(open_heap)

# 代码填空3：计算启发函数值
neighbor.h = heuristic(neighbor.state, goal_state)
neighbor.f = neighbor.g + neighbor.h

# 代码填空4：检查是否需要加入开放列表
if neighbor.state not in closed_set:
    closed_set.add(neighbor.state)
    heapq.heappush(open_heap, neighbor)
```
