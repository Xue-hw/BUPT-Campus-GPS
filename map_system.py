import os
import heapq
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque

# --- 配置与环境 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NODES_PATH = os.path.join(BASE_DIR, 'campus_nodes.csv')
EDGES_PATH = os.path.join(BASE_DIR, 'campus_edges.csv')
IMG_PATH = os.path.join(BASE_DIR, 'map.jpg')

class CampusNavigationSystem:
    def __init__(self):
        """初始化导航系统，加载数据并构建图的底层结构"""
        self.load_data()
        self.build_graph()

    def load_data(self):
        """从CSV读取节点和边的数据"""
        if not os.path.exists(NODES_PATH) or not os.path.exists(EDGES_PATH):
            raise FileNotFoundError("找不到节点或边的数据文件，请确保先运行地图构建工具。")
        self.df_nodes = pd.read_csv(NODES_PATH, index_col='Node_ID')
        self.df_edges = pd.read_csv(EDGES_PATH)

    def build_graph(self):
        """
        构建邻接表用于算法遍历。
        结构: { node_id: { neighbor_id: distance, ... }, ... }
        """
        self.graph = {node_id: {} for node_id in self.df_nodes.index}
        
        # 校园路网视为无向图，需要双向添加边
        for _, row in self.df_edges.iterrows():
            u, v, dist = int(row['Start_ID']), int(row['End_ID']), float(row['Distance'])
            if u in self.graph and v in self.graph:
                self.graph[u][v] = dist
                self.graph[v][u] = dist  # 无向图

    def save_data(self):
        """将内存中的修改持久化保存到CSV文件"""
        self.df_nodes.to_csv(NODES_PATH, encoding='utf-8-sig')
        self.df_edges.to_csv(EDGES_PATH, index=False, encoding='utf-8-sig')
        print("✅ 数据已保存。")

    # ================= 核心要求 1: 图的修改功能 =================
    
    def add_node(self, node_id, name, n_type, x, y, px, py, parent):
        """添加新节点（建筑/路口）"""
        if node_id in self.df_nodes.index:
            print(f"❌ 节点ID {node_id} 已存在。")
            return
        self.df_nodes.loc[node_id] = [name, n_type, x, y, px, py, parent]
        self.build_graph() # 重建图结构
        print(f"✅ 成功添加节点: {name}")

    def delete_node(self, node_id):
        """删除节点及与其相关的边"""
        if node_id not in self.df_nodes.index:
            print(f"❌ 节点ID {node_id} 不存在。")
            return
        # 删除节点
        self.df_nodes.drop(node_id, inplace=True)
        # 删除相关的边
        self.df_edges = self.df_edges[(self.df_edges['Start_ID'] != node_id) & (self.df_edges['End_ID'] != node_id)]
        self.build_graph()
        print(f"✅ 成功删除节点 {node_id} 及其关联边。")

    def add_edge(self, u, v, distance):
        """添加新道路"""
        if u not in self.df_nodes.index or v not in self.df_nodes.index:
            print("❌ 边连接的节点不存在。")
            return
        new_edge = pd.DataFrame([{'Start_ID': u, 'End_ID': v, 'Distance': distance}])
        self.df_edges = pd.concat([self.df_edges, new_edge], ignore_index=True)
        self.build_graph()
        print(f"✅ 成功添加边: {u} <--> {v}, 距离: {distance}")

    def delete_edge(self, u, v):
        """删除道路"""
        mask = ~(((self.df_edges['Start_ID'] == u) & (self.df_edges['End_ID'] == v)) | 
                 ((self.df_edges['Start_ID'] == v) & (self.df_edges['End_ID'] == u)))
        self.df_edges = self.df_edges[mask]
        self.build_graph()
        print(f"✅ 成功尝试删除边: {u} <--> {v}")

    # ================= 核心要求 2: 最短路径算法 =================

    def shortest_path_dijkstra(self, start_id, end_id):
        """
        使用堆优化的 Dijkstra 算法寻找最短路径
        返回: (路径节点ID列表, 总距离)
        """
        if start_id not in self.graph or end_id not in self.graph:
            return None, float('inf')

        distances = {node: float('inf') for node in self.graph}
        distances[start_id] = 0
        previous_nodes = {node: None for node in self.graph}
        
        # 优先队列，存储 (当前最短距离, 节点ID)
        pq = [(0, start_id)]

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            # 如果弹出的距离大于已知最短距离，说明是冗余数据，直接跳过
            if current_dist > distances[current_node]:
                continue

            # 提前终止优化：如果已经找到了终点的最短路径，可以直接退出
            if current_node == end_id:
                break

            # 遍历邻居节点进行松弛操作
            for neighbor, weight in self.graph[current_node].items():
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))

        # 回溯构建路径
        path = []
        curr = end_id
        while curr is not None:
            path.append(curr)
            curr = previous_nodes[curr]
        
        path.reverse()
        
        # 验证是否真的连通
        if path[0] == start_id:
            return path, distances[end_id]
        else:
            return None, float('inf')

    # ================= 核心要求 3: 图的遍历 =================

    def traverse_campus_bfs(self, start_id):
        """广度优先遍历 (BFS) 校园建筑，可用于生成周围建筑推荐"""
        if start_id not in self.graph:
            return []

        visited = set()
        queue = deque([start_id])
        traversal_order = []

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                traversal_order.append(node)
                # 将未访问的邻居加入队列
                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return traversal_order

    # ================= 核心要求 4: 输出与可视化 =================

    def print_path_description(self, path, total_dist):
        """输出路径的文字描述"""
        if not path:
            print("❌ 无法找到连通的路径。")
            return
        
        print(f"\n🗺️ 导航路线规划成功！总距离约: {total_dist:.2f} 米")
        print(" -> ".join([self.df_nodes.loc[nid, 'Name'] for nid in path]))
        print("-" * 40)

    def draw_path(self, path):
        """在地图上绘制导航路线"""
        if not path:
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        try:
            ax.imshow(Image.open(IMG_PATH))
        except Exception as e:
            print(f"图片打开失败: {e}")
            return

        # 1. 绘制所有浅色底图元素（节点和边）
        for _, row in self.df_edges.iterrows():
            u, v = int(row['Start_ID']), int(row['End_ID'])
            if u in self.df_nodes.index and v in self.df_nodes.index:
                p1, p2 = self.df_nodes.loc[u], self.df_nodes.loc[v]
                ax.plot([p1['PX'], p2['PX']], [p1['PY'], p2['PY']], 'k-', linewidth=0.5, alpha=0.3)

        for nid, row in self.df_nodes.iterrows():
            ax.scatter(row['PX'], row['PY'], c='gray', s=10, alpha=0.5)

        # 2. 高亮绘制最短路径
        path_x = [self.df_nodes.loc[nid, 'PX'] for nid in path]
        path_y = [self.df_nodes.loc[nid, 'PY'] for nid in path]
        
        # 绘制红色粗线作为导航路径
        ax.plot(path_x, path_y, 'r-', linewidth=3.5, label='导航路线', zorder=5)
        
        # 标出起点和终点
        ax.scatter(path_x[0], path_y[0], c='green', s=100, marker='*', label='起点', zorder=6)
        ax.scatter(path_x[-1], path_y[-1], c='blue', s=100, marker='*', label='终点', zorder=6)

        ax.legend()
        plt.title("校园导航系统规划路线")
        plt.axis('off')
        plt.show()

    # ================= 拓展功能: 多入口建筑名称导航 =================

    def get_node_ids_by_name(self, name):
        """
        名称解析器：根据输入的名称获取对应的节点 ID 列表。
        优先查找 Parent 为该名称且 Type 为 Door 的所有入口；
        若没有对应的入口，则退化为查找 Name 完全匹配的建筑或路口节点。
        """
        # 1. 优先查找属于该建筑的所有门
        doors = self.df_nodes[(self.df_nodes['Parent'] == name) & (self.df_nodes['Type'] == 'Door')]
        if not doors.empty:
            return doors.index.tolist()
        
        # 2. 如果没有门（或者是路口/无门建筑），精确匹配名称
        exact_match = self.df_nodes[self.df_nodes['Name'] == name]
        if not exact_match.empty:
            return exact_match.index.tolist()
            
        return []

    def shortest_path_dijkstra_multi_target(self, start_id, target_ids):
        """
        多目标 Dijkstra 算法。
        参数:
            start_id: 起点节点 ID
            target_ids: 终点节点 ID 集合 (如某个建筑的多个门)
        返回:
            (最优路径列表, 最短距离)
        """
        if start_id not in self.graph or not target_ids:
            return None, float('inf')

        target_set = set(target_ids)
        # 如果起点本身就在目标集合中，直接到达
        if start_id in target_set:
            return [start_id], 0.0

        distances = {node: float('inf') for node in self.graph}
        distances[start_id] = 0
        previous_nodes = {node: None for node in self.graph}
        
        # 优先队列 (距离, 节点ID)
        pq = [(0, start_id)]
        final_target_id = None  # 记录最终实际到达的是哪个门

        while pq:
            current_dist, current_node = heapq.heappop(pq)

            if current_dist > distances[current_node]:
                continue

            # 【核心修改】：如果当前节点在目标集合中，说明找到了最近的入口，立即终止算法
            if current_node in target_set:
                final_target_id = current_node
                break

            for neighbor, weight in self.graph[current_node].items():
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))

        # 无法到达任何一个目标节点
        if final_target_id is None:
            return None, float('inf')

        # 回溯构建路径
        path = []
        curr = final_target_id
        while curr is not None:
            path.append(curr)
            curr = previous_nodes[curr]
        
        path.reverse()
        return path, distances[final_target_id]

    def navigate_by_name(self, start_name, end_name):
        """
        高度封装的导航包装函数 (Wrapper)。
        支持直接输入起终点名称（如 "南门" -> "校医院"），自动处理多门选优。
        """
        start_ids = self.get_node_ids_by_name(start_name)
        end_ids = self.get_node_ids_by_name(end_name)

        if not start_ids:
            print(f"❌ 找不到与 '{start_name}' 相关的节点或入口。")
            return None, float('inf')
        if not end_ids:
            print(f"❌ 找不到与 '{end_name}' 相关的节点或入口。")
            return None, float('inf')

        best_path = None
        min_dist = float('inf')

        # 处理起点也是多入口的情况（例如从“教一”走到“教三”）
        # 遍历起点所有的门，分别计算到终点集合的最短距离，取全局最优
        for s_id in start_ids:
            path, dist = self.shortest_path_dijkstra_multi_target(s_id, end_ids)
            if dist < min_dist:
                min_dist = dist
                best_path = path

        return best_path, min_dist

# --- 测试与交互示例 ---
if __name__ == "__main__":
    nav = CampusNavigationSystem()
    
    # 假设你要从南门去校医院（如果校医院在CSV中有对应的门节点，会自动寻路到最近的门）
    START_NAME = "南门"
    END_NAME = "校医院"

    print(f"=== 请输入起点和目的地 ===")
    START_NAME = input("请输入起点名称：")
    END_NAME = input("请输入终点名称：")

    print(f"=== 智能导航：正在计算从【{START_NAME}】到【{END_NAME}】的最短路线 ===")
    
    path, dist = nav.navigate_by_name(START_NAME, END_NAME)
    
    if path:
        # 输出起点门和终点门的具体名称
        actual_start = nav.df_nodes.loc[path[0], 'Name']
        actual_end = nav.df_nodes.loc[path[-1], 'Name']
        print(f"📍 实际规划起止点：{actual_start} -> {actual_end}")
        
        nav.print_path_description(path, dist/2)
        nav.draw_path(path)
    else:
        print("导航失败。")

    # --- 拓展：演示图的遍历 ---
    # print("\n=== BFS 遍历周边节点（前10个） ===")
    # bfs_result = nav.traverse_campus_bfs(START_NAME)
    # for nid in bfs_result[:10]:
    #     print(f"[{nid}] {nav.df_nodes.loc[nid, 'Name']}")