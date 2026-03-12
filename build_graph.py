#穿楼导航和部分地标性建筑暂未实现
#路网逻辑：
#对于建筑物之间的连线，考虑直线距离和实际道路，有略作修改（如教四新增了北门）
#对于有标识入口的建筑物，以导航到其任意入口视为导航到该建筑物；
#对于没有标识入口的建筑物，以导航到其节点坐标视为导航到该建筑物；
#路网连线为直线距离且未对齐，存在一定误差，请以实际位置和路线为准
#当前路网视为无向图，后续可根据实际情况添加单向边（如单行道、禁止左转等）

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import numpy as np

# --- 1. 配置与颜色定义 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NODES_PATH = os.path.join(BASE_DIR, 'campus_nodes.csv')
EDGES_PATH = os.path.join(BASE_DIR, 'campus_edges.csv')
IMG_PATH = os.path.join(BASE_DIR, 'map.jpg')

# 颜色配置表
TYPE_COLORS = {
    'Building': 'red',      # 建筑 - 红色
    'Door': 'green',        # 门 - 绿色
    'Intersection': 'blue'  # 路口 - 蓝色
}

class EdgeInteractiveTool:
    def __init__(self):
        # 加载数据
        self.df_nodes = pd.read_csv(NODES_PATH).set_index('Node_ID')
        if os.path.exists(EDGES_PATH):
            self.df_edges = pd.read_csv(EDGES_PATH)
        else:
            self.df_edges = pd.DataFrame(columns=['Start_ID', 'End_ID', 'Distance'])
        
        self.selected_nodes = [] 
        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        self.plot_lines = [] 

    def get_dist(self, id1, id2):
        n1, n2 = self.df_nodes.loc[id1], self.df_nodes.loc[id2]
        return round(np.sqrt((n1['X'] - n2['X'])**2 + (n1['Y'] - n2['Y'])**2), 2)

    def add_edge(self, id1, id2):
        id1, id2 = int(id1), int(id2)
        if id1 == id2: return
        
        u, v = min(id1, id2), max(id1, id2)
        
        if not self.df_edges[(self.df_edges['Start_ID'] == u) & (self.df_edges['End_ID'] == v)].empty:
            print(f"⚠️ 边 {u}-{v} 已存在")
            return

        dist = self.get_dist(u, v)
        new_row = pd.DataFrame([{'Start_ID': u, 'End_ID': v, 'Distance': dist}])
        self.df_edges = pd.concat([self.df_edges, new_row], ignore_index=True)
        
        # 绘制黑色连线
        n1, n2 = self.df_nodes.loc[u], self.df_nodes.loc[v]
        line, = self.ax.plot([n1['PX'], n2['PX']], [n1['PY'], n2['PY']], 'k-', linewidth=1.2, alpha=0.6)
        self.plot_lines.append(line)
        self.fig.canvas.draw()
        print(f"✅ 已连接: {self.df_nodes.loc[u]['Name']} <-> {self.df_nodes.loc[v]['Name']} ({dist/2}m)")
        self.save_data()

    def on_click(self, event):
        if event.button == 1 and event.xdata is not None:
            # 搜索最近节点
            dists = np.sqrt((self.df_nodes['PX'] - event.xdata)**2 + (self.df_nodes['PY'] - event.ydata)**2)
            nearest_id = dists.idxmin()
            
            if dists[nearest_id] < 15: # 减小点击半径，防止误触
                self.selected_nodes.append(nearest_id)
                node_info = self.df_nodes.loc[nearest_id]
                print(f"📍 选中: {node_info['Name']} (ID: {nearest_id}, 类型: {node_info['Type']})")
                
                if len(self.selected_nodes) == 2:
                    self.add_edge(self.selected_nodes[0], self.selected_nodes[1])
                    self.selected_nodes = []

    def manual_input(self):
        print("\n" + "="*30)
        print("--- 手动连线模式 ---")
        print("输入格式: ID1 ID2 (例如: 244 248)")
        print("输入 'q' 退出并保存")
        print("="*30)
        while True:
            inp = input(">> ").strip().lower()
            if inp == 'q': break
            try:
                ids = list(map(int, inp.split()))
                if len(ids) == 2:
                    self.add_edge(ids[0], ids[1])
                else:
                    print("❌ 请输入两个有效的 ID")
            except Exception as e:
                print(f"❌ 输入错误: {e}")

    def save_data(self):
        self.df_edges.to_csv(EDGES_PATH, index=False, encoding='utf-8-sig')

    def run(self):
        self.ax.imshow(Image.open(IMG_PATH))
        
        # --- 按类别分色打印节点 ---
        for n_type, color in TYPE_COLORS.items():
            subset = self.df_nodes[self.df_nodes['Type'] == n_type]
            self.ax.scatter(subset['PX'], subset['PY'], c=color, s=15, label=n_type, edgecolors='white', linewidths=0.5, zorder=5)
        
        # 绘制已有连线（黑色）
        print(f"正在恢复 {len(self.df_edges)} 条连线...")
        for _, row in self.df_edges.iterrows():
            u, v = int(row['Start_ID']), int(row['End_ID'])
            if u in self.df_nodes.index and v in self.df_nodes.index:
                p1, p2 = self.df_nodes.loc[u], self.df_nodes.loc[v]
                self.ax.plot([p1['PX'], p2['PX']], [p1['PY'], p2['PY']], 'k-', linewidth=0.8, alpha=0.4, zorder=4)

        self.ax.legend(loc='upper right')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.title("BUPT Campus Topology Builder (Click 2 points to link)")
        plt.show(block=False) # 不阻塞，允许终端输入
        self.manual_input()
        plt.close()

if __name__ == "__main__":
    if os.path.exists(NODES_PATH) and os.path.exists(IMG_PATH):
        tool = EdgeInteractiveTool()
        tool.run()
    else:
        print("❌ 错误：请确保当前目录下有 campus_nodes.csv 和 map.jpg")