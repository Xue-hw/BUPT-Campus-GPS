# 由于是手动选择坐标，可能会有误差，请根据实际情况进行调整。
# 坐标位置以地图左下角为基准，1单位约等于0.5m，请在采集时注意保持一致性。
# 由于地图经过偏移，无法准确代表真实情况，请以实际环境为准进行调整。
# 物美与眼镜店简化为单一坐标建筑
# 小松林、时光广场等位置由于是开放空间，没有明确边界和入口，故未进行采集
# 家属区建筑在地图上未做命名处理，在程序中难以区分，同时家属区道路情况复杂，故未进行采集
# 由于时间和空间问题，道路绿化、可踩踏的草坪小径等暂未标注。
# 后续优化方向：对齐门、路口的位置，尽量做到横平竖直
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import numpy as np

# --- 1. 环境配置与中文支持 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(BASE_DIR, 'map.jpg')
SAVE_PATH = os.path.join(BASE_DIR, 'campus_nodes.csv')

# 比例尺：1像素 = 0.5米
SCALE = 1

# 全局变量定义
ORIGIN_PX, ORIGIN_PY = None, None 
COLUMNS = ['Node_ID', 'Name', 'Type', 'X', 'Y', 'PX', 'PY', 'Parent']
plot_elements = {}  # 格式: { Node_ID/ 'ORIGIN': (plot_obj, text_obj) }

# --- 2. 数据加载函数 ---
def load_existing_data():
    """从 CSV 加载旧数据，并尝试恢复参考原点"""
    global ORIGIN_PX, ORIGIN_PY
    if os.path.exists(SAVE_PATH):
        try:
            df = pd.read_csv(SAVE_PATH, encoding='utf-8-sig')
            # 必须检查 PX 列是否存在，避免旧版本格式导致崩溃
            if 'PX' in df.columns and 'PY' in df.columns:
                print(f"✅ 已成功加载 {len(df)} 个旧节点。")
                # 默认以 BUPT 经典原点为例，或者你可以根据需求自定义逻辑
                ORIGIN_PX, ORIGIN_PY = 135,1685 
                return df
            else:
                print("⚠️ 发现旧版 CSV 格式不兼容（缺少像素坐标），将重新创建。")
        except Exception as e:
            print(f"读取数据失败: {e}")
    
    # 返回定义好列名的空 DataFrame，避免 FutureWarning
    return pd.DataFrame({c: pd.Series(dtype='object') for c in COLUMNS}).dropna()

# 初始化 DataFrame
df_nodes = load_existing_data()

# --- 3. 交互逻辑函数 ---
def on_click(event):
    global df_nodes, ORIGIN_PX, ORIGIN_PY
    
    # 仅响应左键点击
    if event.button == 1 and event.xdata is not None and event.ydata is not None:
        px, py = event.xdata, event.ydata

        # --- A. 原点设置逻辑 ---
        # 如果当前没有任何数据且原点未设置，则第一次点击定义为原点
        if df_nodes.empty and ORIGIN_PX is None:
            ORIGIN_PX, ORIGIN_PY = px, py
            print(f"\n📍 原点已设定！像素坐标: ({int(px)}, {int(py)})")
            p_ori, = plt.plot(px, py, 'kx', markersize=12, mew=2)
            t_ori = plt.text(px, py, "  ORIGIN (0,0)", color='red', fontweight='bold')
            plot_elements['ORIGIN'] = (p_ori, t_ori)
            plt.draw()
            return 

        # --- B. 检查是否点击了已有点 ---
        for idx, row in df_nodes.iterrows():
            dist = np.sqrt((px - row['PX'])**2 + (py - row['PY'])**2)
            if dist < 10: # 10像素为命中范围
                print(f"\n🔎 选中节点 [ID: {row['Node_ID']}] 名称: {row['Name']}")
                return

        # --- C. 正常采集逻辑 ---
        if ORIGIN_PX is None:
            print("❌ 错误：请先点击地图左下角设置参考原点！")
            return

        # 换算物理坐标
        dx = px - ORIGIN_PX
        dy = ORIGIN_PY - py
        real_x, real_y = round(dx * SCALE, 2), round(dy * SCALE, 2)

        print(f"\n--- 拾取新点: X={real_x}m, Y={real_y}m ---")
        choice = input("选择类型: 1-建筑, 2-门/入口, 3-路口 (输入 Z 撤销): ").strip().lower()
        
        if choice == 'z':
            undo_last()
            return

        # 录入具体名称
        node_id = int(df_nodes['Node_ID'].max() + 1) if not df_nodes.empty else 1
        if choice == '1':
            name = input("请输入【建筑名称】: ").strip()
            type_str, parent = "Building", name
        elif choice == '2':
            parent = input("所属建筑名称: ").strip()
            name = f"{parent}_{input('门名称(如南门): ').strip()}"
            type_str = "Door"
        else:
            name = input("路口/点位名称: ").strip()
            type_str, parent = "Intersection", "None"

        # 使用 Pandas 合并数据
        new_row = pd.DataFrame([{
            'Node_ID': node_id, 'Name': name, 'Type': type_str, 
            'X': real_x, 'Y': real_y, 'PX': px, 'PY': py, 'Parent': parent
        }])
        df_nodes = pd.concat([df_nodes, new_row], ignore_index=True)

        # 绘制标记
        color = 'ro' if choice == '1' else ('go' if choice == '2' else 'bo')
        p, = plt.plot(px, py, color, markersize=3)
        t = plt.text(px, py, f" {name}", fontsize=5, va='bottom')
        plot_elements[node_id] = (p, t)
        plt.draw()
        print(f"✅ 已记录: {name}")

def undo_last(event=None):
    """撤销功能：支持键盘 Z 键触发"""
    global df_nodes, ORIGIN_PX, ORIGIN_PY
    if event is not None and event.key != 'z':
        return

    # 情况 1: 撤销普通节点
    if not df_nodes.empty:
        last_idx = df_nodes.index[-1]
        last_id = df_nodes.at[last_idx, 'Node_ID']
        print(f"⏪ 撤销节点: {df_nodes.at[last_idx, 'Name']}")
        
        if last_id in plot_elements:
            p, t = plot_elements.pop(last_id)
            p.remove()
            t.remove()
        
        df_nodes = df_nodes.drop(last_idx)
        plt.draw()
    
    # 情况 2: 撤销原点设置
    elif 'ORIGIN' in plot_elements:
        print("⏪ 撤销原点设置")
        p, t = plot_elements.pop('ORIGIN')
        p.remove()
        t.remove()
        ORIGIN_PX, ORIGIN_PY = None, None
        plt.draw()
    else:
        print("⚠️ 已经没有可以撤销的内容了。")

def save_to_csv():
    """保存数据"""
    if not df_nodes.empty:
        df_nodes.to_csv(SAVE_PATH, index=False, encoding='utf-8-sig')
        print(f"\n💾 数据已成功保存至: {SAVE_PATH}")
    else:
        # 如果当前没数据但有文件，不进行覆盖，保护旧数据
        print("\n👋 本次未采集新数据。")

# --- 4. 程序入口 ---
if __name__ == "__main__":
    if not os.path.exists(IMG_PATH):
        print(f"❌ 找不到地图文件: {IMG_PATH}\n请确保图片命名为 map.jpg 并放在脚本目录下。")
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        try:
            img = Image.open(IMG_PATH)
            ax.imshow(img)
        except Exception as e:
            print(f"图片打开失败: {e}")
            exit()

        # 重绘旧节点（如果有）
        if not df_nodes.empty:
            print("🔄 正在地图上恢复旧节点...")
            for _, row in df_nodes.iterrows():
                color = 'ro' if row['Type'] == 'Building' else ('go' if row['Type'] == 'Door' else 'bo')
                p, = ax.plot(row['PX'], row['PY'], color, markersize=3)
                t = ax.text(row['PX'], row['PY'], f" {row['Name']}", fontsize=5)
                plot_elements[row['Node_ID']] = (p, t)

        # 绑定事件
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', undo_last)

        print("="*40)
        print("       BUPT 校园坐标采集工具 (Pandas版)")
        print("="*40)
        print("使用说明:")
        print("1. 若首次运行，请点击地图左下角设为【原点】。")
        print("2. 之后点击建筑或路口，在终端输入信息。")
        print("3. 按键盘 [Z] 键可撤销上一步操作。")
        print("4. 关闭地图窗口即自动保存数据。")
        print("-" * 40)

        plt.title("Campus Node Tool - Press 'Z' to Undo")
        plt.show()
        
        # 窗口关闭后执行保存
        save_to_csv()