import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# [설정] 데이터 루트 경로
# ==========================================
ROOT_DIR = os.path.expanduser("~/Result_0109") 
SAVE_DIR = ROOT_DIR 

# 2개 vs 2개 비교를 위한 리스트
ORDER_LIST = [
    "Global_only_3maps",
    # "Global_only_5maps",   <-- 제외됨
    "Global_only_10maps",
    "Global+Ego_3maps",
    # "Global+Ego_5maps",    <-- 제외됨
    "Global+Ego_10maps"
]

def read_avg_file(filepath):
    """avg_metrics.txt에서 evacuation_100_time 값만 추출"""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if "evacuation_100_time" in line:
                    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)
                    if nums:
                        return float(nums[-1])
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None

def collect_avg_data(root_path):
    data_list = []
    if not os.path.exists(root_path):
        print(f"❌ 경로 없음: {root_path}")
        return pd.DataFrame()

    map_folders = sorted([f for f in os.listdir(root_path) if f.startswith("Result_")])
    
    for map_f in map_folders:
        map_full_path = os.path.join(root_path, map_f)
        if not os.path.isdir(map_full_path): continue

        m = re.match(r"Result_(\d+)$", map_f)
        if not m: continue
        map_id = m.group(1)

        cfgs = sorted(os.listdir(map_full_path))
        for cfg_f in cfgs:
            prefix = f"Result_{map_id}_"
            if not cfg_f.startswith(prefix):
                continue
            
            model_name = cfg_f[len(prefix):]
            avg_path = os.path.join(map_full_path, cfg_f, "avg_metrics.txt")
            
            val = read_avg_file(avg_path)
            if val is not None:
                data_list.append({
                    "Map": map_id,
                    "Model": model_name,
                    "AvgTime": val
                })
                
    return pd.DataFrame(data_list)

def plot_bar_charts(df):
    if df.empty:
        print("❌ 데이터가 없습니다.")
        return

    sns.set_theme(style="whitegrid")
    unique_maps = sorted(df['Map'].unique())

    print(f"📂 저장 위치: {SAVE_DIR}")

    for map_id in unique_maps:
        sub_df = df[df['Map'] == map_id]
        
        # 필터링 및 정렬
        plot_data = sub_df[sub_df['Model'].isin(ORDER_LIST)].copy()
        plot_data['Model'] = pd.Categorical(plot_data['Model'], categories=ORDER_LIST, ordered=True)
        plot_data = plot_data.sort_values('Model')

        if plot_data.empty:
            continue

        plt.figure(figsize=(8, 6)) # 그래프 폭을 조금 줄임 (막대가 줄어서)
        
        # [수정됨] 색상: 앞 2개 파랑, 뒤 2개 주황
        colors = ['#4c72b0'] * 2 + ['#dd8452'] * 2
        
        ax = sns.barplot(
            data=plot_data,
            x="Model",
            y="AvgTime",
            palette=colors[:len(plot_data)],
            edgecolor="black"
        )

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=10, fontweight='bold')

        # [수정됨] 구분선 위치: 2번째와 3번째 사이 (인덱스 1.5)
        # 인덱스: 0(3maps), 1(10maps) | 2(3maps), 3(10maps)
        plt.axvline(x=1.5, color='red', linestyle='--', linewidth=2)
        plt.text(1.5, ax.get_ylim()[1]*1.02, ' Separation ', color='red', ha='center', fontweight='bold')

        plt.title(f"[Map {map_id}] Average Evacuation Time", fontsize=15, fontweight='bold')
        plt.ylabel("Time (sec)")
        plt.xlabel("")
        plt.xticks(rotation=20)
        plt.tight_layout()

        save_name = f"Graph_Map_{map_id}_Avg.png"
        full_save_path = os.path.join(SAVE_DIR, save_name)
        
        plt.savefig(full_save_path, dpi=150)
        print(f"✅ Saved: {full_save_path}")
        plt.close()

if __name__ == "__main__":
    print(f"Target Root: {ROOT_DIR}")
    df = collect_avg_data(ROOT_DIR)
    
    if not df.empty:
        print(f"Loaded {len(df)} data points.")
        plot_bar_charts(df)
    else:
        print("No data loaded.")