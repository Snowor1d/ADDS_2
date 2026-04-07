import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# =========================================================
# [USER CONFIG] 여기만 바꾸면 됨
# =========================================================
ROOT_DIR = os.path.expanduser("~/Result_zero")
SAVE_DIR = ROOT_DIR

ORDER_LIST = ["3maps", "10maps", "30maps", "100maps", "300maps"]

# --- Font ---
FONT_FAMILY = "DejaVu Serif"

# --- Figure ---
FIGSIZE = (18, 7)
SAVE_DPI = 300

# --- Bar style ---
BAR_COLOR = "#4c72b0"
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 1.0

# --- Value labels on bars ---
SHOW_BAR_VALUES = False          # ✅ True/False로 숫자 표시 켜기/끄기
BAR_VALUE_FMT = "%.1f"
BAR_VALUE_PADDING = 3

# --- Text sizes (커스텀 가능) ---
TITLE_FONTSIZE = 20
TITLE_FONTWEIGHT = "bold"

YLABEL_FONTSIZE = 50
XLABEL_FONTSIZE = 50
XTICK_FONTSIZE = 40
YTICK_FONTSIZE = 40

BAR_VALUE_FONTSIZE = 11
BAR_VALUE_FONTWEIGHT = "bold"

# --- X tick rotation (0이면 안 눕힘) ---
XTICK_ROTATION = 0

# --- Axis labels text ---
Y_LABEL_TEXT = "Time (sec)"
X_LABEL_TEXT = ""  # 보통 비워둠

# --- Grid/Theme ---
SEABORN_THEME = "whitegrid"
# =========================================================

# --- Plot mode ---
# "bar"  → 막대그래프만
# "line" → 꺾은선그래프만
# "both" → 같은 figure에 막대 + 꺾은선 같이 표시
PLOT_MODE = "line"

# --- Line style ---
LINE_MARKER = "o"
LINE_LINEWIDTH = 7
LINE_MARKERSIZE = 12
LINE_COLOR = "#d62728"   # 선은 다른 색으로 (논문용 대비 좋음)

FIX_Y_LIM = True
Y_LIM_MIN = 0
Y_LIM_MAX = 1400

def read_avg_file(filepath):
    """avg_metrics.txt에서 evacuation_100_time 값만 추출"""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
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
        if not os.path.isdir(map_full_path):
            continue

        m = re.match(r"Result_(\d+)$", map_f)
        if not m:
            continue
        map_id = m.group(1)

        cfgs = sorted(os.listdir(map_full_path))
        for cfg_f in cfgs:
            prefix = f"Result_{map_id}_"
            if not cfg_f.startswith(prefix):
                continue

            model_name = cfg_f[len(prefix):]
            avg_path = os.path.join(map_full_path, cfg_f, "avg_metrics.txt")

            val = read_avg_file(avg_path)/2
            if val is not None:
                data_list.append({"Map": map_id, "Model": model_name, "AvgTime": val})

    return pd.DataFrame(data_list)


def plot_bar_charts(df):
    if df.empty:
        print("❌ 데이터가 없습니다.")
        return

    sns.set_theme(style=SEABORN_THEME, font=FONT_FAMILY)
    
    # 2. 강제로 모든 요소를 Serif 계열로 맞추고 우선순위 부여
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [FONT_FAMILY, "Times New Roman", "Times"]
    plt.rcParams["axes.unicode_minus"] = False
    unique_maps = sorted(df["Map"].unique())
    print(f"📂 저장 위치: {SAVE_DIR}")

    for map_id in unique_maps:
        sub_df = df[df["Map"] == map_id]

        plot_data = sub_df[sub_df["Model"].isin(ORDER_LIST)].copy()
        plot_data["Model"] = pd.Categorical(
            plot_data["Model"], categories=ORDER_LIST, ordered=True
        )
        plot_data = plot_data.sort_values("Model")

        if plot_data.empty:
            continue

        plt.figure(figsize=FIGSIZE)
        ax = plt.gca()

        x_vals = plot_data["Model"]
        y_vals = plot_data["AvgTime"]

        # =====================================================
        # 1️⃣ BAR
        # =====================================================
        if PLOT_MODE in ["bar", "both"]:
            bar = sns.barplot(
                data=plot_data,
                x="Model",
                y="AvgTime",
                color=BAR_COLOR,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                alpha=0.8 if PLOT_MODE == "both" else 1.0,
                ax=ax,
            )

            if SHOW_BAR_VALUES:
                for container in bar.containers:
                    bar.bar_label(
                        container,
                        fmt=BAR_VALUE_FMT,
                        padding=BAR_VALUE_PADDING,
                        fontsize=BAR_VALUE_FONTSIZE,
                        fontweight=BAR_VALUE_FONTWEIGHT,
                    )

        # =====================================================
        # 2️⃣ LINE
        # =====================================================
        if PLOT_MODE in ["line", "both"]:
            ax.plot(
                x_vals,
                y_vals,
                marker=LINE_MARKER,
                linewidth=LINE_LINEWIDTH,
                markersize=LINE_MARKERSIZE,
                color=LINE_COLOR,
                zorder=10,   # 막대 위에 그리기
            )

            if SHOW_BAR_VALUES and PLOT_MODE == "line":
                for x, y in zip(x_vals, y_vals):
                    ax.text(
                        x,
                        y,
                        BAR_VALUE_FMT % y,
                        ha="center",
                        va="bottom",
                        fontsize=BAR_VALUE_FONTSIZE,
                        fontweight=BAR_VALUE_FONTWEIGHT,
                    )

        # =====================================================

        ax.set_title(
            f"[Map {map_id}] Average Evacuation Time",
            fontsize=TITLE_FONTSIZE,
            fontweight=TITLE_FONTWEIGHT,
        )
        ax.set_ylabel(Y_LABEL_TEXT, fontsize=YLABEL_FONTSIZE)
        ax.set_xlabel(X_LABEL_TEXT, fontsize=XLABEL_FONTSIZE)

        ax.tick_params(axis="x", labelsize=XTICK_FONTSIZE, rotation=XTICK_ROTATION)
        ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)

        if FIX_Y_LIM:
            ax.set_ylim(Y_LIM_MIN, Y_LIM_MAX)

        def hide_zero_formatter(x, pose):
            if x==0:
                return ""
            return f"{int(x)}"

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(hide_zero_formatter))
        plt.tight_layout()

        save_name = f"Graph_Map_{map_id}_Avg_{PLOT_MODE}.png"
        full_save_path = os.path.join(SAVE_DIR, save_name)
        plt.savefig(full_save_path, dpi=SAVE_DPI)
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