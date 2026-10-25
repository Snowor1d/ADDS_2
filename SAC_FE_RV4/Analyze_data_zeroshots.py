import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# [USER CONFIG] 여기만 바꾸면 됨
# =========================================================
ROOT_DIR = os.path.expanduser("~/zero_shot_compare")
SAVE_DIR = ROOT_DIR

ORDER_LIST = ["3maps", "30maps", "100maps", "300maps"]
MODEL_CODE = "Q"

# --- Font ---
FONT_FAMILY = "DejaVu Serif"

# --- Figure ---
FIGSIZE = (18, 6)
SAVE_DPI = 300

# --- Bar style ---
BAR_COLOR = "#7389AF"
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 1.0

# --- Repeated-run statistics ---
# Bar: mean, error bar: sample standard deviation (ddof=1)
STEP_SECONDS = 0.5
MAX_STEP_NUM = 3000
SHOW_ERROR_BARS = True
ERROR_BAR_CAPSIZE = 10
ERROR_BAR_LINEWIDTH = 4.0    # 표준편차 세로선과 cap 굵기
SHOW_RAW_POINTS = True
RAW_POINT_COLOR = "black"
RAW_POINT_SIZE = 75          # 개별 실행점 크기
RAW_POINT_ALPHA = 0.55
RAW_POINT_JITTER = 0.10
SHOW_SUCCESS_RATE = False
SUCCESS_TEXT_FONTSIZE = 14

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
Y_LABEL_TEXT = "Time (s)"
X_LABEL_TEXT = ""  # 보통 비워둠

# --- Grid/Theme ---
SEABORN_THEME = "whitegrid"
# =========================================================

# --- Plot mode ---
# "bar"  → 막대그래프만
# "line" → 꺾은선그래프만
# "both" → 같은 figure에 막대 + 꺾은선 같이 표시
PLOT_MODE = "bar"

# --- Line style ---
LINE_MARKER = "o"
LINE_LINEWIDTH = 2.5
LINE_MARKERSIZE = 8
LINE_COLOR = "#d62728"   # 선은 다른 색으로 (논문용 대비 좋음)

FIX_Y_LIM = True
Y_LIM_MIN = 0
Y_LIM_MAX = 1400


def read_metric_value(filepath, key):
    """key=value 형식의 metric 값을 읽는다."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    return float(line.split("=", 1)[1].strip())
    except (OSError, ValueError) as e:
        print(f"Error reading {filepath}: {e}")
    return None


def read_episode_completed(filepath):
    """episode_log.txt의 마지막 생존 인원으로 100% 대피 여부를 판정한다."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            values = ast.literal_eval(f.read().strip())
        if isinstance(values, list) and values:
            return float(values[-1]) <= 0
    except (OSError, ValueError, SyntaxError):
        pass
    return None

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


def collect_data(root_path):
    """
    zero_shot_compare의 훈련 세트별 결과에서 episode 단위 값을 수집한다.

    예상 구조:
      zero_shot_compare/<training_set>/Result_<map>/Result_<map>_<model>/
        Result_<map>_<model>_<run>/metrics.txt

    개별 실행 파일이 없으면 avg_metrics.txt를 평균값 전용 행으로 읽는다.
    """
    data_list = []
    if not os.path.isdir(root_path):
        print(f"❌ 경로 없음: {root_path}")
        return pd.DataFrame()

    discovered_sets = [
        name
        for name in os.listdir(root_path)
        if re.fullmatch(r"\d+maps", name)
        and os.path.isdir(os.path.join(root_path, name))
    ]
    training_sets = [name for name in ORDER_LIST if name in discovered_sets]
    training_sets.extend(
        sorted(
            (name for name in discovered_sets if name not in ORDER_LIST),
            key=lambda name: int(name.removesuffix("maps")),
        )
    )

    for training_set in training_sets:
        training_path = os.path.join(root_path, training_set)
        map_folders = sorted(
            name
            for name in os.listdir(training_path)
            if re.fullmatch(r"Result_\d+", name)
            and os.path.isdir(os.path.join(training_path, name))
        )

        for map_f in map_folders:
            map_id = map_f.removeprefix("Result_")
            map_path = os.path.join(training_path, map_f)
            model_dir_name = f"Result_{map_id}_{MODEL_CODE}"
            model_path = os.path.join(map_path, model_dir_name)
            if not os.path.isdir(model_path):
                print(f"[WARN] 모델 결과 폴더 없음: {model_path}")
                continue

            run_pattern = re.compile(
                rf"^Result_{re.escape(map_id)}_{re.escape(MODEL_CODE)}_(\d+)$"
            )
            run_rows = []
            for run_name in sorted(os.listdir(model_path)):
                run_path = os.path.join(model_path, run_name)
                if not (run_pattern.fullmatch(run_name) and os.path.isdir(run_path)):
                    continue

                metric_path = os.path.join(run_path, "metrics.txt")
                evac_steps = read_metric_value(metric_path, "evacuation_100_time")
                if evac_steps is None:
                    continue

                completed = read_episode_completed(os.path.join(run_path, "episode_log.txt"))
                if completed is None:
                    # 로그가 없으면 기존 metrics 규약에 따라 상한 미만만 완료로 간주한다.
                    completed = evac_steps < MAX_STEP_NUM

                run_rows.append({
                    "Map": map_id,
                    "Model": training_set,
                    "Run": run_name,
                    "EvacSteps": evac_steps,
                    "TimeSec": evac_steps * STEP_SECONDS,
                    "Completed": completed,
                    "Source": "run",
                })

            if run_rows:
                data_list.extend(run_rows)
                continue

            # 개별 run 폴더 없이 평균 파일만 있는 경우의 fallback.
            avg_path = os.path.join(model_path, "avg_metrics.txt")
            avg_steps = read_avg_file(avg_path)
            if avg_steps is not None:
                print(
                    f"[WARN] {training_set}/map {map_id}: 개별 metrics.txt가 없어 "
                    "표준편차를 계산할 수 없습니다. 평균값만 표시합니다."
                )
                data_list.append({
                    "Map": map_id,
                    "Model": training_set,
                    "Run": "avg_only",
                    "EvacSteps": avg_steps,
                    "TimeSec": avg_steps * STEP_SECONDS,
                    "Completed": np.nan,
                    "Source": "average",
                })

    return pd.DataFrame(data_list)


def plot_bar_charts(df):
    if df.empty:
        print("❌ 데이터가 없습니다.")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
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

        summary = (
            plot_data.groupby("Model", observed=True)["TimeSec"]
            .agg(["mean", "std", "count"])
            .reindex(ORDER_LIST)
            .dropna(subset=["mean"])
        )
        models = summary.index.tolist()
        x_pos = np.arange(len(models), dtype=float)
        means = summary["mean"].to_numpy(dtype=float)
        # 표본이 하나뿐인 평균-only 데이터에는 error bar를 그리지 않는다.
        stds = summary["std"].fillna(0.0).to_numpy(dtype=float)

        # =====================================================
        # 1️⃣ BAR
        # =====================================================
        if PLOT_MODE in ["bar", "both"]:
            yerr = stds if SHOW_ERROR_BARS else None
            bars = ax.bar(
                x_pos,
                means,
                yerr=yerr,
                capsize=ERROR_BAR_CAPSIZE if SHOW_ERROR_BARS else 0,
                color=BAR_COLOR,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                alpha=0.8 if PLOT_MODE == "both" else 1.0,
                error_kw={
                    "ecolor": "black",
                    "elinewidth": ERROR_BAR_LINEWIDTH,
                    "capthick": ERROR_BAR_LINEWIDTH,
                },
            )

            if SHOW_BAR_VALUES:
                ax.bar_label(
                    bars,
                    fmt=BAR_VALUE_FMT,
                    padding=BAR_VALUE_PADDING,
                    fontsize=BAR_VALUE_FONTSIZE,
                    fontweight=BAR_VALUE_FONTWEIGHT,
                )

        # =====================================================
        # 2️⃣ LINE
        # =====================================================
        if PLOT_MODE in ["line", "both"]:
            # line-only에서는 SD도 함께 표시하고, both에서는 bar의 SD와 중복하지 않는다.
            line_yerr = stds if (SHOW_ERROR_BARS and PLOT_MODE == "line") else None
            ax.errorbar(
                x_pos,
                means,
                yerr=line_yerr,
                marker=LINE_MARKER,
                linewidth=LINE_LINEWIDTH,
                markersize=LINE_MARKERSIZE,
                color=LINE_COLOR,
                capsize=ERROR_BAR_CAPSIZE if line_yerr is not None else 0,
                elinewidth=ERROR_BAR_LINEWIDTH,
                capthick=ERROR_BAR_LINEWIDTH,
                zorder=10,   # 막대 위에 그리기
            )

            if SHOW_BAR_VALUES and PLOT_MODE == "line":
                for x, y in zip(x_pos, means):
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
        # 3️⃣ RAW EPISODE POINTS + SUCCESS RATE
        # =====================================================
        raw_df = plot_data[plot_data["Source"] == "run"]
        rng = np.random.default_rng(0)
        for i, model_name in enumerate(models):
            model_runs = raw_df[raw_df["Model"] == model_name]
            if model_runs.empty:
                continue

            if SHOW_RAW_POINTS:
                jitter = rng.uniform(-RAW_POINT_JITTER, RAW_POINT_JITTER, len(model_runs))
                ax.scatter(
                    i + jitter,
                    model_runs["TimeSec"].to_numpy(dtype=float),
                    s=RAW_POINT_SIZE,
                    color=RAW_POINT_COLOR,
                    alpha=RAW_POINT_ALPHA,
                    zorder=20,
                )

            if SHOW_SUCCESS_RATE:
                success = int(model_runs["Completed"].sum())
                total = len(model_runs)
                label_y = means[i] + (stds[i] if SHOW_ERROR_BARS else 0.0)
                ax.annotate(
                    f"Success {success}/{total}",
                    xy=(i, label_y),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=SUCCESS_TEXT_FONTSIZE,
                )

        # =====================================================

        ax.set_title(
            f"[Map {map_id}] Average Evacuation Time",
            fontsize=TITLE_FONTSIZE,
            fontweight=TITLE_FONTWEIGHT,
        )
        ax.set_ylabel(Y_LABEL_TEXT, fontsize=YLABEL_FONTSIZE)
        ax.set_xlabel(X_LABEL_TEXT, fontsize=XLABEL_FONTSIZE)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)
        ax.tick_params(axis="x", labelsize=XTICK_FONTSIZE, rotation=XTICK_ROTATION)
        ax.tick_params(axis="y", labelsize=YTICK_FONTSIZE)

        if FIX_Y_LIM:
            # SD, raw point, 성공률 annotation이 기존 상한에 잘리지 않도록 확장한다.
            raw_max = float(raw_df["TimeSec"].max()) if not raw_df.empty else 0.0
            error_max = float(np.max(means + stds)) if len(means) else 0.0
            required_max = max(raw_max, error_max)
            upper = max(Y_LIM_MAX, required_max * 1.12)
            ax.set_ylim(Y_LIM_MIN, upper)

        plt.tight_layout()

        save_name = f"Graph_Map_{map_id}_Avg_{PLOT_MODE}.png"
        full_save_path = os.path.join(SAVE_DIR, save_name)
        plt.savefig(full_save_path, dpi=SAVE_DPI)
        print(f"✅ Saved: {full_save_path}")
        plt.close()


if __name__ == "__main__":
    print(f"Target Root: {ROOT_DIR}")
    df = collect_data(ROOT_DIR)

    if not df.empty:
        print(f"Loaded {len(df)} data points.")
        plot_bar_charts(df)
    else:
        print("No data loaded.")
