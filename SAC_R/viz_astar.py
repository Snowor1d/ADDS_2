# viz_astar.py ───────────────────────────────────────────
from functools import partial
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.UserParam import NumberInput
from model import FightingModel

# ───── 기본 파라미터 ─────────────────────────────────────
WIDTH, HEIGHT   = 50, 50           # 격자 크기
N_AGENTS        = 20               # Crowd 수(기본값)
MAP_CANDIDATES  = [6, 7, 8]        # <- start_training.py 의 MAP_NUM_RANDOM
PORT            = 8522             # Web GUI 포트
CELL_PX         = 20               # 셀 하나의 픽셀수

# 0) 서버가 ‘미리보기용’ 모델을 하나 요구하므로 여기서 한 번 만들어 둡니다.
preview_model = FightingModel(N_AGENTS, WIDTH, HEIGHT,
                              model_num=MAP_CANDIDATES[0],
                              robot='A')

# 1) ‘로봇=A*’ 인스턴스를 매번 새로 생성해 주는 팩터리
def factory(number_agents=N_AGENTS,
            width=WIDTH,
            height=HEIGHT,
            model_num=MAP_CANDIDATES[0]):
    """
    Tornado-Server가 호출할 때마다 FightingModel(robot='A') 를 반환.
    model_num 은 UI에서 바꿀 수 있게 매개변수로 둡니다.
    """
    # 무작위 맵 선택 (6,7,8번 중)
    if model_num == -1:
        import random
        model_num = random.choice(MAP_CANDIDATES)

    return FightingModel(number_agents, width, height,
                         model_num=model_num, robot='A')

# 2) agent → 화면 표시 변환
def portray(agent):
    if getattr(agent, "buried", False):
        return {"Shape": "circle", "Color": "white", "Filled": True,
                "r": 0.1, "Layer": 0}

    # 벽 / 출구
    if agent.type == 11:
        return {"Shape": "rect", "w": 1, "h": 1,
                "Color": "black", "Filled": True, "Layer": 3}
    if agent.type == 10:
        return {"Shape": "rect", "w": 1, "h": 1,
                "Color": "green", "Filled": True, "Layer": 2}

    # Crowd
    if agent.type in (0, 1, 2):
        colour = "magenta" if agent.type == 0 else "royalblue"
        return {"Shape": "circle", "Color": colour, "Filled": True,
                "r": 0.5, "Layer": 1}

    # 로봇
    if agent.type == 3:
        col = "red" if agent.model.robot_mode == "GUIDE" else "purple"
        return {"Shape": "circle", "Color": col, "Filled": True,
                "r": 0.6, "Layer": 4}

    # 기타
    return {"Shape": "circle", "Color": "grey", "Filled": True,
            "r": 0.4, "Layer": 0}

# 3) 시각화 위젯들
grid  = CanvasGrid(
            portray,
            WIDTH, HEIGHT,
            WIDTH * CELL_PX, HEIGHT * CELL_PX
        )

chart = ChartModule(
            [{"Label": "Remained Agents", "Color": "blue"}],
            data_collector_name="datacollector_currents",
            canvas_height=250
        )

# 4) 사용자 조정 파라미터 (왼쪽 패널)
params = {
    "number_agents": NumberInput("Crowd size", value=N_AGENTS),
    "width"        : WIDTH,
    "height"       : HEIGHT,
    "model_num"    : -1,              # -1 이면 자동으로 6/7/8 중 랜덤
}

# 5) Tornado 서버 생성
server = ModularServer(
            factory,                 # 모델 팩터리 (callable)
            [grid, chart],           # 시각화 요소
            "ADDS – A* Robot Demo",  # 타이틀
            params,                  # GUI 에 노출할 파라미터
            PORT,
            preview_model            # ★ 초기 미리보기 모델 (필수!)
        )

server.port = PORT

if __name__ == "__main__":
    server.launch()
# ─────────────────────────────────────────────────────────
