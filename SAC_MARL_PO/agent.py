#this source code requires Mesa==2.2.1 
#^__^
from core import Agent
import socket
import time 
import math
import numpy as np
import random
import copy
import sys 
from collections import deque
from heapq import heappush, heappop
from shapely.geometry import Point
from shapely.geometry import Polygon, MultiPolygon
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import *
from ADDS_AS_reinforcement import FrameStackWithStep


 # goals의 가운데를 가져오는 함수
 # 어디로 향하게 할 것인가? -> goals의 가운데 

class WallAgent(Agent): ## wall .. 탈출구 범위 내에 agents를 채워넣어서 탈출구라는 것을 보여주고 싶었음.. 
    def __init__(self, unique_id, model, pos, agent_type):
        super().__init__(unique_id, model)
        self.pos = pos
        self.type = agent_type
        self.buried = 0
        self.dead = 0
        self.xy =pos

    
    
class CrowdAgent(Agent):
    """An agent that fights."""

    def __init__(self, unique_id, model, pos, type_agent): 
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.next_mesh = None
        self.past_mesh = None
        self.previous_mesh = None
        self.pos = pos
        self.behavior_probability = [random.gauss(0.9, 0.1), random.gauss(0.2, 0.1), random.gauss(0.1, 0.1)] #robot #동조 #myway
        self.robot_step = 0
        self.type = type_agent

        self.dead = False

        self.danger = 0
        self.previous_danger = 0

        self.drag = 0
        self.dead_count = 0
        self.buried = False
        self.previous_stage = []
        self.now_goal = [0,0]
        self.now_pointing_mesh = None
        self.robot_previous_goal = [0, 0]
        self.robot_initialized = 0
        self.direction = [0, 0]

        # print(isinstance(pos, tuple))
        self.xy = pos
        self.vel = [0, 0]
        self.acc = [0, 0]
        # self.mass = 3
        self.mass = np.random.normal(66, 4.16) # agent의 mass, 평균 66kg, 표준 편차 4.16kg
        if self.type == 3: # robot mass는 3으로 고정
            self.mass = 30

        self.desired_speed_a = np.random.normal(AGENT_SPEED_MEAN, 0.2) # agent의 desired_speed, 평균 1.5m/s, 표준 편차 0.2m/s

        self.is_effected_by_robot = 0
        self.blocked = False

        self.decision_flag = random.randint(1,5) # self.decision_flag == 0 -> 결정 다시 내림
        self.decision_period = random.randint(15,35) #self.decision_period == 0 -> 결정 다시 내림, 군중 마다 얼마만큼의 시간동안 자신의 결정을 번복하지 않는가 모델링


        self.model.robot_mode = "GUIDE"
 
        self.exit_belief = None       # {"idx": int, "score": float, "alpha": int}
        self.life_time = 0
        self.body_radius = AGENT_BODY_RADIUS
        self.vision_radius = AGENT_VISION
        self.meeting_robot = 0

        self.meet_index = 5
        self.following_robot_index = 5


    def step(self) -> None:

        """Handles the step of the model dor each agent.
        Sets the flags of each agent during the simulation.
        """
        if not self.dead:
            self.life_time += 1
            
        # buried agents do not move (Do they???? :))
        if self.buried:
            return

        # dead for too long it is buried not being displayed 
        if self.dead_count > 4:
            self.buried = True
            return

        # no health and not buried increment the count
        if self.dead and not self.buried:
            self.dead_count += 1
            return

        if(self.type != 3): #robot은 죽지 않는다
            if self.model.in_exit(self.xy, tol=0.3):
                self.dead = True
                self.model.space.remove(self.unique_id)
                return
        


        self.move()


    def choice_safe_mesh(self, point):
        point_grid = (int(round(point[0])), int(round(point[1])))
        x = point_grid[0]
        y = point_grid[1]
        while_checking = 0

        candidates = [(x+1,y+1), (x+1, y), (x, y+1), (x-1, y-1), (x-1, y), (x, y-1), (x+1, y-1), (x-1, y+1), (x-2, y), (x+2, y), (x, y-2), (x, y+2)]

        if x==self.model.width and y==self.model.height: 
            point_grid = (self.model.width -1, self.model.height -1) 
            return self.model.match_grid_to_mesh[point_grid]
        
        if (point_grid not in self.model.match_grid_to_mesh.keys()) or (self.model.match_grid_to_mesh[point_grid] not in self.model.pure_mesh):
            #print("다른 후보 찾기")
            #print("-")
            for c in candidates:
                if (c in self.model.match_grid_to_mesh.keys()) and (self.model.match_grid_to_mesh[c] in self.model.pure_mesh):
                    point_grid = c
                    return self.model.match_grid_to_mesh[point_grid]
                
            
            raise Exception(f"{x}, {y} 지점에서 오류 발생, safe mesh를 찾지 못했습니다")
        return self.model.match_grid_to_mesh[point_grid]
        



    def mesh_to_mesh_distance(self, point1, point2):
        point1_mesh = self.choice_safe_mesh(point1)
        point2_mesh = self.choice_safe_mesh(point2)

        return self.model.distance[point1_mesh][point2_mesh]

    def point_to_point_distance(self, point1, point2):

        point1_mesh = self.choice_safe_mesh(point1)
        point2_mesh = self.choice_safe_mesh(point2)
        if self.model.next_vertex_matrix[point1_mesh][point2_mesh] == None:
            return 99999999999
        
        distance = 0
        now_mesh = point1_mesh

        if (self.model.next_vertex_matrix[now_mesh][point2_mesh] == point2_mesh):
            return math.sqrt(pow(point1[0]-point2[0],2)+pow(point1[1]-point2[1],2))

        now_mesh = self.model.next_vertex_matrix[now_mesh][point2_mesh]
        now_mesh_middle = ((now_mesh[0][0]+now_mesh[1][0]+now_mesh[2][0])/3, (now_mesh[0][1]+now_mesh[1][1]+now_mesh[2][1])/3)
        distance += math.sqrt(pow(now_mesh_middle[0]-point1[0],2)+pow(point1[1]-now_mesh_middle[1],2))

        while(self.model.next_vertex_matrix[now_mesh][point2_mesh] != point2_mesh):
            distance += self.model.distance[now_mesh][self.model.next_vertex_matrix[now_mesh][point2_mesh]]
            now_mesh = self.model.next_vertex_matrix[now_mesh][point2_mesh]
        
        now_mesh_middle = ((now_mesh[0][0]+now_mesh[1][0]+now_mesh[2][0])/3, (now_mesh[0][1]+now_mesh[1][1]+now_mesh[2][1])/3)    

        distance += math.sqrt(pow(now_mesh_middle[0]-point2[0],2)+pow(now_mesh_middle[1]-point2[1],2))
        
        return distance


    def _neighbors(self, radius):
        # 1) 기존처럼 반경 후보
        candidates = self.model.space.query_radius(self.xy, radius, predicate=None)
        if not candidates:
            return []

        # 2) 시야 폴리곤은 '사전계산된 것'을 조회만
        poly = self.model.vision_atlas.polygon_at(
            self.xy[0], self.xy[1], radius, self.model.obstacles_version
        )

        minx, miny, maxx, maxy = poly.bounds
        

        out = []
        if not poly.is_empty:
            for b in candidates:
                ref = b.ref
                if (ref is None) or (ref is self) or getattr(ref, "dead", False):
                    continue

                x, y = b.pos[0], b.pos[1]
                if poly.covers(Point(b.pos[0], b.pos[1])):
                    out.append(ref)

        return out
    
    def move(self) -> None:
        """Handles the movement behavior.
        Here the agent decides   if it moves,
        drinks the heal potion,
        or attacks other agent."""

        if (self.type == 3):
            self.robot_step += 1

                   
            if self.model.robot_type == "Q":
                new_position_robot = self.robot_policy_Q()
            
            elif self.model.robot_type == "T":
                new_position_robot = self.robot_policy_Q()
            elif self.model.robot_type == "R":
                new_position_robot = self.robot_policy_Q()
            else:
                raise ValueError(f"Unknown robot_type {self.model.robot_type}")
            

            self.model.space.move(self.unique_id, self.xy)
            self.pos = new_position_robot
            return
        
        if self.type in (0, 1, 2):               # (로봇이 아니면)
            # (2) 힘 계산·충돌 예측·이동 ----------
            self.pos = (round(self.xy[0]), round(self.xy[1]))
            new_pos  = self.agent_modeling()      # ← 내부에서 predict_collision() 포함
            self.pos = (self.xy[0], self.xy[1])
    
    def _wall_repulsion(self):
        from shapely.geometry import Point
        Fwx = Fwy = 0.0
        KN = 1.8e5
        CN = 1000
        MU_T = 2.5e5
        p = Point(self.xy[0], self.xy[1])
        F_contact_x = 0.0
        F_contact_y = 0.0
        F_fric_x = 0.0
        F_fric_y = 0.0
        obstacle_polys = self.model._obstacle_polys.copy()
        obstacle_polys.append(Polygon([(0,0), (self.model.width-1,0), (self.model.width-1,self.model.height-1), (0,self.model.height-1)]))  # 맵 외곽 벽 추가

        for poly in obstacle_polys:
            d = poly.exterior.distance(p)
            r_sum = self.body_radius + 1  # wall radius = 1
            if d < r_sum :
                
                q = poly.exterior.interpolate(poly.exterior.project(p))
                dx = self.xy[0]-q.x
                dy = self.xy[1]-q.y
                dist = math.hypot(dx, dy) or 1e-9
                ux, uy = dx/dist, dy/dist
                

                penetration = r_sum - d
                Fn_k = KN * penetration
                # (b) 상대속도에 대한 법선 감쇠
                v_jx = 0
                v_jy = 0
                rel_vx, rel_vy = (self.vel[0] - v_jx), (self.vel[1] - v_jy)
                v_n = rel_vx*ux + rel_vy*uy        # 법선 성분
                
                if v_n < 0:
                    restitution = 0.2
                    drop = (1.0-restitution)*v_n
                    self.vel[0] -= drop*ux
                    self.vel[1] -= drop*uy

                Fn_c = -CN * v_n                   # 접근할수록(음수) + 방향은 법선
                Fn = max(Fn_k + Fn_c, 0.0)         # 법선 힘은 음수가 되지 않게

                F_contact_x += Fn * ux
                F_contact_y += Fn * uy

                # (c) 접선 방향(미끄럼) 마찰: v_t = rel_v - v_n n
                vt_x = rel_vx - v_n*ux
                vt_y = rel_vy - v_n*uy
                F_fric_x += -MU_T * vt_x
                F_fric_y += -MU_T * vt_y

            if d < r_sum * 2:
                q = poly.exterior.interpolate(poly.exterior.project(p))
                dx = self.xy[0]-q.x
                dy = self.xy[1]-q.y
                dist = math.hypot(dx, dy) or 1e-9
                nx, ny = dx/dist, dy/dist
                mag = 200 * math.exp(-(d/0.2))
                Fwx += mag * nx
                Fwy += mag * ny
        return Fwx, Fwy

    # ---- 스윕 이동(터널링 방지) ----
    def swept_move(self, xy, vel, dt):
        nx, ny = xy[0], xy[1]
        max_disp = max(abs(vel[0]*dt), abs(vel[1]*dt))
        steps = max(1, int(math.ceil(max_disp / 0.5)))
        sdt = dt / steps
        for _ in range(steps):
            tx = nx + vel[0]*sdt
            ty = ny + vel[1]*sdt
            if self.model.is_free((tx, ty)):
                nx, ny = tx, ty
            else:
                # 축 분리
                if self.model.is_free((nx + vel[0]*sdt, ny)):
                    nx += vel[0]*sdt
                if self.model.is_free((nx, ny + vel[1]*sdt)):
                    ny += vel[1]*sdt
        return [nx, ny]

        

    def agent_modeling(self):
        """
        Helbing + Contact (penalty) model
        - 비관통(원-원) 접촉: 탄성(스프링) + 점성 감쇠 + 접선 마찰
        - 공기저항 형태 속도 감쇠로 관성 억제
        """
        import math



        # ====== 기본 파라미터 (필요하면 수치만 조정) ======
        dt   = AGENT_TIME_STEP
        tau  = 1                  
        A_MAX = 1.5                    # 가속 클립 ↑ 약간 강화
        V_MAX_MULT = 1.00              # 목표속도보다 과속 안하게
        BODY_RADIUS = 0.5             # 군중 몸 반지름 [cell] 0.5m
        WALL_RADIUS = 1.5             # 격자벽을 둥근 장애물로 근사
        # 접촉(법선) 스프링/감쇠, 접선 마찰
        KN = 0.8e5                   # 법선 스프링 상수, modified crowd simulation 논문에선 1.2*10^5
        CN =  1000                     # 법선 점성(접근속도 감쇠)
        MU_T = 2.5e5                    # 접선 마찰(미끄럼 속도 감쇠) modifided crowd simulation 논문에선 2.4*10^5
        # 지수형 반발은 약화(근거리에서만 의미)
        K_AGENT = 200 # modified 참고
        K_WALL  = 500 # modified 참고
        LAMBDA_A = 0.3 # modifided 참고
        # 공기저항(속도 감쇠) → 둥둥 뜨는 느낌 제거
        BETA = 0                     # F_drag = -BETA * v

        # ---- 유틸 ----
        def get_radius(agent):
            if getattr(agent, "type", None) == 3:
                return ROBOT_BODY_RADIUS
            elif getattr(agent, "type", None) in (9, 11):  # 벽/장애물
                return WALL_RADIUS
            else:
                return BODY_RADIUS

        def soft_clip_vec(x, y, lim):
            n = math.hypot(x, y)
            if n <= lim: return x, y
            s = math.tanh(n/lim) / (n/lim)
            return x*s, y*s
        
        self.danger = 1e9
        self.danger = self.model.distance_to_exit(self.xy)
        if (self.danger > 1e5):
            self.dead = True

        # 이웃 상호작용 (사람/로봇)
        sensor_R = self.vision_radius
        near_agents = self._neighbors(sensor_R)

        self.which_goal_agent_want(near_agents)

        # ---- 목표 방향 ----
        gx = self.now_goal[0] - self.xy[0]
        gy = self.now_goal[1] - self.xy[1]
        gd = math.hypot(gx, gy)
        if gd > 0:
            dir_x, dir_y = gx/gd, gy/gd
        else:
            dir_x, dir_y = 0.0, 0.0

        # ---- 원하는 속도 → Helbing desired force ----
        v_des_x = self.desired_speed_a * dir_x
        v_des_y = self.desired_speed_a * dir_y
        F_des_x = self.mass * (v_des_x - self.vel[0]) / tau
        F_des_y = self.mass * (v_des_y - self.vel[1]) / tau

        # ---- 기존의 약한(원거리) 반발력 (지수) ----
        F_rep_x = 0.0
        F_rep_y = 0.0

        # ---- 접촉(비관통) + 마찰 모델(핵심 추가) ----
        F_contact_x = 0.0
        F_contact_y = 0.0
        F_fric_x    = 0.0
        F_fric_y    = 0.0

        r_i = BODY_RADIUS
        # 자기 상태 (속도)
        v_ix, v_iy = self.vel[0], self.vel[1]
        self.meeting_robot = 0
        for nb in near_agents:
            if nb is self or getattr(nb, "dead", False):
                continue
            if nb.type == 3:
                self.meeting_robot = 1
                self.meet_index = nb.robot_index

            dx = self.xy[0] - nb.xy[0]
            dy = self.xy[1] - nb.xy[1]
            d  = math.hypot(dx, dy)
            if d < 1e-9:
                # 완전 겹침 초기 해소(랜덤 툭 치기)
                jx, jy = (1.0, -1.0) if random.random() < 0.5 else (-1.0, 1.0)
                F_contact_x += jx * KN * 0.01
                F_contact_y += jy * KN * 0.01
                continue

            ux, uy = dx/d, dy/d  # (nb -> self) 법선 방향
            nb_R = getattr(nb, "radius", BODY_RADIUS)
            r_sum = self.body_radius + nb_R

            # 원거리 지수 반발
            mag = K_AGENT * math.exp((r_sum-d) / max(LAMBDA_A, 1e-6))
            F_rep_x += mag * ux
            F_rep_y += mag * uy

            # # # 1) 원거리 지수 반발(부드러운 회피)
            # if getattr(nb, "type", None) in (11, 9):
            #     mag = K_WALL * math.exp((r_sum - d) / LAMBDA_A)
            #     F_rep_x += mag * ux
            #     F_rep_y += mag * uy
            # else:
            #     mag = K_AGENT * math.exp((r_sum - d) / LAMBDA_A)
            #     F_rep_x += mag * ux
            #     F_rep_y += mag * uy

            # 2) 근거리 접촉(비관통) + 점성 감쇠 + 접선 마찰
            if d < r_sum:
                # 침투량(양수면 겹침)
                penetration = (r_sum - d)
                # (a) 법선 스프링
                Fn_k = KN * penetration
                # (b) 상대속도에 대한 법선 감쇠
                v_jx = getattr(nb, "vel", [0,0])[0] if hasattr(nb, "vel") else 0.0
                v_jy = getattr(nb, "vel", [0,0])[1] if hasattr(nb, "vel") else 0.0
                rel_vx, rel_vy = (v_ix - v_jx), (v_iy - v_jy)
                v_n = rel_vx*ux + rel_vy*uy        # 법선 성분
                
                if v_n < 0:
                    restitution = 0
                    drop = (1.0-restitution)*v_n
                    self.vel[0] -= drop*ux
                    self.vel[1] -= drop*uy

                Fn_c = -CN * v_n                   # 접근할수록(음수) + 방향은 법선
                Fn = max(Fn_k + Fn_c, 0.0)         # 법선 힘은 음수가 되지 않게

                F_contact_x += Fn * ux
                F_contact_y += Fn * uy

                # (c) 접선 방향(미끄럼) 마찰: v_t = rel_v - v_n n
                vt_x = rel_vx - v_n*ux
                vt_y = rel_vy - v_n*uy
                F_fric_x += -MU_T * vt_x
                F_fric_y += -MU_T * vt_y
        
        BETA=0
        decay = 1
        self.vel[0] *= decay
        self.vel[1] *= decay
        # ---- 공기저항(속도 감쇠) ----
        F_drag_x = -BETA * self.vel[0]
        F_drag_y = -BETA * self.vel[1]


        # ---- 외곽 지대 나가지 않게 ----

        # 🔹 (추가) 맵 outer wall 반발력
        W = self.model.width
        H = self.model.height
        MARGIN = 2.0         # 이 거리 안으로 들어오면 힘 발생
        K_BORDER = 200.0     # 경계 힘 세기 (필요하면 조절)
        F_wx = 0
        F_wy = 0

        W_x, W_y = self._wall_repulsion()
        F_wx += W_x
        F_wy += W_y

        # left 벽 (x = 0 부근)
        dx = max(0.0, MARGIN - self.xy[0])
        if dx > 0.0:
            # 왼쪽 벽에 가까우면 +x 방향으로 민다
            F_wx += K_BORDER * dx

        # right 벽 (x = W 부근)
        dx = max(0.0, self.xy[0] - (W - MARGIN))
        if dx > 0.0:
            # 오른쪽 벽에 가까우면 -x 방향으로 민다
            F_wx -= K_BORDER * dx

        # bottom 벽 (y = 0 부근)
        dy = max(0.0, MARGIN - self.xy[1])
        if dy > 0.0:
            # 아래쪽 벽에 가까우면 +y 방향
            F_wy += K_BORDER * dy

        # top 벽 (y = H 부근)
        dy = max(0.0, self.xy[1] - (H - MARGIN))
        if dy > 0.0:
            # 위쪽 벽에 가까우면 -y 방향
            F_wy -= K_BORDER * dy


        # ---- 스윕 이동(터널링 방지) ----
        def swept_move(xy, vel, dt):
            nx, ny = xy[0], xy[1]
            max_disp = max(abs(vel[0]*dt), abs(vel[1]*dt))
            steps = max(1, int(math.ceil(max_disp / 0.5)))
            sdt = dt / steps
            for _ in range(steps):
                tx = nx + vel[0]*sdt
                ty = ny + vel[1]*sdt
                ix, iy = int(round(tx)), int(round(ty))
                if self.model.valid_space.get((ix, iy), False):
                    nx, ny = tx, ty
                    continue
                # 축 분리 시도
                ix_only = int(round(nx + vel[0]*sdt))
                if self.model.valid_space.get((ix_only, int(round(ny))), False):
                    nx = nx + vel[0]*sdt
                iy_only = int(round(ny + vel[1]*sdt))
                if self.model.valid_space.get((int(round(nx)), iy_only), False):
                    ny = ny + vel[1]*sdt
            return [nx, ny]

        #self.xy = swept_move(self.xy, self.vel, dt)
        # self.xy[0] = self.xy[0] + self.vel[0] * dt
        # self.xy[1] = self.xy[1] + self.vel[1] * dt



        # ---- 총합 힘 ----
        F_x = F_des_x + F_rep_x + F_contact_x + F_fric_x + F_drag_x + F_wx
        F_y = F_des_y + F_rep_y + F_contact_y + F_fric_y + F_drag_y + F_wy

        # ---- 가속도 계산 + 클립 ----
        a_x = F_x / self.mass
        a_y = F_y / self.mass
        a_x, a_y = soft_clip_vec(a_x, a_y, A_MAX)

        self.acc[0], self.acc[1] = a_x, a_y

        # ---- 속도 업데이트 ----
        self.vel[0] += a_x * dt
        self.vel[1] += a_y * dt

        # 속도 클립 (벡터 노름)
        v_des_scalar = max(self.desired_speed_a, 1e-6)
        V_MAX = V_MAX_MULT * v_des_scalar
        spd = math.hypot(self.vel[0], self.vel[1])
        if spd > V_MAX:
            s = V_MAX / spd
            self.vel[0] *= s; self.vel[1] *= s
        #print("agent desired speed : ", v_des_scalar)
        #print("agent speed : ", self.vel[0], self.vel[1])
        self.xy = self.swept_move(self.xy, self.vel, dt)
        #self.model.space.clamp(self.xy)
        self.model.space.move(self.unique_id, self.xy)

        self.direction = [self.vel[0], self.vel[1]]

        return tuple(self.xy)

    
    def which_goal_agent_want(self, neighbors, find_another: bool = False) -> None:
        """
        Modified Social-Force 기반 목표 결정:
            · self.exit_belief = {"idx": 출구 index, "score": S_ij, "alpha": hop}
            · self.now_goal    = [x, y]  (다음 time-step 까지 유효한 가상 목표)
        """
        # ────────── 파라미터 ──────────
        ROBOT_R = ROBOT_BODY_RADIUS
        VISION_R = AGENT_VISION
        AGENT_R = AGENT_VISION
        ROBOT_R = ROBOT_VISION
        EXIT_CONFIRM_R = EXIT_CONFIRM_RADIUS
        P_robot_following = 1 #로봇을 따라갈 확률
        P_neighbor_following = 0.7 #군중을 따라갈 확률

        best_idx = None
        best_score = -float("inf")

        # (1) 시야 기반으로 '보이는 출구' 후보 수집
        visible = self.model.visible_exits(self.xy, self.vision_radius)

        best_idx = None
        best_score = -float("inf")

        # (2) 보이는 출구가 있으면 그 중 점수 최고를 직접 인지로 채택
        for idx in visible:
            # "가까우면 확신" 같은 조건을 유지하고 싶으면 아래 d 조건을 추가로 걸어도 됨
            # d = self.model.distance_to_exit_idx(self.xy, idx)  # 있으면 메쉬 거리 추천
            # if d > (EXIT_CONFIRM_R + EXIT_CONFIRM_RADIUS_BONUS): 
            #     continue

            s = self.model.exit_score(self, idx, alpha=0)
            if s > best_score:
                best_score = s
                best_idx = idx

        if best_idx is not None:
            self.exit_belief = {"idx": best_idx, "score": best_score, "alpha": 0}
        else:
            self.exit_belief = None

        if best_idx is not None:
            self.exit_belief = {"idx": best_idx, "score": best_score, "alpha": 0}
        else:
            self.exit_belief = None

            
        if best_idx is not None:
            self.exit_belief = {"idx" : best_idx, "score": best_score, "alpha": 0}
        else:
            self.exit_belief = None
        for nb in neighbors:
            if nb.exit_belief:
                alpha = nb.exit_belief["alpha"] + 1
                s = self.model.exit_score(self, nb.exit_belief["idx"], alpha=alpha)
                if (self.exit_belief is None) or (s > self.exit_belief["score"]):
                    self.exit_belief = {"idx": nb.exit_belief["idx"], "score": s, "alpha": alpha}

        # ─ 2단계: 출구 정보가 있으면 그 출구, 없으면 탐험(Random walk) ─
        if self.exit_belief:
            ex_idx = self.exit_belief["idx"]
            _, q, d = self.model.nearest_exit(self.xy)  # 또는 ex_idx만 대상으로 따로 구해도 됨

            if d < 1.0:
                gx, gy = self.model.goal_point_into_exit(ex_idx, self.xy, eps=0.6)
                self.now_goal = [gx, gy]
            else:
                self.now_goal = [q[0], q[1]]

            return

        # ─ 4단계: 행동 타입 결정 (로봇/이웃/마이웨이) ─
        #robot_d = self.point_to_point_distance(self.xy, self.model.robot.xy) # 이거 좀 부정확함
        # if self.model.robot_version == 'N':
        #     robot_d = 9999999999
        # else:
        #     robot_d = math.sqrt(pow(self.xy[0]-self.model.robots[self.following_robot_index].xy[0], 2) + pow(self.xy[1]-self.model.robots[self.following_robot_index].xy[1], 2))
        
        self.met_robot_list = []
        following = 5
        the_number_of_crowds_robot_following = -1
        nearest_robot_d = 9999999999
        if self.model.robot_version == "N":
            nearest_robot_d = 9999999999
        else:
            for robo in self.model.robots:
                d = math.sqrt(pow(self.xy[0]-robo.xy[0], 2) + pow(self.xy[1]-robo.xy[1], 2))
                if d < nearest_robot_d:
                    nearest_robot_d = d
                if d < ROBOT_R:
                    self.met_robot_list.append(robo)
                    num_crowds = self.model.agents_near_robot_num_robot_index(robo.robot_index)
                    if num_crowds > the_number_of_crowds_robot_following:
                        the_number_of_crowds_robot_following = num_crowds
                        following = robo.robot_index
        if following !=5 :
            self.following_robot_index = following                 
    
        
        
        
        
        if(len(self.met_robot_list) == 0 and self.type==0 ): ### 로봇을 따라가던 애가 반경을 벗어나면 flag = 0
            self.decision_flag = 0
        if(self.decision_flag == 0 or len(self.met_robot_list) > 0): 
            #print(f"Agent{self.unique_id} 는 새로운 결정을 내리기로 했습니다.")

            if(len(self.met_robot_list) > 0 and self.model.robot_mode == "GUIDE"):  ####### 2.3*ROBOT_R 뭐임?
                if random.random() < P_robot_following: 
                    #print(f"Agent{self.unique_id} 는 로봇을 따라갑니다!")
                    self.type = 0
                    #print(self.following_robot_index)
                    self.now_goal = self.model.robots[self.following_robot_index].xy[:]
                    self.is_effected_by_robot = 1
                else:
                    self.type = 1
                    #print(f"Agent{self.unique_id} 가 로봇을 외면했습니다! - My Way")
                
            else :
                followable_neighbors = []
                for n in neighbors:
                    if (n.type != 2): #서로가 서로를 따라갈 수는 없음
                        followable_neighbors.append(n)
                if(len(followable_neighbors) == 0): ########## 이 경우는 마지막 agent에만 해당되는 거?
                    #print(f"Agent{self.unique_id} 는 주위에 아무것도 없습니다. - My Way")
                    self.type = 1 #따라갈 군중이 없으니 my-way
                else: # 따라갈 군중이 있음
                    if random.random() < (1-P_neighbor_following): #따라갈 군중이 있어도 제 갈길 가는 Agent
                        #print(f"Agent{self.unique_id} 가 이웃을 외면했습니다!")
                        self.type = 1
                    else: # 이웃 군중 따라가는 Agent
                        self.type = 2
                        self.follow_agent_id = followable_neighbors[0].unique_id # 이제 가장 믿을만한 이웃을 고를거임, 일단 초기화
                        max_score = -99999
                        for n in followable_neighbors:
                            dist = self.point_to_point_distance(self.xy, n.xy)
                            score = 0
                            if (n.exit_belief):
                                score = n.exit_belief["score"]
                            else: # 이웃한테 탈출구 정보가 없으면 일단 후순위
                                dist = self.point_to_point_distance(self.xy, n.xy)
                                score = -1000 - dist # 후순위 이웃 중 자기한테 가까울수록 신뢰함
                            if score > max_score:
                                self.follow_agent_id = n.unique_id 
                        #print(f"Agent{self.unique_id} 가 Agent{self.follow_agent_id} 를 따라갑니다!")
            self.decision_flag = self.decision_period
            
        else:
            self.decision_flag -= 1

        if self.type==0:
            self.decision_flag = 5
            self.now_goal = self.model.robots[self.following_robot_index].xy 

        elif self.type==1:
            now_mesh = self.choice_safe_mesh(self.xy)

            if(self.now_pointing_mesh != None):   
            
                pointing_mesh_center = ((self.now_pointing_mesh[0][0]+self.now_pointing_mesh[1][0]+self.now_pointing_mesh[2][0])/3, 
                                        (self.now_pointing_mesh[0][1]+self.now_pointing_mesh[1][1]+self.now_pointing_mesh[2][1])/3)
                # print(f"pointing_mesh_center : {pointing_mesh_center}")
                if (math.sqrt(pow(self.xy[0]-pointing_mesh_center[0],2)+pow(self.xy[1]-pointing_mesh_center[1],2))<2): #향햐던 mesh에 도달했을 때
                    self.now_pointing_mesh = None

            if (self.now_pointing_mesh == None): # 향하던 mesh에 도달하면 -> None으로 설정 -> 다시 탐색하게 하기
                self.now_pointing_mesh = random.choice(self.model.pure_mesh)

            self.now_goal = self._explore_randomly(now_mesh)
            
        elif self.type==2:
            self.now_goal = self.model.return_agent_id(self.follow_agent_id).xy


        # type==2 일 때 추종 대상의 실시간 위치로 업데이트
        if self.type == 2:
            self.now_goal = self.model.return_agent_id(
                                self.follow_agent_id).xy


    def _explore_randomly(self, now_mesh):
        goal_mesh = self.now_pointing_mesh
        nxt = self.model.next_mesh_from_to(now_mesh, goal_mesh)
        if nxt is None: 
            return [self.xy[0], self.xy[1]]
        return [ (nxt[0][0]+nxt[1][0]+nxt[2][0])/3.0,
                (nxt[0][1]+nxt[1][1]+nxt[2][1])/3.0 ]


  
class RobotAgent(CrowdAgent):
    def __init__(self, unique_id, model, pos, type1, robot_index):
        super().__init__(unique_id, model, pos, type1)
        self.action = [0, 0, 0]
        self.past_xy = deque(maxlen=20)
        self.collision_check = 0
        self.detect_abnormal_order = 0
        self.is_game_finished = 0
        self.robot_index = robot_index

        self.robot_waypoint = [0, 0]
        self.now_exploration = 0

        self.acc = [0, 0]
        self.vel = [0, 0]
        self.body_radius = ROBOT_BODY_RADIUS
        self.vision_radius = ROBOT_VISION

        #self.model.space.add(self.unique_id, self.xy, self.radius, ref=self, vel=(0,0,0,0))

        self.desired_speed_a = 2
        self.target_agent = None

        self.angle = 0.0
        self.planner_target = None
        self.planner_speed = 0.0
        self.planner_active = False
        self.planner_fallback_vec = [0.0, 0.0]
        self.new_order_need = True
        self.ego_stack = FrameStackWithStep(4, FRAME_STEP)

        self.map_scale_x = 1
        self.map_scale_y = 1

        self.latest_ego_f = None
        self.latest_glob_f = None
        self.latest_ego_state = None
        self.latest_glob_state = None
        self.latest_robot_state = None
        self.latest_action = None
    
    # ------------------------------------------------------------
    # 외부에서 호출되는 단일 정책 함수
    # ------------------------------------------------------------

    # def robot_policy_go_and_back(self):
    #     if (self.target_agent == None):
    #         max_d = -1 
    #         max_d_ag = None
    #         for ag in self.model.crowds:
    #             if not ag.dead:
    #                 d = self.point_to_point_distance(self.xy, ag.xy)
    #                 if d > max_d:
    #                     max_d = d
    #                     max_d_ag = ag
    #         if max_d_ag is not None:
    #             self.target_agent = max_d_ag

    #     if (self.target_agent == None):
    #         return
        
    #     if (self.target_agent.dead):
    #         self.target_agent = None
    #         return

    #     goal = [0, 0]
    #     if (self.point_to_point_distance(self.xy, self.target_agent.xy) < 5):
    #         goal = self.model.exit_point[0]
    #     else :
    #         goal = self.target_agent.xy

    #     goal_mesh = self.model.match_grid_to_mesh[int(round(goal[0])), int(round(goal[1]))]
    #     now_mesh = self.model.match_grid_to_mesh[int(round(self.xy[0])), int(round(self.xy[1]))]
    #     next_mesh = self.model.next_vertex_matrix[now_mesh][goal_mesh]
    #     if(now_mesh == next_mesh):
    #         goal_x = goal[0] - self.xy[0]
    #         goal_y = goal[1] - self.xy[1]
            
    #     else:
    #         next_mesh_middle = ((next_mesh[0][0]+next_mesh[1][0]+next_mesh[2][0])/3, (next_mesh[0][1]+next_mesh[1][1]+next_mesh[2][1])/3)
    #         goal_x = next_mesh_middle[0] - self.xy[0]
    #         goal_y = next_mesh_middle[1] - self.xy[1]

    #     goal_x = ROBOT_SPEED_MAX * goal_x / math.sqrt(pow(goal_x, 2) + pow(goal_y, 2))
    #     goal_y = ROBOT_SPEED_MAX * goal_y / math.sqrt(pow(goal_x, 2) + pow(goal_y, 2))
    #     self.receive_action([goal_x, goal_y])

    def _safe_normalize(self, x, y, eps=1e-9):
        n = math.hypot(x, y)
        if n < eps:
            return 0.0, 0.0
        return x / n, y / n 

    # def robot_policy_going_exit(self):
    #     ed_idx, q, d = self.model.nearest_exit(self.xy)
    #     goal = q
    #     if self.point_to_point_distance(self.xy, goal) < 2:
    #         self.receive_action([0, 0])  # stop
        
    #     else :
    #         goal_mesh = self.model.match_grid_to_mesh[int(round(goal[0])), int(round(goal[1]))]
    #         now_mesh = self.model.match_grid_to_mesh[int(round(self.xy[0])), int(round(self.xy[1]))]
    #         next_mesh = self.model.next_vertex_matrix[now_mesh][goal_mesh]
    #         if(now_mesh == next_mesh):
    #             goal_x = goal[0] - self.xy[0]
    #             goal_y = goal[1] - self.xy[1]
                
    #         else:
    #             next_mesh_middle = ((next_mesh[0][0]+next_mesh[1][0]+next_mesh[2][0])/3, (next_mesh[0][1]+next_mesh[1][1]+next_mesh[2][1])/3)
    #             goal_x = next_mesh_middle[0] - self.xy[0]
    #             goal_y = next_mesh_middle[1] - self.xy[1]

    #         goal_x = 1* goal_x / math.sqrt(pow(goal_x, 2) + pow(goal_y, 2))
    #         goal_y = 1* goal_y / math.sqrt(pow(goal_x, 2) + pow(goal_y, 2))
    #         self.receive_action([goal_x, goal_y])

    def robot_policy_going_exit(self):
        ed_idx, q, d = self.model.nearest_exit(self.xy)
        goal = q

        dx = goal[0] - self.xy[0]
        dy = goal[1] - self.xy[1]
        dist = math.hypot(dx, dy)

        if dist < 0.5:
            self.receive_action([0.0, 0.0, 0.0])
            return

        world_theta = math.atan2(dy, dx)
        rel_theta = world_theta - self.angle

        while rel_theta > math.pi:
            rel_theta -= 2 * math.pi
        while rel_theta < -math.pi:
            rel_theta += 2 * math.pi

        self.receive_action([rel_theta, dist, ROBOT_SPEED_MAX])
    
    def receive_action_from_policy(self, action):
        """
        action = [radius, dx, dy, speed]
        dx, dy: World 좌표계 기준의 방향 벡터 (Normalized 권장)
        """
        # 1. 분해
        r, dx, dy, spd = action[0], action[1], action[2], action[3]
        
        # 2. World 좌표계 기준의 절대 각도(atan2) 계산
        # math.atan2(y, x)는 [-pi, pi] 범위의 라디안을 반환합니다.
        world_theta = math.atan2(dy, dx)
        
        # 3. 로봇의 현재 헤딩(self.angle)을 뺀 상대 각도 계산
        # Relative Theta = World Theta - Current Robot Angle
        relative_theta = world_theta - self.angle
        
        # 각도 범위를 [-pi, pi] 사이로 정규화 (선택 사항이지만 안전함)
        relative_theta = (relative_theta + math.pi) % (2.0 * math.pi) - math.pi

        # 4. 로봇 형식에 맞게 [relative_theta, radius, target_speed] 전달
        formatted_action = [relative_theta, r, spd]
        
        self.receive_action(formatted_action)


    def receive_action(self, action):
        """
        action = [theta, radius, target_speed]
        theta: 현재 self.angle에 더해질 상대 각도 (라디안)
        """
        if action is None:
            return np.array(self.action, dtype=float)

        # 검증 (입력은 여전히 3개 [theta, r, spd] 임)
        if len(action) != 3:
            raise ValueError("Robot action must be [theta, radius, target_speed]")

        theta, radius, target_speed = float(action[0]), float(action[1]), float(action[2])

        # Clip 및 값 정제
        radius = max(0.0, radius)
        target_speed = max(0.0, min(target_speed, ROBOT_SPEED_MAX))

        self.action[0] = theta
        self.action[1] = radius
        self.action[2] = target_speed

        # 상대 각도를 절대 각도로 변환
        world_theta = self.angle + theta

        # 목표 지점(Waypoint) 계산
        goal_x = self.xy[0] + radius * math.cos(world_theta)
        goal_y = self.xy[1] + radius * math.sin(world_theta)

        self.planner_target = [goal_x, goal_y]
        self.planner_speed = target_speed
        
        # 아주 작은 움직임은 무시
        self.planner_active = (radius > 1e-5) and (target_speed > 1e-5)

        # 폴백 벡터 업데이트
        fx = math.cos(world_theta)
        fy = math.sin(world_theta)
        self.planner_fallback_vec = [fx, fy]

        return np.array(self.action, dtype=float)
    
    def _cancel_planner(self, stop_velocity=False):
        self.planner_target = None
        self.planner_speed = 0.0
        self.planner_active = False
        self.planner_fallback_vec = [0.0, 0.0]
        if stop_velocity:
            self.vel = [0.0, 0.0]
    
    def _goal_reached(self, goal, tol=0.5):
        if goal is None:
            return True
        if math.hypot(self.xy[0] - goal[0], self.xy[1] - goal[1]) <= tol:
            return True
        else:
            return False

    def _new_goal_trigger(self, goal, tol=1):
        if goal is None:
            return True
        if math.hypot(self.xy[0] - goal[0], self.xy[1] - goal[1]) <= tol:
            self.new_order_need = True
        else:
            return False

    def _planner_direction_to_goal(self, goal):
        if goal is None:
            return 0.0, 0.0

        gx = goal[0] - self.xy[0]
        gy = goal[1] - self.xy[1]
        dist_to_goal = math.hypot(gx, gy)
        fallback_dx, fallback_dy = self._safe_normalize(gx, gy)

        try:
            if [int(round(goal[0])), int(round(goal[1]))] not in self.model.match_grid_to_mesh.keys():
                return fallback_dx, fallback_dy
            goal_mesh = self.model.match_grid_to_mesh[int(round(goal[0])), int(round(goal[1]))] 
            now_mesh = self.choice_safe_mesh(self.xy)

            # 1) 가시성 최적화: 목표 지점까지 장애물 없이 직선으로 갈 수 있다면 바로 직진
            # (이 함수는 따로 구현되어 있다고 가정하거나, 단순히 dist가 가까우면 수행)
            if now_mesh == goal_mesh :
                return fallback_dx, fallback_dy

            next_mesh = self.model.next_vertex_matrix[now_mesh][goal_mesh]
            if next_mesh is None:
                return fallback_dx, fallback_dy

            # 2) 공유 변(Shared Edge) 찾기
            # now_mesh와 next_mesh의 정점들 중 공통된 2개의 정점을 찾습니다.
            shared_vertices = [v for v in now_mesh if v in next_mesh]

            if len(shared_vertices) >= 2:
                # 공유 변의 양 끝점
                v1, v2 = shared_vertices[0], shared_vertices[1]
                
                # 변의 중앙점 대신, 로봇 위치에서 이 선분(v1-v2)에 내린 수선의 발(Clamped)을 찾음
                # 이것이 '가장 가까운 통로' 지점이 됩니다.
                target_pt = self._closest_point_on_segment(v1, v2, self.xy)
                
                # 만약 로봇이 이미 이 타겟 근처에 있다면, 다음 메쉬의 중심으로 타겟을 옮겨서 '통과'하게 만듦
                if math.hypot(target_pt[0] - self.xy[0], target_pt[1] - self.xy[1]) < 0.2:
                    target_pt = (
                        (next_mesh[0][0] + next_mesh[1][0] + next_mesh[2][0]) / 3.0,
                        (next_mesh[0][1] + next_mesh[1][1] + next_mesh[2][1]) / 3.0,
                    )
            else:
                # 공유 변을 찾지 못한 경우 기존 무게중심 방식 유지
                target_pt = (
                    (next_mesh[0][0] + next_mesh[1][0] + next_mesh[2][0]) / 3.0,
                    (next_mesh[0][1] + next_mesh[1][1] + next_mesh[2][1]) / 3.0,
                )

            dx = target_pt[0] - self.xy[0]
            dy = target_pt[1] - self.xy[1]
            return self._safe_normalize(dx, dy)

        except Exception:
            return fallback_dx, fallback_dy

    def _closest_point_on_segment(self, v1, v2, p):
        """선분 v1-v2 위에서 점 p와 가장 가까운 점을 반환"""
        dx, dy = v2[0] - v1[0], v2[1] - v1[1]
        mag_sq = dx**2 + dy**2
        if mag_sq < 1e-9: return v1
        
        t = ((p[0] - v1[0]) * dx + (p[1] - v1[1]) * dy) / mag_sq
        t = max(0.1, min(0.9, t))  # 벽에 너무 붙지 않도록 0.1~0.9 사이로 클램핑(여유 공간)
        return (v1[0] + t * dx, v1[1] + t * dy)
        
    def robot_policy_Q(self):
        K_WALL = 1500
        LAMBDA_A = 0.35

        self.previous_danger = getattr(self, "danger", 1e9)
        self.danger = self.model.distance_to_exit(self.xy)

        if self.model.alived_agents() < 1:
            self.is_game_finished = 1

        if self.robot_initialized == 0:
            self.robot_initialized = 1
            return (self.xy[0], self.xy[1])

        self.past_xy.append(tuple(self.xy))
        time_step = ROBOT_TIME_STEP
        self.model.robot_mode = "GUIDE"

        # 1) collision check
        #self._compute_wall_collision_flag()

        # 4) choose desired moving direction
        if self.planner_active and self.planner_target is not None:
            dir_x, dir_y = self._planner_direction_to_goal(self.planner_target)
            desired_speed = self.planner_speed
        else:
            dir_x, dir_y = 0.0, 0.0
            desired_speed = 0.0

        # update heading if there is meaningful motion direction
        if abs(dir_x) > 1e-9 or abs(dir_y) > 1e-9:
            self.angle = math.atan2(dir_y, dir_x)

        intend_force = 30
        desired_force = [
            intend_force * desired_speed * dir_x,
            intend_force * desired_speed * dir_y,
        ]

        # wall repulsion
        F_wx = 0.0
        F_wy = 0.0
        p = Point(self.xy[0], self.xy[1])

        obstacle_polys = self.model._obstacle_polys.copy()
        obstacle_polys.append(
            Polygon([
                (0, 0),
                (self.model.width - 1, 0),
                (self.model.width - 1, self.model.height - 1),
                (0, self.model.height - 1)
            ])
        )
        self.collision_check = 0
        for poly in obstacle_polys:
            d = poly.exterior.distance(p)
            #print(d)
            if d <= self.body_radius * 2.2:
                self.collision_check = 1
            if d > 3 * self.body_radius:
                continue

            q = poly.exterior.interpolate(poly.exterior.project(p))
            dx = self.xy[0] - q.x
            dy = self.xy[1] - q.y
            dist = math.hypot(dx, dy) or 1e-9
            nx, ny = dx / dist, dy / dist

            mag = K_WALL * math.exp(-(d / max(LAMBDA_A, 1e-6)))
            F_wx += mag * nx
            F_wy += mag * ny

        # 2) collision -> cancel current planner
        if self.collision_check == 1:
            #print("충돌함!")
            self.new_order_need = True
            self._cancel_planner(stop_velocity=False)

        # 3) if target reached, stop planner
        self._new_goal_trigger(self.planner_target, tol=2)
        if self.planner_active and self._goal_reached(self.planner_target, tol=0.3):
            self._cancel_planner(stop_velocity=False)


        F_x = desired_force[0] + F_wx
        F_y = desired_force[1] + F_wy

        vel = [0.0, 0.0]
        vel[0] = F_x / self.mass
        vel[1] = F_y / self.mass

        # clamp speed
        spd = math.hypot(vel[0], vel[1])
        if spd > ROBOT_SPEED_MAX and spd > 1e-9:
            s = ROBOT_SPEED_MAX / spd
            vel[0] *= s
            vel[1] *= s

        future_xy = [self.xy[0] + vel[0] * time_step, self.xy[1] + vel[1] * time_step]

        move_dx = future_xy[0] - self.xy[0]
        move_dy = future_xy[1] - self.xy[1]

        self.vel[0] = vel[0]
        self.vel[1] = vel[1]
        self.xy = future_xy
        self.model.space.clamp(self.xy)

        if abs(move_dx) > 1e-9 or abs(move_dy) > 1e-9:
            self.angle = math.atan2(move_dy, move_dx)

        return tuple(self.xy)
    

    def update_latest_state_images(
        self,
        ego_f,
        glob_f,
        ego_state=None,
        glob_state=None,
        robot_state=None,
        action=None
    ):
        """
        run_sim.py visualization/debug용으로
        현재 policy input state를 robot 객체 안에 저장한다.
        """
        self.latest_ego_f = None if ego_f is None else ego_f.copy()
        self.latest_glob_f = None if glob_f is None else glob_f.copy()

        self.latest_ego_state = None if ego_state is None else ego_state.copy()
        self.latest_glob_state = None if glob_state is None else glob_state.copy()
        self.latest_robot_state = None if robot_state is None else robot_state.copy()

        if action is not None:
            self.latest_action = np.array(action, dtype=np.float32).copy()

    def shared_visible_crowds(self, radius=None):
        return self.model.get_shared_observable_crowds(
            radius=ROBOT_VISION if radius is None else radius
        )
    
    def can_any_robot_observe_agent(self, agent, radius=None):
        return self.model.is_crowd_observable_by_any_robot(
            agent, radius=ROBOT_VISION if radius is None else radius
        )