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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import *



AGENT_TIME_STEP = 0.25
ROBOT_TIME_STEP = 0.25


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

        self.desired_speed_a = np.random.normal(3, 0.2)*1.2 # agent의 desired_speed, 평균 1.5m/s, 표준 편차 0.2m/s

        self.is_effected_by_robot = 0
        self.blocked = False

        self.decision_flag = random.randint(1,5) # self.decision_flag == 0 -> 결정 다시 내림
        self.decision_period = random.randint(15,35) #self.decision_period == 0 -> 결정 다시 내림, 군중 마다 얼마만큼의 시간동안 자신의 결정을 번복하지 않는가 모델링


        self.model.robot_mode = "GUIDE"
 
        self.exit_belief = None       # {"idx": int, "score": float, "alpha": int}
        self.life_time = 0
        self.body_radius = AGENT_BODY_RADIUS
        self.vision_radius = AGENT_VISION


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

    def choice_safe_mesh(self, point):
        m = self.model.find_mesh(point)
        if m is None:
        # 소폭 난수 지터로 재시도
            for _ in range(6):
                jitter = (random.uniform(-0.4,0.4), random.uniform(-0.4,0.4))
                m = self.model.find_mesh((point[0]+jitter[0], point[1]+jitter[1]))
                if m is not None:
                    break
        if m is None:
            raise RuntimeError("safe mesh를 찾지 못했습니다.")
        return m



    def mesh_to_mesh_distance(self, point1, point2):
        point1_mesh = self.choice_safe_mesh(point1)
        point2_mesh = self.choice_safe_mesh(point2)

        return self.model.distance[point1_mesh][point2_mesh]

    def point_to_point_distance(self, p1, p2):
        m1 = self.choice_safe_mesh(p1)
        m2 = self.choice_safe_mesh(p2)
        if self.model.next_mesh_from_to(m1, m2) is None:
            return 1e12
        dist = 0.0
        cur = m1
        c1 = ((cur[0][0]+cur[1][0]+cur[2][0])/3.0, (cur[0][1]+cur[1][1]+cur[2][1])/3.0)
        dist += math.hypot(p1[0]-c1[0], p1[1]-c1[1])
        while cur != m2:
            nxt = self.model.next_mesh_from_to(cur, m2)
            c0 = ((cur[0][0]+cur[1][0]+cur[2][0])/3.0, (cur[0][1]+cur[1][1]+cur[2][1])/3.0)
            cN = ((nxt[0][0]+nxt[1][0]+nxt[2][0])/3.0, (nxt[0][1]+nxt[1][1]+nxt[2][1])/3.0)
            dist += math.hypot(cN[0]-c0[0], cN[1]-c0[1])
            cur = nxt
        c2 = ((cur[0][0]+cur[1][0]+cur[2][0])/3.0, (cur[0][1]+cur[1][1]+cur[2][1])/3.0)
        dist += math.hypot(p2[0]-c2[0], p2[1]-c2[1])
        return dist
            

    
    def move(self) -> None:
        """Handles the movement behavior.
        Here the agent decides   if it moves,
        drinks the heal potion,
        or attacks other agent."""
        if(self.model.robot_version != 'N'):
            cells_with_agents = []
            robot_xy = [self.model.robot.xy[0], self.model.robot.xy[1]]

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

    def choice_near_goal(self, pos):
        shortest_distance = float('inf')
        near_goal = None
        for i in self.model.exit_point:
            distance = self.mesh_to_mesh_distance(i, pos)
            for i in self.model.exit_point:
                d = self.mesh_to_mesh_distance(i, pos)
                if d < shortest_distance:
                    shortest_distance = d
                    near_goal = i
        return near_goal  

    def choice_near_exit(self):
        shortest_distance = 9999999999
        near_exit = None
        for i in self.model.exit_point:
            if (self.mesh_to_mesh_distance(self.xy, i) < shortest_distance):
                shortest_distance = self.mesh_to_mesh_distance(self.xy, i)
                near_exit = i
        return near_exit

    
    def _wall_repulsion(self):
        from shapely.geometry import Point
        Fwx = Fwy = 0.0
        p = Point(self.xy[0], self.xy[1])
        for poly in self.model._obstacle_polys:
            d = poly.exterior.distance(p)
            if d < self.body_radius * 1.5:
                q = poly.exterior.interpolate(poly.exterior.project(p))
                dx = self.xy[0]-q.x
                dy = self.xy[1]-q.y
                dist = math.hypot(dx, dy) or 1e-9
                nx, ny = dx/dist, dy/dist
                mag = 200.0 * math.exp(-(d/0.2))
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
        BODY_RADIUS = 0.5             # 군중 몸 반지름 [cell] 0.25m로 설정 -> 0.5칸이 되어야 0.25
        ROBOT_BODY_RADIUS = 1       # 로봇 몸 반지름 [cell] 0.25m로 설정 ->0.5칸이 되어야 0.25
        WALL_RADIUS = 1             # 격자벽을 둥근 장애물로 근사
        # 접촉(법선) 스프링/감쇠, 접선 마찰
        KN = 1.2e5                   # 법선 스프링 상수, modified crowd simulation 논문에선 1.2*10^5
        CN =  1000                     # 법선 점성(접근속도 감쇠)
        MU_T = 2.5e5                    # 접선 마찰(미끄럼 속도 감쇠) modifided crowd simulation 논문에선 2.4*10^5
        # 지수형 반발은 약화(근거리에서만 의미)
        K_AGENT = 200 # modified 참고
        K_WALL  = 200 # modified 참고
        LAMBDA_A = 0.2 # modifided 참고
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
        F_rep_x = 0.0; F_rep_y = 0.0

        # ---- 접촉(비관통) + 마찰 모델(핵심 추가) ----
        F_contact_x = 0.0; F_contact_y = 0.0
        F_fric_x    = 0.0; F_fric_y    = 0.0

        r_i = BODY_RADIUS
        # 자기 상태 (속도)
        v_ix, v_iy = self.vel[0], self.vel[1]
        
        # for nb in near_agents:
        #     if nb is self or getattr(nb, "dead", False):
        #         continue

        #     dx = self.xy[0] - nb.xy[0]
        #     dy = self.xy[1] - nb.xy[1]
        #     d  = math.hypot(dx, dy)
        #     if d < 1e-9:
        #         # 완전 겹침 초기 해소(랜덤 툭 치기)
        #         jx, jy = (1.0, -1.0) if random.random() < 0.5 else (-1.0, 1.0)
        #         F_contact_x += jx * KN * 0.01
        #         F_contact_y += jy * KN * 0.01
        #         continue

        #     ux, uy = dx/d, dy/d  # (nb -> self) 법선 방향
        #     nb_R = getattr(nb, "radius", BODY_RADIUS)
        #     r_sum = self.radius + nb_R

        #     # 원거리 지수 반발
        #     mag = K_AGENT * math.exp((r_sum-d) / max(LAMBDA_A, 1e-6))
        #     F_rep_x += mag * ux
        #     F_rep_y += mag * uy

        #     # # 1) 원거리 지수 반발(부드러운 회피)
        #     # if getattr(nb, "type", None) in (11, 9):
        #     #     mag = K_WALL * math.exp((r_sum - d) / LAMBDA_A)
        #     #     F_rep_x += mag * ux
        #     #     F_rep_y += mag * uy
        #     # else:
        #     #     mag = K_AGENT * math.exp((r_sum - d) / LAMBDA_A)
        #     #     F_rep_x += mag * ux
        #     #     F_rep_y += mag * uy

        #     # 2) 근거리 접촉(비관통) + 점성 감쇠 + 접선 마찰
        #     if d < r_sum:
        #         # 침투량(양수면 겹침)
        #         penetration = (r_sum - d)
        #         # (a) 법선 스프링
        #         Fn_k = KN * penetration
        #         # (b) 상대속도에 대한 법선 감쇠
        #         v_jx = getattr(nb, "vel", [0,0])[0] if hasattr(nb, "vel") else 0.0
        #         v_jy = getattr(nb, "vel", [0,0])[1] if hasattr(nb, "vel") else 0.0
        #         rel_vx, rel_vy = (v_ix - v_jx), (v_iy - v_jy)
        #         v_n = rel_vx*ux + rel_vy*uy        # 법선 성분
                
        #         if v_n < 0:
        #             restitution = 0.2
        #             drop = (1.0-restitution)*v_n
        #             self.vel[0] -= drop*ux
        #             self.vel[1] -= drop*uy

        #         Fn_c = -CN * v_n                   # 접근할수록(음수) + 방향은 법선
        #         Fn = max(Fn_k + Fn_c, 0.0)         # 법선 힘은 음수가 되지 않게

        #         F_contact_x += Fn * ux
        #         F_contact_y += Fn * uy

        #         # (c) 접선 방향(미끄럼) 마찰: v_t = rel_v - v_n n
        #         vt_x = rel_vx - v_n*ux
        #         vt_y = rel_vy - v_n*uy
        #         F_fric_x += -MU_T * vt_x
        #         F_fric_y += -MU_T * vt_y
        
        BETA=0
        decay = 0.95
        self.vel[0] *= decay
        self.vel[1] *= decay
        # ---- 공기저항(속도 감쇠) ----
        F_drag_x = -BETA * self.vel[0]
        F_drag_y = -BETA * self.vel[1]

        # ---- 총합 힘 ----
        F_x = F_des_x + F_rep_x + F_contact_x + F_fric_x + F_drag_x
        F_y = F_des_y + F_rep_y + F_contact_y + F_fric_y + F_drag_y

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


        self.xy = self.swept_move(self.xy, self.vel, dt)
        self.model.space.clamp(self.xy)
        self.model.space.move(self.unique_id, self.xy)

        self.direction = [self.vel[0], self.vel[1]]

        return tuple(self.xy)

    
    def which_goal_agent_want(self, neighbors, find_another: bool = False) -> None:
        """
        Modified Social-Force 기반 목표 결정:
            · self.exit_belief = {"idx": 출구 index, "score": S_ij, "alpha": hop}
            · self.now_goal    = [x, y]  (다음 time-step 까지 유효한 가상 목표)
        """
        ROBOT_BODY_RADIUS = 1
        # ────────── 파라미터 ──────────
        ROBOT_R = ROBOT_BODY_RADIUS
        VISION_R = AGENT_VISION
        AGENT_R = AGENT_VISION
        ROBOT_R = ROBOT_VISION
        EXIT_CONFIRM_R = EXIT_CONFIRM_RADIUS
        P_robot_following = 1 #로봇을 따라갈 확률
        P_neighbor_following = 0.7 #군중을 따라갈 확률

        # ─ 0단계: 직접 출구를 보면 α=0 정보 기록 ─
        for idx, center in enumerate(self.model.exit_point):
            #print(f"센터 : {center}")
            #print(f"거리 : {idx}", self.point_to_point_distance(self.xy, center))
            #if self.point_to_point_distance(self.xy, center) < VISION_R:
            if math.sqrt(pow(self.xy[0]-center[0], 2) + pow(self.xy[1]-center[1], 2)) < EXIT_CONFIRM_R:
                s = self.model.exit_score(self, idx, alpha=0)
                self.exit_belief = {"idx": idx, "score": s, "alpha": 0}

        for nb in neighbors:
            if nb.exit_belief:
                alpha = nb.exit_belief["alpha"] + 1
                s = self.model.exit_score(self, nb.exit_belief["idx"], alpha=alpha)
                if (self.exit_belief is None) or (s > self.exit_belief["score"]):
                    self.exit_belief = {"idx": nb.exit_belief["idx"], "score": s, "alpha": alpha}

        # ─ 2단계: 출구 정보가 있으면 그 출구, 없으면 탐험(Random walk) ─
        if self.exit_belief:                                       # 정보 有
            self.now_goal = self.model.exit_point[self.exit_belief["idx"]][:]
            #print("출구로 향하자!")
            return

        # ─ 4단계: 행동 타입 결정 (로봇/이웃/마이웨이) ─
        #robot_d = self.point_to_point_distance(self.xy, self.model.robot.xy) # 이거 좀 부정확함
        if self.model.robot_version == 'N':
            robot_d = 9999999999
        else:
            robot_d = math.sqrt(pow(self.xy[0]-self.model.robot.xy[0], 2) + pow(self.xy[1]-self.model.robot.xy[1], 2))
        if(robot_d >= ROBOT_R and self.type==0 ): ### 로봇을 따라가던 애가 반경을 벗어나면 flag = 0
            self.decision_flag = 0
        if(self.decision_flag == 0 or robot_d < ROBOT_R): 
            #print(f"Agent{self.unique_id} 는 새로운 결정을 내리기로 했습니다.")
            if(robot_d < ROBOT_R and self.model.robot_mode == "GUIDE" and self.point_to_point_distance(self.model.robot.xy, self.xy)<2.3*ROBOT_R):  ####### 2.3*ROBOT_R 뭐임?
                if random.random() < P_robot_following: 
                    #print(f"Agent{self.unique_id} 는 로봇을 따라갑니다!")
                    self.type = 0
                    self.now_goal = self.model.robot.xy[:]
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
                            dist = self.point_to_point_distance
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
            self.now_goal = self.model.robot.xy 

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
    

    def __init__(self, unique_id, model, pos, type1):
        super().__init__(unique_id, model, pos, type1)
        AGENT_RADIUS = 1
        self.action = [0, 0, "GUIDE"]
        self.past_xy = deque(maxlen=20)
        self.collision_check = 0
        self.detect_abnormal_order = 0
        self.is_game_finished = 0

        self.robot_waypoint = [0, 0]
        self.now_exploration = 0

        self.acc = [0, 0]
        self.vel = [0, 0]
        self.body_radius = ROBOT_BODY_RADIUS
        self.vision_radius = ROBOT_VISION

        #self.model.space.add(self.unique_id, self.xy, self.radius, ref=self, vel=(0,0,0,0))

        self.desired_speed_a = 4
        self.target_agent = None
    
    # ------------------------------------------------------------
    # 외부에서 호출되는 단일 정책 함수
    # ------------------------------------------------------------

    def robot_policy_go_and_back(self):
        if (self.target_agent == None):
            max_d = -1 
            max_d_ag = None
            for ag in self.model.crowds:
                if not ag.dead:
                    d = self.point_to_point_distance(self.xy, ag.xy)
                    if d > max_d:
                        max_d = d
                        max_d_ag = ag
            if max_d_ag is not None:
                self.target_agent = max_d_ag

        if (self.target_agent == None):
            return
        
        if (self.target_agent.dead):
            self.target_agent = None
            return

        goal = [0, 0]
        if (self.point_to_point_distance(self.xy, self.target_agent.xy) < 5):
            goal = self.model.exit_point[0]
        else :
            goal = self.target_agent.xy

        goal_mesh = self.model.match_grid_to_mesh[int(round(goal[0])), int(round(goal[1]))]
        now_mesh = self.model.match_grid_to_mesh[int(round(self.xy[0])), int(round(self.xy[1]))]
        next_mesh = self.model.next_vertex_matrix[now_mesh][goal_mesh]
        if(now_mesh == next_mesh):
            goal_x = goal[0] - self.xy[0]
            goal_y = goal[1] - self.xy[1]
            
        else:
            next_mesh_middle = ((next_mesh[0][0]+next_mesh[1][0]+next_mesh[2][0])/3, (next_mesh[0][1]+next_mesh[1][1]+next_mesh[2][1])/3)
            goal_x = next_mesh_middle[0] - self.xy[0]
            goal_y = next_mesh_middle[1] - self.xy[1]

        goal_x = 2* goal_x / math.sqrt(pow(goal_x, 2) + pow(goal_y, 2))
        goal_y = 2* goal_y / math.sqrt(pow(goal_x, 2) + pow(goal_y, 2))
        self.receive_action([goal_x, goal_y])


    def robot_policy_going_exit(self):
        goal = self.model.exit_point[0]
        if self.point_to_point_distance(self.xy, goal) < 2:
            self.receive_action([0, 0])  # stop
        
        else :
            goal_mesh = self.model.match_grid_to_mesh[int(round(goal[0])), int(round(goal[1]))]
            now_mesh = self.model.match_grid_to_mesh[int(round(self.xy[0])), int(round(self.xy[1]))]
            next_mesh = self.model.next_vertex_matrix[now_mesh][goal_mesh]
            if(now_mesh == next_mesh):
                goal_x = goal[0] - self.xy[0]
                goal_y = goal[1] - self.xy[1]
                
            else:
                next_mesh_middle = ((next_mesh[0][0]+next_mesh[1][0]+next_mesh[2][0])/3, (next_mesh[0][1]+next_mesh[1][1]+next_mesh[2][1])/3)
                goal_x = next_mesh_middle[0] - self.xy[0]
                goal_y = next_mesh_middle[1] - self.xy[1]

            goal_x = 2* goal_x / math.sqrt(pow(goal_x, 2) + pow(goal_y, 2))
            goal_y = 2* goal_y / math.sqrt(pow(goal_x, 2) + pow(goal_y, 2))
            self.receive_action([goal_x, goal_y])
    

    def receive_action(self, action):
                
        
        direction_probs = action[0]
        

        self.action[0] = action[0]
        self.action[1] = action[1]

        
        if(self.now_exploration == 1):
            print("exploration 중")
            if(self.robot_waypoint == [0, 0]):
                self.robot_waypoint = self.model.choice_random_waypoint()
            now_mesh = self.model.match_grid_to_mesh[int(round(self.xy[0])), int(round(self.xy[1]))]
            goal_mesh = self.model.match_grid_to_mesh[int(round(self.xy[0])), int(round(self.xy[1]))]
            next_mesh = self.model.next_vertex_matrix[now_mesh][goal_mesh]
            if(now_mesh == next_mesh):
                goal_x = self.robot_waypoint[0] - self.xy[0]
                goal_y = self.robot_waypoint[1] - self.xy[1]

            else:
                next_mesh_middle = ((next_mesh[0][0]+next_mesh[1][0]+next_mesh[2][0])/3, (next_mesh[0][1]+next_mesh[1][1]+next_mesh[2][1])/3)
                goal_x = next_mesh_middle[0] - self.xy[0]
                goal_y = next_mesh_middle[1] - self.xy[1]

            goal_d = math.sqrt(pow(goal_x,2) + pow(goal_y,2))
            goal_x = goal_x/goal_d
            goal_y = goal_y/goal_d
            self.action[0] = goal_x
            self.action[1] = goal_y
        

        return np.array(self.action)
    
    def robot_policy_Q(self):

        K_AGENT = 200
        LAMBDA_A = 0.2

        if(math.sqrt(pow(self.xy[0]-self.robot_waypoint[0], 2)+pow(self.xy[1]-self.robot_waypoint[1], 2))<2):
            self.now_exploration = 0
            self.robot_waypoint = [0, 0]

        self.previous_danger = getattr(self, "danger", 1e9)
        self.danger = 1e9
        for i in self.model.exit_point:
            self.danger = min(self.danger, self.point_to_point_distance([self.xy[0], self.xy[1]], i))
        
        if(self.model.alived_agents()< 1):
            self.is_game_finished = 1

        if(self.robot_initialized == 0 ):
            self.robot_initialized = 1
            return (self.model.robot.xy[0], self.model.robot.xy[1]) ## 오호라... 처음에 리스폰 되는 거 피하려고 
        self.past_xy.append(self.xy)

        time_step = ROBOT_TIME_STEP


        goal_x = 0
        goal_y = 0
        
        goal_x += self.action[0]
        goal_y += self.action[1]
        
        #print(f"robot desired go to {goal_x}, {goal_y}") 
        self.model.robot_mode = "GUIDE"

        intend_force = 15
        desired_speed = 2

            

        desired_force = [intend_force*(desired_speed*(goal_x)), intend_force*(desired_speed*(goal_y))]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        

        sense_R = self.vision_radius
        bodies = self.model.space.query_radius(self.xy, sense_R)
        neighbors = []

        for b in bodies:
            ref = b.ref
            if (ref is None) or (ref is self) or getattr(ref, "dead", False):
                continue
            neighbors.append(ref)
        # 현재 로봇은 군중한테 물리적 영향을 받지 않음 -> 아래 코드 주석처리
        # for nb in neighbors:
        #     dx = self.xy[0] - nb.xy[0]
        #     dy = self.xy[1] - nb.xy[1]
        #     d  = math.hypot(dx, dy)
        #     if d < 1e-9:
        #         # 완전 중첩 초기 해소
        #         jx, jy = ((1.0,-1.0) if random.random()<0.5 else (-1.0,1.0))
        #         F_c_x += jx * KN * 0.01
        #         F_c_y += jy * KN * 0.01
        #         continue

        #     ux, uy = dx/d, dy/d
        #     nb_r = getattr(nb, "radius", 0.5)
        #     r_sum = self.radius + nb_r

        #     # 원거리 지수 반발
        #     mag = K_AGENT * math.exp((r_sum - d) / max(LAMBDA_A, 1e-6))
        #     F_rep_x += mag * ux
        #     F_rep_y += mag * uy

        #     # 근거리 접촉/마찰
        #     if d < r_sum:
        #         penetration = (r_sum - d)
        #         Fn_k = KN * penetration

        #         v_jx, v_jy = getattr(nb, "vel", [0.0, 0.0])
        #         rel_vx, rel_vy = (v_ix - v_jx), (v_iy - v_jy)
        #         v_n = rel_vx*ux + rel_vy*uy

        #         if v_n < 0:
        #             restitution = 0.2
        #             drop = (1.0 - restitution) * v_n
        #             self.vel[0] -= drop * ux
        #             self.vel[1] -= drop * uy

        #         Fn_c = -CN * v_n
        #         Fn = max(Fn_k + Fn_c, 0.0)
        #         F_c_x += Fn * ux
        #         F_c_y += Fn * uy

        #         vt_x = rel_vx - v_n*ux
        #         vt_y = rel_vy - v_n*uy
        #         F_t_x += -MU_T * vt_x
        #         F_t_y += -MU_T * vt_y
        
        F_wx = F_wy = 0
        p = Point(self.xy[0], self.xy[1])
        self.collision_check = 0
        for poly in self.model._obstacle_polys:
            # 경계선까지 최소거리
            d = poly.exterior.distance(p)
            if d <= self.body_radius * 0.8:   # 매우 근접 → 충돌 경보
                self.collision_check = 1
            if d > 1.5 * self.body_radius:    # 멀면 무시
                continue
            q = poly.exterior.interpolate(poly.exterior.project(p))
            dx = self.xy[0] - q.x
            dy = self.xy[1] - q.y
            dist = math.hypot(dx, dy) or 1e-9
            nx, ny = dx/dist, dy/dist
            # 사람이랑 같은 톤으로 지수 반발(상수는 좀 더 세게 하고 싶으면 K_WALL 따로 둬도 됨)
            mag = K_AGENT * math.exp(-(d / max(LAMBDA_A, 1e-6)))
            F_wx += mag * nx
            F_wy += mag * ny


        F_x = 0
        F_y = 0
        F_x += desired_force[0]
        F_y += desired_force[1]
        

        F_x += F_wx
        F_y += F_wy
        vel = [0,0]
        vel[0] = F_x/self.mass
        vel[1] = F_y/self.mass
        future_xy = self.xy.copy()
        future_xy[0] += vel[0] * time_step
        future_xy[1] += vel[1] * time_step

        self.xy = future_xy
        self.model.space.clamp(self.xy)
        self.model.space.move(self.unique_id, self.xy)

        return tuple(self.xy)

