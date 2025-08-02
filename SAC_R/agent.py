#this source code requires Mesa==2.2.1 
#^__^
from mesa import Agent
import socket
import time 
import math
import numpy as np
import random
import copy
import sys 
from collections import deque
from heapq import heappush, heappop

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Start_training import K1, K2, K3



def send_command(command):
    global s
    s.sendall((command +"\n").encode())




host = '172.20.10.7'
port = 80
weight_changing = [1, 1, 1, 1] # 각 w1, w2, w3, w4에 해당하는 weight를 변화시킬 것인가 

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect((host, port))

num_remained_agent = 0
NUMBER_OF_CELLS = 50


one_foot = 1
SumList = [0, 0, 0, 0, 0]

exit_w = 5
exit_h = 5
exit_area = [[0,exit_w], [0, exit_h]]

random_disperse = 1

check_initialize = 0
exit_area = [[0,exit_w], [0,exit_h]]
mode = "GUIDE"
robot_step_num = 0
robot_xy = [2, 2]
robot_radius = 7 #로봇 반경 -> 10미터 
robot_status = 0
robot_ringing = 0
robot_goal = [0, 0]
past_target = ((0,0), (0,0))
robot_prev_xy = [0,0]
AGENT_TIME_STEP = 0.2
ROBOT_TIME_STEP = 0.15

now_danger_sum = 0

def angle_between_vectors(v1, v2):
    # v1과 v2는 [x, y] 형식의 벡터입니다.
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # 코사인 값이 -1~1 사이에 있도록 클램핑
    cos_theta = max(min(dot_product / (max(magnitude_v1 * magnitude_v2, 0.01)), 1), -1)
    angle = math.acos(cos_theta)  # 라디안 각도
    return math.degrees(angle)  # 도(degree)로 변환

def find_closest_direction(xy, target_direction, directions):
    min_angle = float('inf')
    closest_direction = None
    
    for direction in directions:
        angle = angle_between_vectors(target_direction, [direction[0]-xy[0], direction[1]-xy[1]])
        if angle < min_angle:
            min_angle = angle
            closest_direction = direction
    
    return closest_direction

def calculate_degree(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    m1 = np.linalg.norm(vector1)
    m2 = np.linalg.norm(vector2)
    
    cos_theta = dot_product / (m1 * m2)
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)
    # print("계산된 각도 : ", angle_degrees)
    
    return angle_degrees




goal_list = [[(71, 52)], [(89, 52)]]

def central_of_goal(goals):
    real_goal = [0, 0]
    for i in goals:
        real_goal[0] += i[0]
        real_goal[1] += i[1]
    real_goal[0] /= len(goals)
    real_goal[1] /= len(goals) 
    return real_goal

def check_departure(pose, goals):
    for i in goals:
        if (i[0]>pose[0] and i[1]>pose[1]):
            return True
    return False

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


# def set_agent_type_settings(agent, type):
#     """Updates the agent's instance variables according to its type.

#     Args:
#         agent (FightingAgent): The agent instance.
#         type (int): The type of the agent.
#     """
#     if type == 1:
#         agent.health = 2 * INITIAL_HEALTH ## 200
#         agent.attack_damage = 2 * ATTACK_DAMAGE ## 100
#     if type == 2:
#         agent.health = math.ceil(INITIAL_HEALTH / 2) ## 50
#         agent.attack_damage = math.ceil(ATTACK_DAMAGE / 2) ## 25
#     if type == 3:
#         agent.health = math.ceil(INITIAL_HEALTH / 4) ## 25
#         agent.attack_damage = ATTACK_DAMAGE * 4 ## 80
#     if type == 10: ## 구분하려고 아무 숫자 함, exit_rec 채우는 agent type
#         agent.health = 500 ## ''
#         agent.attack_damage = 0 ## ''
#     if type == 11: ## 마찬가지.. 이건 wall list 채우는 agent의 type
#         agent.health = 500
#         agent.attack_damage = 0

    
    
class CrowdAgent(Agent):
    """An agent that fights."""

    def __init__(self, unique_id, model, pos, type): 
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.next_mesh = None
        self.past_mesh = None
        self.previous_mesh = None
        self.agent_pos_initialized = 0
        self.pos = pos
        self.not_tracking = 0
        self.behavior_probability = [random.gauss(0.9, 0.1), random.gauss(0.2, 0.1), random.gauss(0.1, 0.1)] #robot #동조 #myway
        self.is_learning_state = 1
        self.robot_step = 0
        self.gain = 0
        self.gain2 = 0
        self.goal_init = 0
        self.type = type
        self.robot_previous_action = "UP"

        self.dead = False
        self.robot_tracked = 0
        self.danger = 0
        self.previous_danger = 0
        self.robot_guide = 0
        self.drag = 0
        self.dead_count = 0
        self.buried = False
        self.which_goal = 0
        self.previous_stage = []
        self.now_goal = [0,0]
        self.now_pointing_mesh = None

        global robot_prev_xy
        self.robot_previous_goal = [0, 0]
        self.robot_initialized = 0
        self.is_traced = 0
        self.direction = [0, 0]
        
        self.switch_criteria = 0.5
        self.velocity_a = 2
        self.velocity_b = 5

        #self.robot_xy = [2,2]
        #self.robot_status = 0
        # print(isinstance(pos, tuple))
        self.xy = pos
        self.vel = [0, 0]
        self.acc = [0, 0]
        self.is_near_robot = 0
        # self.mass = 3
        self.mass = (3/70)*np.random.normal(66, 4.16) # agent의 mass, 평균 66kg, 표준 편차 4.16kg
        if self.type == 3: # robot mass는 3으로 고정
            self.mass = 3

        self.desired_speed_a = np.random.normal(1.5, 0.2) # agent의 desired_speed, 평균 1.5m/s, 표준 편차 0.2m/s
        self.previous_goal = [0,0]

        self.now_action = ["UP", "GUIDE"]

        #for robot
        self.robot_space = ((0,0), (5,45))
        self.mission_complete = 1
        self.going = 0
        self.guide = 0
        self.save_target = 0
        self.save_point = 0
        self.robot_now_path = []
        self.robot_goal_mesh = None
        self.robot_waypoint_index = 0

        self.delay = 0
        self.xy1 = [0,0]
        self.xy2 = [0,0]
        self.previous_type = None

        self.go_path_num= 0
        self.back_path_num = 0

        self.is_confirmed = 0
        self.is_confirmed_past = 0
        
        self.is_effected_by_robot = 0
        self.blocked = False

        self.decision_flag = random.randint(1,5) # self.decision_flag == 0 -> 결정 다시 내림
        self.decision_period = random.randint(15,35) #self.decision_period == 0 -> 결정 다시 내림, 군중 마다 얼마만큼의 시간동안 자신의 결정을 번복하지 않는가 모델링


        self.model.robot_mode = "GUIDE"

        # self.xy[0] = self.random.randrange(self.model.grid.width)
        # self.xy[1] = self.random.randrange(self.model.grid.height)
        

        self.judge_list = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] #앞에 있는 것이 우선순위. 0 : guide, 1 : agent following, 2 : my way
        self.agent_judge_probability = [random.gauss(60, 15)/100, random.gauss(50, 15)/100] #[로봇을 따라갈 확률, 다른 agent를 따라갈 확률]

        self.mesh_c = 0
        self.type_0_flag = 0
        self.type_1_flag = 0
        self.type_2_flag = 0

        self.previous_escaped_agents = 0
        self.escaped_agents = 0
        self.exit_belief = None       # {"idx": int, "score": float, "alpha": int}
        self.info_decay  = 0          # 받은 정보의 전파 단계 α
        self.vision_r    = 7          # 이웃 탐색 반경 (한 칸=0.5 m라면 ≈3.5 m)



    def step(self) -> None:
        global check_initialize
        # if(self.type==1 or self.type==0):
        #     print(self.unique_id, " : pass")
        #     if(self.xy[0] == robot_xy[0] and self.xy[1]==robot_xy[1]):
        #         print("문제 발생!!!!!")
        #         sys.exit()

        #print("model A: ", robot_xy)
        global exit_area
        global goal_list

        """Handles the step of the model dor each agent.
        Sets the flags of each agent during the simulation.
        """

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
            if self.model.exit_grid[int(self.xy[0])][int(self.xy[1])]:
                self.dead = True
                return


        self.move()

    def choice_safe_mesh(self, point):
        point_grid = (int(round(point[0])), int(round(point[1])))
        x = point_grid[0]
        y = point_grid[1]
        while_checking = 0

        candidates = [(x+1,y+1), (x+1, y), (x, y+1), (x-1, y-1), (x-1, y), (x, y-1), (x+1, y-1), (x-1, y+1)]
        while (point_grid not in self.model.match_grid_to_mesh.keys()) or (self.model.match_grid_to_mesh[point_grid] not in self.model.pure_mesh):
            while_checking += 1
            if(while_checking == 50):
                raise Exception("safe mesh를 찾지 못하였습니다.")
            point_grid = candidates[random.randint(0, len(candidates)-1)]
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

    
    def change_learning_state(self, learning):
        self.is_learning_state = learning


    def check_stage_agent(self): ## 이건 언제 쓰이나??? agent 움직일 때 현재 자기가 있는 위치 알 때
        x = self.xy[0]
        y = self.xy[1]
        now_stage = []
        for i in self.model.space_list:
            if (x>i[0][0] and x<i[1][0] and y>i[0][1] and y<i[1][1]):
                now_stage = i
                break
        if(len(now_stage) != 0):
            now_stage = ((now_stage[0][0], now_stage[0][1]), (now_stage[1][0], now_stage[1][1]))
        else:
            now_stage = ((0,0), (5, 45))
        return now_stage
    
    def move(self) -> None:
        global goal_list
        global num_remained_agent
        global robot_prev_xy
        """Handles the movement behavior.
        Here the agent decides   if it moves,
        drinks the heal potion,
        or attacks other agent."""

        cells_with_agents = []
        robot_xy = [self.model.robot.xy[0], self.model.robot.xy[1]]
        robot_prev_xy[0] = robot_xy[0]
        robot_prev_xy[1] = robot_xy[1]
        
        if (self.type == 3):
            self.robot_step += 1

                   
            if self.model.robot_type == "Q":
                new_position_robot = self.robot_policy_Q()
            
            elif self.model.robot_type == "A":
                 new_position_robot = self.robot_policy_A()

            else:
                raise ValueError(f"Unknown robot_type {self.model.robot_type}")
            

            self.model.grid.move_agent(self, new_position_robot)
            self.pos = new_position_robot
            return
        
        if self.type in (0, 1, 2):               # (로봇이 아니면)
            # (1) 목표 재계산 --------------------
            self.which_goal_agent_want()          # ← 새 버전 호출
            # (2) 힘 계산·충돌 예측·이동 ----------
            self.pos = (round(self.xy[0]), round(self.xy[1]))
            new_pos  = self.agent_modeling()      # ← 내부에서 predict_collision() 포함
            new_pos  = (int(round(new_pos[0])), int(round(new_pos[1])))
            self.model.grid.move_agent(self, new_pos)

    def choice_near_goal(self, pos):
        shortest_distance = 9999999999
        near_goal = None
        for i in self.model.exit_point:
            distance = self.mesh_to_mesh_distance(i, pos)
            if (self.mesh_to_mesh_distance(i, pos) < distance):
                near_goal = i
                distance = self.mesh_to_mesh_distance(i, pos)
                if (distance < shortest_distance):
                    shortest_distance = distance
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


    
    def change_value(self, velocity_a, velocity_b, switch):
        self.velocity_a = velocity_a
        self.velocity_b = velocity_b 
        self.switch_criteria = switch
    
    

    def agents_in_robot_area(self, robot_xyP):
        #from model import Model
        number_a = 0
        robot_radius = 7
        for i in self.model.crowds:
            if(i.dead == False and (i.type == 0 or i.type == 1 or i.type == 2)): ##  agent가 살아있을 때 / 끌려가는 agent 일 때
                if (pow(robot_xyP[0]-i.xy[0], 2) + pow(robot_xyP[1]-i.xy[1], 2)) < pow(robot_radius, 2) : ## 로봇 반경 내에 agent가 있다면
                    number_a += 1
        return number_a

    

        
    def agent_modeling(self):
        global robot_radius
        global robot_status
        global robot_step_num
        global random_disperse

        x = int(round(self.xy[0]))
        y = int(round(self.xy[1]))
        temp_loc = [(x-1, y), (x+1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1), (x-2,y), (x+2, y), (x, y+2), (x, y-2)]
        near_loc = []
        for i in temp_loc:
            if(i[0]>0 and i[1]>0 and i[0]<self.model.grid.width and i[1] < self.model.grid.height):
                near_loc.append(i)
        near_agents_list = []
        for i in near_loc:
            near_agents = self.model.grid.get_cell_list_contents([i])
            if len(near_agents):
                for near_agent in near_agents:
                    near_agents_list.append(near_agent) #kinetic 모델과 동일
        F_x = 0
        F_y = 0
        k = 3
        valid_distance = 3
        intend_force = 2
        time_step = AGENT_TIME_STEP #time step... 작게하면? 현실의 연속적인 시간과 비슷해져 현실적인 결과를 얻을 수 있음. 그러나 속도가 느려짐
                        # 크게하면? 속도가 빨라지나 비현실적.. (agent가 튕기는 등..)
        #time_step마다 desired_speed로 가고, desired speed의 단위는 1픽셀, 1픽셀은 0.5m
        #만약 time_step가 0.1이고, desired_speed가 2면.. 0.1초 x 2x0.5m = 한번에 최대 0.1m 이동 가능..
        # desired_speed = 2 # agent가 갈 수 있는 최대 속도, 나중에는 정규분포화 시킬 것
        repulsive_force = [0, 0]
        self.previous_danger = self.danger
        self.danger = 99999
        for i in self.model.exit_point:
            self.danger = min(self.danger, self.point_to_point_distance([self.xy[0], self.xy[1]], i))
        
        self.gain = (self.previous_danger - self.danger) ## ??? 왜
        if(self.danger<5):
            self.gain = 0
        for near_agent in near_agents_list:
            n_x = near_agent.xy[0]
            n_y = near_agent.xy[1]
            d_x = self.xy[0] - n_x
            d_y = self.xy[1] - n_y
            d = math.sqrt(pow(d_x, 2) + pow(d_y, 2))
            if(valid_distance<d):
                continue    

            F = k * (valid_distance-d)
            if(near_agent.dead == True):
                continue
                
            if(d!=0):
                if(near_agent.type == 12): ## 가상 벽
                    repulsive_force[0] += 0
                    repulsive_force[1] += 0

                elif(near_agent.type == 1 or near_agent.type==3 or near_agent.type==2 or near_agent.type==0): ## agents
                    if(near_agent.type==3):
                        repulsive_force[0] += 3*np.exp(-(d/2))*(d_x/d) 
                        repulsive_force[1] += 3*np.exp(-(d/2))*(d_y/d)
                    repulsive_force[0] += 3*np.exp(-(d/2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
                    repulsive_force[1] += 3*np.exp(-(d/2))*(d_y/d) 

                elif(near_agent.type == 11 or near_agent.type == 9):## 검정벽 
                    repulsive_force[0] += 15*np.exp(-(d/2))*(d_x/d)
                    repulsive_force[1] += 15*np.exp(-(d/2))*(d_y/d)
            else :
                if(random_disperse):
                    repulsive_force = [1, -1]
                    random_disperse = 0
                else:
                    repulsive_force = [-1, 1] # agent가 정확히 같은 위치에 있을시 따로 떨어트리기 위함 
                    random_disperse = 1
        
        goal_x = self.now_goal[0] - self.xy[0]
        goal_y = self.now_goal[1] - self.xy[1]
        goal_d = math.sqrt(pow(goal_x,2) + pow(goal_y,2))

        robot_x = self.model.robot.xy[0] - self.xy[0]
        robot_y = self.model.robot.xy[1] - self.xy[1]
        robot_d = math.sqrt(pow(robot_x,2)+pow(robot_y,2))

        if(robot_d<robot_radius):
            self.is_near_robot = 1
        else:
            self.is_near_robot = 0

        # if (self.blocked == False):
        #     self.which_goal_agent_want()
        # else :
        #     self.which_goal_agent_want(find_another = True)

        if(self.robot_initialized == 1):
            self.robot_initialized += 1
            self.now_goal = [self.xy[0], self.xy[1]]
        self.previous_type = self.type

                
        tau = 0.5                               # relaxation time [s]
        if goal_d != 0:
            dir_x, dir_y = goal_x/goal_d, goal_y/goal_d
        else:                                    # 목표 바로 위
            dir_x, dir_y = 0.0, 0.0

        v_des_x = self.desired_speed_a * dir_x
        v_des_y = self.desired_speed_a * dir_y

        desired_force = [(v_des_x - self.vel[0]) / tau,
                        (v_des_y - self.vel[1]) / tau]
        
        F_x += desired_force[0]
        F_y += desired_force[1]
        
        F_x += repulsive_force[0]
        F_y += repulsive_force[1]
        

        self.acc[0] = F_x/self.mass
        self.acc[1] = F_y/self.mass

        self.vel[0] = self.acc[0]
        self.vel[1] = self.acc[1]
        #self.xy = [self.xy[0], self.xy[1]]
        self.direction = [self.vel[0], self.vel[1]]

        future_xy = self.xy.copy()
        future_xy[0] += self.vel[0] * time_step
        future_xy[1] += self.vel[1] * time_step
        future_xy = self.predict_collision(future_xy)

        if self.model.valid_space[(int(round(future_xy[0])), int(round(future_xy[1])))]:
            self.xy = future_xy
            self.blocked = False
        else:
            self.blocked = True      # 벽이면 이동 보류

        if(self.xy[0] < 1):
            self.xy[0] = 1
        if(self.xy[1] < 1):
            self.xy[1] = 1
        if(self.xy[0] > self.model.width-1):
            self.xy[0] = self.model.width-1
        if(self.xy[1] > self.model.height-1):
            self.xy[1] = self.model.height-1

        next_x = int(round(self.xy[0]))
        next_y = int(round(self.xy[1]))

        self.robot_guide = 0
        return (next_x, next_y)

 

    def which_goal_agent_want(self, find_another: bool = False) -> None:
        """
        Modified Social-Force 기반 목표 결정:
            · self.exit_belief = {"idx": 출구 index, "score": S_ij, "alpha": hop}
            · self.now_goal    = [x, y]  (다음 time-step 까지 유효한 가상 목표)
        """

        # ────────── 파라미터 ──────────
        VISION_R, AGENT_R, ROBOT_R, EXIT_CONFIRM_R = 10, 7, 7, 7
        P_robot_following = 1 #로봇을 따라갈 확률
        P_neighbor_following = 0.7 #군중을 따라갈 확률

        # ─ 0단계: 직접 출구를 보면 α=0 정보 기록 ─
        for idx, center in enumerate(self.model.exit_point):
            #print(f"센터 : {center}")
            #print(f"거리 : {idx}", self.point_to_point_distance(self.xy, center))
            if self.point_to_point_distance(self.xy, center) < VISION_R:
                s = self.model.exit_score(self, idx, alpha=0)
                self.exit_belief = {"idx": idx, "score": s, "alpha": 0}
        # ─ 1단계: 이웃에게서 정보 수신 & 비교 ─
        neighbors = [ag for ag in self.model.crowds
                    if (ag is not self) and not ag.dead
                    and self.point_to_point_distance(self.xy, ag.xy) < AGENT_R]

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
        robot_d = self.point_to_point_distance(self.xy, self.model.robot.xy)
        if(robot_d >= ROBOT_R and self.type==0):
            self.decision_flag = 0
        if(self.decision_flag == 0 or robot_d < ROBOT_R): 
            #print(f"Agent{self.unique_id} 는 새로운 결정을 내리기로 했습니다.")
            if(robot_d < ROBOT_R and self.model.robot_mode == "GUIDE"):
                if random.random() < P_robot_following:
                    #print(f"Agent{self.unique_id} 는 로봇을 따라갑니다!")
                    self.type = 0
                    self.now_goal = self.model.robot_xy[:]
                    self.is_effected_by_robot = 1
                else:
                    self.type = 1
                    #print(f"Agent{self.unique_id} 가 로봇을 외면했습니다! - My Way")
                
            else :
                followable_neighbors = []
                for n in neighbors:
                    if (n.type != 2): #서로가 서로를 따라갈 수는 없음
                        followable_neighbors.append(n)
                if(len(followable_neighbors) == 0):
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
            self.now_goal = self.model.robot.xy 

        elif self.type==1:
            now_mesh = self.choice_safe_mesh(self.xy)

            if (now_mesh == self.now_pointing_mesh): #향햐던 mesh에 도달했을 때
                self.now_pointing_mesh = None

            if (self.now_pointing_mesh == None): # 향하던 mesh에 도달하면 -> None으로 설정 -> 다시 탐색하게 하기
                self.now_pointing_mesh = random.choice(self.model.pure_mesh)

            self.now_goal = self._explore_randomly(now_mesh)
            
        elif self.type==2:
            self.now_goal = self.model.return_agent_id(self.follow_agent_id).xy


        # robot_d = self.point_to_point_distance(self.xy, self.model.robot.xy)
        # if self.not_tracking > 0:
        #     self.not_tracking -= 1

        # if (robot_d < ROBOT_R and self.model.robot_mode == "GUIDE"
        #         and self.not_tracking == 0):
        #     # 4-A) 로봇 근처 → 80 % follow
        #     if random.random() < 0.8:        # 80 %
        #         self.type          = 0       # guide
        #         self.robot_tracked = 7
        #         self.now_goal      = self.model.robot.xy[:]
        #         self.is_effected_by_robot = 1
        #     else:                            # 20 %
        #         self.type = 1                # 마이-웨이
        #         self.not_tracking = 7

        # else:                                # 로봇 영향권 밖
        #     if neighbors and random.random() < 0.7:   # 70 %
        #         follow = min(neighbors,
        #                     key=lambda nb: self.point_to_point_distance(
        #                                     self.xy, nb.xy))
        #         self.type = 2               # agent-following
        #         self.now_goal = follow.xy[:]
        #         self.follow_agent_id = follow.unique_id
        #     else:                           # 30 % (또는 이웃 없음)
        #         self.type = 1               # 마이-웨이

        # # ─ 5단계: 목표에 거의 도착했거나 재탐색 필요 ─
        # if (find_another or
        #     self.point_to_point_distance(self.xy, self.now_goal) < 2):

        #     now_mesh = self.choice_safe_mesh(self.xy)
        #     neigh    = self.model.adjacent_mesh.get(now_mesh, [])
        #     if neigh:
        #         self.now_goal = find_closest_direction(
        #             self.xy, self.direction,
        #             [((m[0][0]+m[1][0]+m[2][0])/3,
        #             (m[0][1]+m[1][1]+m[2][1])/3) for m in neigh])

        #     # “도착” 이후 다시 랜덤 결정
        #     if (random.random() < 0.5 and self.agent_pos_initialized):
        #         # 진행중인 방향 유지(50 %)
        #         pass
        #     else:
        #         rnd_mesh = random.choice(self.model.pure_mesh)
        #         while rnd_mesh in (now_mesh, getattr(self, "past_mesh", None)):
        #             rnd_mesh = random.choice(self.model.pure_mesh)
        #         nxt_mesh  = self.model.next_vertex_matrix[now_mesh][rnd_mesh]
        #         self.now_goal = [ (nxt_mesh[0][0]+nxt_mesh[1][0]+nxt_mesh[2][0])/3,
        #                         (nxt_mesh[0][1]+nxt_mesh[1][1]+nxt_mesh[2][1])/3 ]
        #     self.agent_pos_initialized = 1

        # type==2 일 때 추종 대상의 실시간 위치로 업데이트
        if self.type == 2:
            self.now_goal = self.model.return_agent_id(
                                self.follow_agent_id).xy
        if self.robot_tracked > 0:
            self.robot_tracked -= 1


    def _explore_randomly(self, now_mesh):
        
        next_mesh = self.model.next_vertex_matrix[now_mesh][self.now_pointing_mesh]
        return [ (next_mesh[0][0]+next_mesh[1][0]+next_mesh[2][0])/3,
                (next_mesh[0][1]+next_mesh[1][1]+next_mesh[2][1])/3 ]
        
        # # 이웃 mesh 의 중심점 리스트
        # neigh = self.model.adjacent_mesh.get(now_mesh, [])
        # neigh_centers = [((m[0][0]+m[1][0]+m[2][0])/3,
        #                 (m[0][1]+m[1][1]+m[2][1])/3) for m in neigh]

        # # ─ ① 진행방향 유지
        # if self.agent_pos_initialized and self.direction != [0, 0] and \
        # random.random() < 0.9 and neigh_centers:
        #     return find_closest_direction(self.xy, self.direction, neigh_centers)

        # # ─ ② 완전 랜덤
        # rnd_mesh = random.choice(self.model.pure_mesh)
        # while rnd_mesh in (now_mesh, self.past_mesh):
        #     rnd_mesh = random.choice(self.model.pure_mesh)
        # return [ (rnd_mesh[0][0]+rnd_mesh[1][0]+rnd_mesh[2][0])/3,
                #(rnd_mesh[0][1]+rnd_mesh[1][1]+rnd_mesh[2][1])/3 ]

    def predict_collision(self, future_xy):
        for ag in self.model.crowds:
            if ag is self or ag.dead: continue
            # 원형 반지름 r_i ≈ 0.5 cell
            if math.hypot(future_xy[0]-ag.xy[0], future_xy[1]-ag.xy[1]) < 1.0:
                # 동일 진행 방향 축 상에서 안전 위치 반환
                dir_vec = np.array(future_xy) - np.array(self.xy)
                if np.linalg.norm(dir_vec)==0: return self.xy
                dir_vec = dir_vec/np.linalg.norm(dir_vec)
                safe_xy = ag.xy - dir_vec   # 한 칸 뒤로
                return safe_xy.tolist()
        return future_xy


  
class RobotAgent(CrowdAgent):
    SEEK, LEAD, WAIT = 0, 1, 2           # 상태 코드
    LEAD_RADIUS  = 3                     # agent를 ‘붙잡았다’고 간주하는 반경
    HOLD_RADIUS  = 4                  # lead 중 agent가 이 범위를 벗어나면 STOP
    EXIT_THRESH  = 2                     # agent-to-exit 거리가 이하면 WAIT

    def __init__(self, unique_id, model, pos, type1):
        super().__init__(unique_id, model, pos, type1)
        self.action = [0, 0, "GUIDE"]
        self.past_xy = deque(maxlen=20)
        self.collision_check = 0
        self.detect_abnormal_order = 0
        self.is_game_finished = 0

        self.robot_waypoint = [0, 0]
        self.now_exploration = 0

        self.astar_state   : int   = self.SEEK
        self.lead_target_id: int|None = None

        self._seek_path: list[tuple[int,int]] = []
        self._seek_idx = 0

        self.exit_path : list[tuple[int,int]] = []
        self._lead_idx = 0

    @staticmethod
    def _astar_grid(start, goal, valid, width, height):
        """4-connected A* on the discrete grid."""
        def heur(p):                                # ← 함수 이름을 h → heur 로 변경
            return ((p[0]-goal[0])**2 + (p[1]-goal[1])**2) ** 0.5

        open_q, came, g = [], {}, {start: 0}
        heappush(open_q, (heur(start), start))
        nbr = [(1,0),(-1,0),(0,1),(0,-1)]
        while open_q:
            _, cur = heappop(open_q)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]; path.append(cur)
                return path[::-1]

            for dx, dy in nbr:
                nx, ny = cur[0] + dx, cur[1] + dy
                if 0 <= nx < width and 0 <= ny < height and valid[(nx, ny)]:
                    ng = g[cur] + 1
                    if ng < g.get((nx, ny), 1e9):
                        g[(nx, ny)] = ng
                        came[(nx, ny)] = cur
                        heappush(open_q, (ng + heur((nx, ny)), (nx, ny)))
        return []  

    # ------------------------------------------------------------
    def _xy_int(self):
        return (int(round(self.xy[0])), int(round(self.xy[1])))

    def _nearest_exit_cell(self, pos):
        """가장 가까운 출구 셀(정수 좌표) 리턴"""
        tgt = min(self.model.exit_point,
                  key=lambda e: self.point_to_point_distance(pos, e))
        return (int(round(tgt[0])), int(round(tgt[1])))

    def _choose_farthest_agent(self):
        max_d, far = -1, None
        for ag in self.model.crowds:
            if not ag.dead:
                d = self.shortest_distance(self.xy, ag.xy)
                if d > max_d:
                    max_d, far = d, ag
        return far
    

    def shortest_distance(self, point1, point2):
        """point1과 point2 사이의 최단 거리"""
        return math.sqrt(pow(point1[0]-point2[0], 2) + pow(point1[1]-point2[1], 2))

    # ------------------------------------------------------------
    # 상태별 세부 로직
    # ------------------------------------------------------------
    def _seek(self):                                          # ← CHANGED (이전 로직 전면 교체)
        """Continuously track the (moving) target agent."""
        
        # 1) 타깃이 없으면 새로 선택
        if self.lead_target_id is None or \
           self.model.return_agent_id(self.lead_target_id) is None:
            tgt = self._choose_farthest_agent()
            if tgt is None:                                   # 모든 agent 탈출 완료
                return self._xy_int()
            self.lead_target_id = tgt.unique_id               # ← CHANGED

        tgt = self.model.return_agent_id(self.lead_target_id)

        # 2) ‘현재’ 위치 기준으로 1-step A* 경로 계산  ← NEW
        s = self._xy_int()
        g = (int(round(tgt.xy[0])), int(round(tgt.xy[1])))
        path = self._astar_grid(
            s, g, self.model.valid_space,
            self.model.width, self.model.height)

        if len(path) >= 2:                                    # path[0]==s, path[1]==next cell
            next_cell = path[1]
            self.xy = [float(next_cell[0]), float(next_cell[1])]
        # 경로가 없으면 제자리

        # 3) 반경 안에 들어오면 LEAD 상태로 전환     ← CHANGED
        if self.shortest_distance(self.xy, tgt.xy) <= self.LEAD_RADIUS:
            self.astar_state = self.LEAD
            self.exit_path, self._lead_idx = [], 0

        return self._xy_int()

    # ------------------------------------------------------------
    def _lead(self):
        #print("leading!")
        """agent를 데리고 출구로 이동"""
        tgt = self.model.return_agent_id(self.lead_target_id)
        if tgt is None or tgt.dead:               # agent가 사라졌으면 종료
            self._reset_to_seek()
            return self._xy_int()

        # 1) exit 경로를 한 번만 계산
        if not self.exit_path:
            s_cell = (int(round(tgt.xy[0])), int(round(tgt.xy[1])))
            e_cell = self._nearest_exit_cell(tgt.xy)
            self.exit_path = self._astar_grid(
                s_cell, e_cell,
                self.model.valid_space,
                self.model.width, self.model.height)
            self._lead_idx = 0
            print(f"[A*] exit path len={len(self.exit_path)}")

        # 2) agent가 멀리 떨어졌으면 대기
        dist = self.shortest_distance(self.xy, tgt.xy)
        if dist > self.HOLD_RADIUS:
            print(f"[A*] agent too far: {dist} > {self.HOLD_RADIUS}")
            return self._xy_int()       # STOP·대기

        # 3) 다음 way-point로 전진
        if self._lead_idx < len(self.exit_path):
            print(f"[A*] lead_idx={self._lead_idx}, exit_path_len={len(self.exit_path)}")
            wp = self.exit_path[self._lead_idx]
            if self._xy_int() == wp:
                self._lead_idx += 1
                if self._lead_idx < len(self.exit_path):
                    wp = self.exit_path[self._lead_idx]
            self.xy = [float(wp[0]), float(wp[1])]

        # 4) 도착 판정 → WAIT
        if self.shortest_distance(
                tgt.xy, self._nearest_exit_cell(tgt.xy)) < self.EXIT_THRESH:
            self.astar_state = self.WAIT
            print("[A*] WAIT at exit")

        return self._xy_int()

    # ------------------------------------------------------------
    def _wait(self):
        print("waiting!")
        """출구 앞에서 agent 탈출을 기다림"""
        tgt = self.model.return_agent_id(self.lead_target_id)
        if tgt is None or tgt.dead:
            print("[A*] agent escaped → SEEK")
            self._reset_to_seek()
        return self._xy_int()

    # ------------------------------------------------------------
    def _reset_to_seek(self):
        """SEEK 상태로 초기화"""
        self.astar_state   = self.SEEK
        self.lead_target_id= None
        self._seek_path, self.exit_path = [], []

    # ------------------------------------------------------------
    # 외부에서 호출되는 단일 정책 함수
    # ------------------------------------------------------------
    def robot_policy_A(self):
        if   self.astar_state == self.SEEK: return self._seek()
        elif self.astar_state == self.LEAD: return self._lead()
        else:                               return self._wait()
     

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

        if(math.sqrt(pow(self.xy[0]-self.robot_waypoint[0], 2)+pow(self.xy[1]-self.robot_waypoint[1], 2))<2):
            self.now_exploration = 0
            self.robot_waypoint = [0, 0]

        self.previous_danger = self.danger
        self.danger = 99999
        for i in self.model.exit_point:
            self.danger = min(self.danger, self.point_to_point_distance([self.xy[0], self.xy[1]], i))
        
        if(self.model.alived_agents()< 1):
            self.is_game_finished = 1

        time_step = ROBOT_TIME_STEP
        robot_radius = 7

        if(self.robot_initialized == 0 ):
            self.robot_initialized = 1
            return (self.model.robot.xy[0], self.model.robot.xy[1]) ## 오호라... 처음에 리스폰 되는 거 피하려고 
        self.past_xy.append(self.xy)

        goal_x = 0
        goal_y = 0
        
        goal_x += self.action[0]
        goal_y += self.action[1]
        
        #print(f"robot desired go to {goal_x}, {goal_y}") 
        self.model.robot_mode = "GUIDE"

        intend_force = 2
        desired_speed = 3

            

        desired_force = [intend_force*(desired_speed*(goal_x)), intend_force*(desired_speed*(goal_y))]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        
        x=int(round(self.xy[0]))
        y=int(round(self.xy[1]))
 
        temp_loc = [(x-1, y), (x+1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1), (x-2, y), (x+2, y), (x, y+2), (x, y-2)]
        near_loc = []
        for i in temp_loc:
            if(i[0]>=0 and i[1]>=0 and i[0]<self.model.grid.width and i[1] < self.model.grid.height):
                near_loc.append(i)
        near_agents_list = []
        for i in near_loc:
            near_agents = self.model.grid.get_cell_list_contents([i])
            if len(near_agents):
                for near_agent in near_agents:
                    near_agents_list.append(near_agent) #kinetic 모델과 동일
        repulsive_force = [0, 0]
        obstacle_force = [0, 0]

        k=4
        self.collision_check = 0
        for near_agent in near_agents_list:
            n_x = near_agent.xy[0]
            n_y = near_agent.xy[1]
            d_x = self.xy[0] - n_x
            d_y = self.xy[1] - n_y
            d = math.sqrt(pow(d_x, 2) + pow(d_y, 2))


            if(near_agent.dead == True):
                continue
                
            if(d!=0):
                if(near_agent.type == 12): ## 가상 벽
                    repulsive_force[0] += 0
                    repulsive_force[1] += 0
    
                elif(near_agent.type == 1 or near_agent.type ==0 or near_agent.type == 2): ## agents   
                    repulsive_force[0] += 0/4*np.exp(-(d/2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
                    repulsive_force[1] += 0/4*np.exp(-(d/2))*(d_y/d) 

                elif(near_agent.type == 11 or near_agent.type == 9):## 검정벽 
                    self.collision_check = 1
                    repulsive_force[0] += 8*np.exp(-(d/2))*(d_x/d)
                    repulsive_force[1] += 8*np.exp(-(d/2))*(d_y/d)
                    #print("repulsive_force : ", repulsive_force)

        F_x = 0
        F_y = 0
        # print("self.xy : ", self.xy)
        # print("desired_force : ", desired_force)
        # print("repulsive_force : ", repulsive_force)
        F_x += desired_force[0]
        F_y += desired_force[1]
        

        F_x += repulsive_force[0]
        F_y += repulsive_force[1]
        vel = [0,0]
        vel[0] = F_x/self.mass
        vel[1] = F_y/self.mass
        future_xy = self.xy.copy()
        future_xy[0] += vel[0] * time_step
        future_xy[1] += vel[1] * time_step

        if (self.model.valid_space[(int(round(future_xy[0])), int(round(future_xy[1])))]):
            self.xy = future_xy.copy()
            self.blocked = False 
        else :
            self.blocked = True

        if(self.xy[0]<1):
            self.xy[0] = 1
        if(self.xy[1]<1):
            self.xy[1] = 1
        if(self.xy[0]>self.model.width-2):
            self.xy[0] = self.model.width-2
        if(self.xy[1]>self.model.height-2):
            self.xy[1] = self.model.height-2
            

        next_x = int(round(self.xy[0]))
        next_y = int(round(self.xy[1]))

            
        robot_goal = [next_x, next_y]
        #print(robot_goal)
        return (next_x, next_y)


    def make_buffer(self):
        robot_xy = self.model.robot.xy
        robot_action = self.now_action
        
        image = self.model.return_current_image()

        self.buffer.add((robot_xy, robot_action, image, self.model.reward_based_gain()))





    def reward_distance(self, state, action, mode):
        from model import space_connected_linear
        global SumList
        SumOfDistances = 0 ##agent 하나로부터 출구까지의 거리의 합
        floyd_distance = self.model.floyd_distance

        evacuation_points = [] ## 출구 찾기~
        if(self.model.is_left_exit): 
            evacuation_points.append(((0,0), (5, 45)))
        if(self.model.is_up_exit):
            evacuation_points.append(((0,45), (45, 49)))
        if(self.model.is_right_exit):
            evacuation_points.append(((45,5), (49, 49)))
        if(self.model.is_down_exit):
            evacuation_points.append(((5,0), (49, 5)))

        for i in self.model.crowds: ##SumOfDistaces 구하는 과정
            if(i.dead == False and (i.type==0 or i.type==1)):
                agent_space = self.model.grid_to_space[int(round(i.xy[0]))][int(round(i.xy[1]))]
                
                next_goal = space_connected_linear(((agent_space[0][0],agent_space[0][1]), (agent_space[1][0], agent_space[1][1])), self.model.floyd_warshall()[0][((agent_space[0][0],agent_space[0][1]), (agent_space[1][0], agent_space[1][1]))][evacuation_points[0]])
                agent_space_x_center = (agent_space[0][0] + agent_space[1][0])/2
                agent_space_y_center = (agent_space[1][0] + agent_space[1][1])/2
                a = (floyd_distance[((agent_space[0][0],agent_space[0][1]), (agent_space[1][0], agent_space[1][1]))][evacuation_points[0]] 
                - math.sqrt(pow(agent_space_x_center-next_goal[0],2) + pow(agent_space_y_center-next_goal[1],2)) 
                + math.sqrt(pow(next_goal[0]-i.xy[0],2) + pow(next_goal[1]-i.xy[1],2)))
                
                ###준아야 너는 아래 코드를 수정해야 하며, 문제는 같은 space 내에서 agents가 움직이는 걸 반영하지 못하는 것에 있단다. 위 코드를 보며 수정하도록 야호^^
                # SumOfDistances += floyd_distance[(agent_space[0][0], agent_space[0][1]), (agent_space[1][0], agent_space[1][1])][evacuation_points[0]]
                SumOfDistances += a

        t = SumList[4]


        SumList[4] = SumList[3]
        SumList[3] = SumList[2]
        SumList[2] = SumList[1]
        SumList[1] = SumList[0]
        SumList[0] = SumOfDistances

        reward = (SumList[1]+SumList[2]+SumList[3]+SumList[4])/4 - SumOfDistances

        return reward
