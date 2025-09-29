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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



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

    def choice_safe_mesh(self, point, max_radius: int = 20):
        """
        주어진 좌표 주변에서 pure_mesh 를 찾아 반환한다.
        1) 현재 셀 확인
        2) 반지름 1 → max_radius 로 확장하며 첫 pure_mesh 발견 즉시 반환
        3) 그래도 없으면 임의의 pure_mesh 반환 (이론상 도달 불가)
        """
        gx, gy = int(round(point[0])), int(round(point[1]))

        def valid_tri(x, y):
            tri = self.model.match_grid_to_mesh.get((x, y))
            return tri if tri in self.model.pure_mesh else None

        tri = valid_tri(gx, gy)
        if tri:
            return tri

        # ── 주변을 나선형(맨해튼 원) 으로 확장 탐색
        for r in range(1, max_radius + 1):
            # 위·아래 변
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    tri = valid_tri(gx + dx, gy + dy)
                    if tri:
                        return tri
            # 좌·우 변(모서리는 이미 검사)
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    tri = valid_tri(gx + dx, gy + dy)
                    if tri:
                        return tri

        # ── 이론상 도달 불가하지만, 혹시 모를 경우 대비
        return random.choice(self.model.pure_mesh)

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

                   
            if (self.model.robot_type == "Q"):
                new_position_robot = self.robot_policy_Q()
            
            self.model.grid.move_agent(self, new_position_robot)
            self.pos = new_position_robot
            return
        
        if(self.type == 0 or self.type == 1 or self.type == 2):
            self.pos = (int(round(self.xy[0])), int(round(self.xy[1])))
            new_position = self.agent_modeling()
            new_position = (int(round(new_position[0])), int(round(new_position[1])))
            self.model.grid.move_agent(self, new_position) ## 그 위치로 이동

    def choice_near_goal(self, pos):
        min_d = float('inf')
        near   = None
        for g in self.model.exit_point:
            d = self.mesh_to_mesh_distance(g, pos)
            if d < min_d:
                min_d  = d
                near   = g
        return near

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
                    repulsive_force[0] += 10*np.exp(-(d/2))*(d_x/d)
                    repulsive_force[1] += 10*np.exp(-(d/2))*(d_y/d)
                    self.blocked = True
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

        if (self.blocked == False):
            self.which_goal_agent_want()
        else :
            self.which_goal_agent_want(find_another = True)

        if(self.robot_initialized == 1):
            self.robot_initialized += 1
            self.now_goal = [self.xy[0], self.xy[1]]
        self.previous_type = self.type

                
        if(goal_d != 0):
          desired_force = [intend_force*(self.desired_speed_a*(goal_x/goal_d)), intend_force*(self.desired_speed_a*(goal_y/goal_d))] #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        else :
          desired_force = [0, 0]
        
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

        #if (self.model.valid_space[(int(round(future_xy[0])), int(round(future_xy[1])))]):
        if self.model.valid_space[int(future_xy[0]), int(future_xy[1])]:
            self.xy = future_xy.copy()
            self.blocked = False
        else :
            self.blocked = True

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

 
    def which_goal_agent_want(self, find_another = False):
        global robot_prev_xy
        robot_radius = 7
        agent_radius = 7
        exit_confirm_radius = 7
        
        to_follow_agents = [] ## 같은 mesh에 따라갈 agent가 있는지 확인하려는 list
        for agent in self.model.crowds : ## 같은 mesh에 있는 agent들 중에서 int(round(agent.xy[0]))
            if (agent.type == 0 or agent.type == 1): ## 로봇 following/ myway 인 agent만 확인
                distance = math.sqrt(pow(self.xy[0]-agent.xy[0],2)+pow(self.xy[1]-agent.xy[1],2))
                if distance < agent_radius and not agent.dead: ## agent 반경 내에 있으면
                    to_follow_agents.append(agent)

        now_mesh = self.choice_safe_mesh(self.xy) ## agent가 있는 mesh
        if(self.danger == 0) :
          self.danger = self.model.mesh_danger[now_mesh]
        shortest_distance = math.sqrt(pow(self.xy[0]-self.model.exit_point[0][0],2)+pow(self.xy[1]-self.model.exit_point[0][1],2)) ## agent와 가장 가까운 탈출구 사이의 거리
        shortest_goal = self.model.exit_point[0]

        exit_point_index = 0
        for index, i in enumerate(self.model.exit_point): ## agent가 가장 가까운 탈출구로 이동
            if  (math.sqrt(pow(self.xy[0]-i[0],2)+pow(self.xy[1]-i[1],2)) < shortest_distance):
                shortest_distance = math.sqrt(pow(self.xy[0]-i[0],2)+pow(self.xy[1]-i[1],2))
                exit_point_index = index
        if(self.is_confirmed == 1):
            self.is_confirmed_past = 1

        
        if (shortest_distance < exit_confirm_radius): ## agent가 탈출구에 도착했을 때
            self.now_goal = self.model.exit_point[exit_point_index]
            self.is_confirmed = 1
            #self.danger = 0
            return  
        
        robot_d = math.sqrt(pow(self.xy[0]-self.model.robot.xy[0],2)+pow(self.xy[1]-self.model.robot.xy[1],2))
        
        if self.not_tracking > 0:
            self.not_tracking -= 1
        if(robot_d < robot_radius and self.model.robot_mode == "GUIDE" and self.not_tracking == 0):
            self.robot_tracked = 7
            self.type = 0
            if(self.is_effected_by_robot == 0):
                self.model.new_founded_agent_danger += self.danger
            self.is_effected_by_robot = 1
            if self.previous_type != 0:
                if random.choices([0, 1], weights=[0.1, 0.9], k=1)[0] == 0:
                    self.type = 1
                    self.not_tracking = 7 #7step동안 로봇을 안따라가게 
            if (self.type == 0):
                goal_x = self.model.robot.xy[0]
                goal_y = self.model.robot.xy[1]
                self.type = 0
                self.now_goal = [goal_x, goal_y]
        else:
            self.type = 1
            if(len(to_follow_agents) > 0):
                if(self.previous_mesh != now_mesh):
                    if(random.choices([0, 1], weights=[0.9, 0.1], k=1)[0] == 0):
                        self.type = self.previous_type
                    elif random.choices([0, 1], weights=[0.6, 0.4], k=1)[0] == 0:
                        self.type = 2
                        if(self.previous_type !=2):
                            self.follow_agent_id = random.choice(to_follow_agents).unique_id 
                    else:
                        self.type = 1
            else :
                self.type = 1

        if(math.sqrt((pow(self.xy[0]-self.now_goal[0],2)+pow(self.xy[1]-self.now_goal[1],2))<2 and self.type==1) or self.agent_pos_initialized == 0 or find_another): #로봇에 의해 가이드되고 있을때는 골에 근접하더라도 골 초기화 x
            ## agent가 가고 있는 골에 도착했을 때, 처음 agent가 생성되었을 때 
            self.type = 1
            self.previous_mesh = now_mesh
            self.past_mesh = self.previous_mesh

            is_ongoing_direction = random.choices([0, 1], weights=[0.5, 0.5], k=1)[0] #80프로 확률로 가던 방향 선택하게 할 것 ## 50%로 바꿈
            
            if (is_ongoing_direction and self.agent_pos_initialized == 1):
                neighbors_coords = []
                for neighbor in self.model.adjacent_mesh[now_mesh]:
                    neighbor_coord = ((neighbor[0][0]+neighbor[1][0]+neighbor[2][0])/3, (neighbor[0][1]+neighbor[1][1]+neighbor[2][1])/3)
                    neighbors_coords.append(neighbor_coord)
                #print("neighbors_coords : ", neighbors_coords)
                #print("self.direction : ", self.direction)
                self.now_goal = find_closest_direction(self.xy, self.direction, neighbors_coords)
                #print(self.now_goal)
            else :
                while True:
                    mesh_index = random.randint(0, len(self.model.pure_mesh)-1)
                    random_mesh_choice = self.model.pure_mesh[mesh_index]

                    if random_mesh_choice in (now_mesh, self.past_mesh):
                        continue
                    next_mesh = self.model.next_vertex_matrix[now_mesh][random_mesh_choice]
                    if next_mesh is None:
                        continue
                    break

                next_mesh = self.model.next_vertex_matrix[now_mesh][self.model.pure_mesh[mesh_index]] ## agent가 가고 있는 골에서 다음으로 가야할 골


                while (random_mesh_choice == now_mesh or random_mesh_choice == self.past_mesh):
                    random_mesh_choice = self.model.pure_mesh[random.randint(0, len(self.model.pure_mesh)-1)]
                    #print("무한루프 걸림")
                # print("now_mesh : ")
                # print("next_mesh : ")
                    next_mesh = self.model.next_vertex_matrix[now_mesh][self.model.pure_mesh[mesh_index]] ## agent가 가고 있는 골에서 다음으로 가야할 골
                    if next_mesh is None:
                        continue
                self.now_goal =  [(next_mesh[0][0]+next_mesh[1][0]+next_mesh[2][0])/3, (next_mesh[0][1]+next_mesh[1][1]+next_mesh[2][1])/3] ## 다음으로 가야할 골의 중심
            self.agent_pos_initialized = 1
        
        if self.type == 2:
            self.now_goal =  self.model.return_agent_id(self.follow_agent_id).xy
        if (self.robot_tracked>0):
            self.robot_tracked -= 1


  
class RobotAgent(CrowdAgent):
    def __init__(self, unique_id, model, pos, type1):
        super().__init__(unique_id, model, pos, type1)
        self.action = [0, 0, "GUIDE"]
        self.past_xy = deque(maxlen=20)
        self.collision_check = 0
        self.detect_abnormal_order = 0
        self.is_game_finished = 0

        self.robot_waypoint = [0, 0]
        self.now_exploration = 0
        

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
        # print("robot_danger : ", self.danger)
            

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
    

