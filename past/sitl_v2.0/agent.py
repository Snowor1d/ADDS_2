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




def send_command(command):
    global s
    s.sendall((command +"\n").encode())




host = '172.20.10.7'
port = 80
weight_changing = [1, 1, 1, 1] # 각 w1, w2, w3, w4에 해당하는 weight를 변화시킬 것인가 

# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect((host, port))

num_remained_agent = 0
NUMBER_OF_CELLS = 70 


one_foot = 1
SumList = [0, 0, 0, 0, 0]
DifficultyList = [0, 0, 0, 0, 0]

ATTACK_DAMAGE = 50
INITIAL_HEALTH = 100
HEALING_POTION = 20
exit_w = 5
exit_h = 5
exit_area = [[0,exit_w], [0, exit_h]]
STRATEGY = 1
random_disperse = 1

theta_1 = random.randint(1,10)
theta_2 = random.randint(1,10)
theta_3 = random.randint(1,10)

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

def Multiple_linear_regresssion(distance_ratio, remained_ratio, now_affected_agents_ratio, v_min, v_max):
    global theta_1, theta_2, theta_3
    v = distance_ratio*theta_1 + remained_ratio*theta_2 + now_affected_agents_ratio*theta_3
    if (v>v_max):
        return v_max
    elif (v<v_min):
        return v_min
    else:
        return v




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


def set_agent_type_settings(agent, type):
    """Updates the agent's instance variables according to its type.

    Args:
        agent (FightingAgent): The agent instance.
        type (int): The type of the agent.
    """
    if type == 1:
        agent.health = 2 * INITIAL_HEALTH ## 200
        agent.attack_damage = 2 * ATTACK_DAMAGE ## 100
    if type == 2:
        agent.health = math.ceil(INITIAL_HEALTH / 2) ## 50
        agent.attack_damage = math.ceil(ATTACK_DAMAGE / 2) ## 25
    if type == 3:
        agent.health = math.ceil(INITIAL_HEALTH / 4) ## 25
        agent.attack_damage = ATTACK_DAMAGE * 4 ## 80
    if type == 10: ## 구분하려고 아무 숫자 함, exit_rec 채우는 agent type
        agent.health = 500 ## ''
        agent.attack_damage = 0 ## ''
    if type == 11: ## 마찬가지.. 이건 wall list 채우는 agent의 type
        agent.health = 500
        agent.attack_damage = 0

    
    
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
        self.goal_init = 0
        self.type = type
        self.robot_previous_action = "UP"
        self.health = INITIAL_HEALTH
        self.attack_damage = ATTACK_DAMAGE
        self.attacked = False
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

        
        file_path = 'weight.txt'
        file = open(file_path, 'r')
        lines = file.readlines()
        file.close()

        self.w1 = float(lines[0])
        self.w2 = float(lines[1])
        self.w3 = float(lines[2])
        self.w4 = float(lines[3])

        self.feature_weights_guide = [self.w1, self.w2]
        self.feature_weights_not_guide = [self.w3, self.w4]

        self.model.robot_mode = "NOT_GUIDE"

        # self.xy[0] = self.random.randrange(self.model.grid.width)
        # self.xy[1] = self.random.randrange(self.model.grid.height)
        
        set_agent_type_settings(self, type)

        self.judge_list = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] #앞에 있는 것이 우선순위. 0 : guide, 1 : agent following, 2 : my way
        self.agent_judge_probability = [random.gauss(60, 15)/100, random.gauss(50, 15)/100] #[로봇을 따라갈 확률, 다른 agent를 따라갈 확률]

        self.mesh_c = 0
        self.type_0_flag = 0
        self.type_1_flag = 0
        self.type_2_flag = 0

        self.previous_escaped_agents = 0
        self.escaped_agents = 0


    def __repr__(self) -> str:
        return f"{self.unique_id} -> {self.health}"

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


        # when attacked needs one turn until be able to attack
        if self.attacked:
            self.attacked = False
            return
        if(self.type != 3): #robot은 죽지 않는다
            #print("self.xy[0] : ", self.xy[0])
            if self.model.exit_grid[int(self.xy[0])][int(self.xy[1])]:
                self.dead = True
                return
            
        # if(self.type == 0 or self.type==1):
        #     print("agent 위치 : ", self.xy)
        #     print("robot과의 거리 : ", self.agent_to_agent_distance_real(self.xy, robot_xy))
        #     print("--------------------")

        self.move()

    def choice_safe_mesh(self, point):
        point_grid = (int(round(point[0])), int(round(point[1])))
        x = point_grid[0]
        y = point_grid[1]
        while_checking = 0

        candidates = [(x+1,y+1), (x+1, y), (x, y+1), (x-1, y-1), (x-1, y), (x, y-1)]
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


    def attackOrMove(self, cells_with_agents, possible_steps) -> None:
        """Decides if the user is going to attack or just move.
        Acts randomly.

        Args:
            cells_with_agents (list[FightingAgent]): The list of other agents nearby.
            possible_steps (list[Coordinates]): The list of available cell where to go.
        """
        should_attack = self.random.randint(0, 1) ## 50% 확률로 attack
        if should_attack:
            self.attack(cells_with_agents)
            return
        new_position = self.random.choice(possible_steps) ## 다음 step에 이동할 위치 설정
        self.model.grid.move_agent(self, new_position) ## 그 위치로 이동

    def attack(self, cells_with_agents) -> None:
        """Handles the attack of the agent.
        Gets the list of cells with the agents the agent can attack.

        Args:
            cells_with_agents (list[FightingAgent]): The list of other agents nearby.
        """
        agentToAttack = self.random.choice(cells_with_agents) ## agent끼리 마주쳤을 때 맞을 애는 랜덤으로 고름
        agentToAttack.attacked = True ## 맞은 애 attacked 됐다~ 
        if agentToAttack.health <= 0: ## health 가 0보다 작으면 dead
            agentToAttack.dead = True

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
            self.make_buffer()

                   
            if (self.model.robot_type == "Q"):
                new_position_robot = self.robot_policy_Q()
                
            elif (self.model.robot_type == "A"):
                new_position_robot = self.robot_policy_A()

            #new_position_robot = self.robot_policy_A()
            #print("self.model.robot_mode", self.model.robot_mode)
            #self.model.reward_distance_difficulty()


            self.model.grid.move_agent(self, new_position_robot)

            
            return
        if(self.type == 0 or self.type == 1 or self.type == 2):
            new_position = self.agent_modeling()
            new_position = (int(round(new_position[0])), int(round(new_position[1])))
            self.pos = (int(round(self.pos[0])), int(round(self.pos[1])))
            self.model.grid.move_agent(self, new_position) ## 그 위치로 이동

    def choice_near_goal(self, pos):
        shortest_distance = 9999999999
        near_goal = None
        for i in self.model.exit_point:
            if (self.mesh_to_mesh_distance(i, pos) < distance):
                near_goal = i
                distance = self.mesh_to_mesh_distance(i, pos)
                if (distnace < shortest_distance):
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
        for i in self.model.agents:
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
        temp_loc = [(x-1, y), (x+1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
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
        time_step = 0.2 #time step... 작게하면? 현실의 연속적인 시간과 비슷해져 현실적인 결과를 얻을 수 있음. 그러나 속도가 느려짐
                        # 크게하면? 속도가 빨라지나 비현실적.. (agent가 튕기는 등..)
        #time_step마다 desired_speed로 가고, desired speed의 단위는 1픽셀, 1픽셀은 0.5m
        #만약 time_step가 0.1이고, desired_speed가 2면.. 0.1초 x 2x0.5m = 한번에 최대 0.1m 이동 가능..
        # desired_speed = 2 # agent가 갈 수 있는 최대 속도, 나중에는 정규분포화 시킬 것
        repulsive_force = [0, 0]
        self.previous_danger = self.danger
        self.danger = 99999
        for i in self.model.exit_point:
            self.danger = min(self.danger, self.point_to_point_distance([self.xy[0], self.xy[1]], i))
        self.gain = self.previous_danger - self.danger
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
                        repulsive_force[0] += 1*np.exp(-(d/2))*(d_x/d) 
                        repulsive_force[1] += 1*np.exp(-(d/2))*(d_y/d)
                    repulsive_force[0] += 1*np.exp(-(d/2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
                    repulsive_force[1] += 1*np.exp(-(d/2))*(d_y/d) 

                elif(near_agent.type == 11 or near_agent.type == 9):## 검정벽 
                    repulsive_force[0] += 2*np.exp(-(d/2))*(d_x/d)
                    repulsive_force[1] += 2*np.exp(-(d/2))*(d_y/d)
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

        self.which_goal_agent_want()
        if(self.robot_initialized == 1):
            self.robot_initalized += 1
            self.now_goal = [self.xy[0], self.xy[1]]
        self.previous_type = self.type
        # for agent in self.model.agents:
            # if (agent.type == 0):
                # print(f"Type: {agent.type}, {agent.unique_id} ({self.model.return_agent_id(agent.unique_id).xy}) is following ROBOT ({self.model.robot.xy}), and now_goal: {agent.now_goal}")
                
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

        self.xy[0] += self.vel[0] * time_step
        self.xy[1] += self.vel[1] * time_step
        next_x = int(round(self.xy[0]))
        next_y = int(round(self.xy[1]))

        if(next_x<0):
            next_x = 0
        if(next_y<0):
            next_y = 0
        if(next_x>self.model.width-1):
            next_x = self.model.width-1
        if(next_y>self.model.height):
            next_y = self.model.height-1

        self.robot_guide = 0
        return (next_x, next_y)

 
    def which_goal_agent_want(self):
        global robot_prev_xy
        robot_radius = 7
        agent_radius = 7
        exit_confirm_radius = 12
        
        to_follow_agents = [] ## 같은 mesh에 따라갈 agent가 있는지 확인하려는 list
        for agent in self.model.agents: ## 같은 mesh에 있는 agent들 중에서 int(round(agent.xy[0]))
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


        
        if (shortest_distance < exit_confirm_radius): ## agent가 탈출구에 도착했을 때
            self.now_goal = self.model.exit_point[exit_point_index]
            #self.danger = 0
            return

        
        
        robot_d = math.sqrt(pow(self.xy[0]-self.model.robot.xy[0],2)+pow(self.xy[1]-self.model.robot.xy[1],2))
        
        if self.not_tracking > 0:
            self.not_tracking -= 1
        if(robot_d < robot_radius and self.model.robot_mode == "GUIDE" and self.not_tracking == 0):
            self.robot_tracked = 7
            self.type = 0
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

        if(math.sqrt((pow(self.xy[0]-self.now_goal[0],2)+pow(self.xy[1]-self.now_goal[1],2))<2 and self.type==1) or self.agent_pos_initialized == 0): #로봇에 의해 가이드되고 있을때는 골에 근접하더라도 골 초기화 x
            ## agent가 가고 있는 골에 도착했을 때, 처음 agent가 생성되었을 때 
            self.type = 1
            self.previous_mesh = now_mesh
            self.past_mesh = self.previous_mesh

            is_ongoing_direction = random.choices([0, 1], weights=[0.2, 0.8], k=1)[0] #80프로 확률로 가던 방향 선택하게 할것
            
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
                mesh_index = random.randint(0, len(self.model.pure_mesh)-1)
                random_mesh_choice = self.model.pure_mesh[mesh_index]

                while (random_mesh_choice == now_mesh or random_mesh_choice == self.past_mesh):
                    random_mesh_choice = self.model.pure_mesh[random.randint(0, len(self.model.pure_mesh)-1)]
                    #print("무한루프 걸림")
                next_mesh = self.model.next_vertex_matrix[now_mesh][self.model.pure_mesh[mesh_index]] ## agent가 가고 있는 골에서 다음으로 가야할 골
                self.now_goal =  [(next_mesh[0][0]+next_mesh[1][0]+next_mesh[2][0])/3, (next_mesh[0][1]+next_mesh[1][1]+next_mesh[2][1])/3] ## 다음으로 가야할 골의 중심
            self.agent_pos_initialized = 1
        
        if self.type == 2:
            self.now_goal =  self.model.return_agent_id(self.follow_agent_id).xy
        if (self.robot_tracked>0):
            self.robot_tracked -= 1


  
class RobotAgent(CrowdAgent):
    def __init__(self, unique_id, model, pos, type1):
        super().__init__(unique_id, model, pos, type1)
        self.buffer = ReplayBuffer(capacity=800)


    def robot_mode_switch(self):
        global robot_radius
        spaces = self.model.spaces_of_map
        max_danger = 0
        danger = 0
        dangerous_space = 0
        dangerous_space_coordinate = [0, 0]
        # 맵 별 가장 높은 danger 밀도 계산
        if self.model.map_num == 1:
            for index, space in enumerate(spaces) :
                for agent in self.model.agents :
                    if not isinstance(agent.xy[0], int): # [0, 0] , [4, 60] 같이 없는 agent인데 agent.xy 찍으면 나오는 애들 있음; int인 애들 걸러버림
                        if agent.xy[0] >= space[0][0] and agent.xy[0] <= space[1][0] and agent.xy[1] >= space[0][1] and agent.xy[1] <= space[1][1] :
                            if agent.type == 0 or agent.type == 1 or agent.type == 2:
                                danger += agent.danger
                danger = danger / ((space[1][0] - space[0][0]) * (space[1][1] - space[0][1]))
                if (index + 1) == 12 :
                    danger = danger * 2
                if danger > max_danger :
                    max_danger = danger
                    dangerous_space = index
            dangerous_space_coordinate = [(spaces[dangerous_space][0][0] + spaces[dangerous_space][1][0]) / 2, (spaces[dangerous_space][0][1] + spaces[dangerous_space][1][1]) / 2]
            if dangerous_space + 1 == 12 :
                dangerous_space_coordinate = [40, 30]
            # print("max_danger", max_danger)
            # print("dangerous_space", dangerous_space + 1)
            # print("dangerous_space_coordinate", dangerous_space_coordinate)
        
        elif self.model.map_num == 2 or self.model.map_num == 3:
            for index, space in enumerate(spaces) :
                for agent in self.model.agents :
                    if not isinstance(agent.xy[0], int): # [0, 0] , [4, 60] 같이 없는 agent인데 agent.xy 찍으면 나오는 애들 있음; int인 애들 걸러버림
                        if agent.xy[0] >= space[0][0] and agent.xy[0] <= space[1][0] and agent.xy[1] >= space[0][1] and agent.xy[1] <= space[1][1] :
                            if agent.type == 0 or agent.type == 1 or agent.type == 2:
                                danger += agent.danger
                danger = danger / ((space[1][0] - space[0][0]) * (space[1][1] - space[0][1]))
                if danger > max_danger :
                    max_danger = danger
                    dangerous_space = index
            dangerous_space_coordinate = [(spaces[dangerous_space][0][0] + spaces[dangerous_space][1][0]) / 2, (spaces[dangerous_space][0][1] + spaces[dangerous_space][1][1]) / 2]
            # print("max_danger", max_danger)
            # print("dangerous_space", dangerous_space + 1)
            # print("dangerous_space_coordinate", dangerous_space_coordinate)

        elif self.model.map_num == 4:
            for index, space in enumerate(spaces) :
                for agent in self.model.agents :
                    if not isinstance(agent.xy[0], int): # [0, 0] , [4, 60] 같이 없는 agent인데 agent.xy 찍으면 나오는 애들 있음; int인 애들 걸러버림
                        if agent.xy[0] >= space[0][0] and agent.xy[0] <= space[1][0] and agent.xy[1] >= space[0][1] and agent.xy[1] <= space[1][1] :
                            if agent.type == 0 or agent.type == 1 or agent.type == 2:
                                danger += agent.danger
                if index + 1 == 2:
                    area = 108
                elif index + 1 == 6:
                    area = 32
                elif index + 1 == 7:
                    area = 246
                elif index + 1 == 8:
                    area = 192
                elif index + 1 == 9:
                    area = 222  
                elif index + 1 == 10:
                    area = 144
                elif index + 1 == 11:
                    area = 178
                elif index + 1 == 14:
                    area = 312
                elif index + 1 == 15:
                    area = 32
                else :
                    area = (space[1][0] - space[0][0]) * (space[1][1] - space[0][1])
                danger = danger / area                 
                
                if danger > max_danger :
                    max_danger = danger
                    dangerous_space = index
            dangerous_space_coordinate = [(spaces[dangerous_space][0][0] + spaces[dangerous_space][1][0]) / 2, (spaces[dangerous_space][0][1] + spaces[dangerous_space][1][1]) / 2]
            if dangerous_space + 1 == 2:
                dangerous_space_coordinate = [26, 67]


        elif self.model.map_num == 5:
            for index, space in enumerate(spaces) :
                for agent in self.model.agents :
                    if not isinstance(agent.xy[0], int): # [0, 0] , [4, 60] 같이 없는 agent인데 agent.xy 찍으면 나오는 애들 있음; int인 애들 걸러버림
                        if agent.xy[0] >= space[0][0] and agent.xy[0] <= space[1][0] and agent.xy[1] >= space[0][1] and agent.xy[1] <= space[1][1] :
                            if agent.type == 0 or agent.type == 1 or agent.type == 2:
                                danger += agent.danger
                if index + 1 == 6:
                    area = 372
                elif index + 1 == 7:
                    area = 216
                else :
                    area = (space[1][0] - space[0][0]) * (space[1][1] - space[0][1])
                danger = danger / area                 
                
                if danger > max_danger :
                    max_danger = danger
                    dangerous_space = index
            dangerous_space_coordinate = [(spaces[dangerous_space][0][0] + spaces[dangerous_space][1][0]) / 2, (spaces[dangerous_space][0][1] + spaces[dangerous_space][1][1]) / 2]
        

        # 로봇 주변의 danger 밀도 계산. 로봇 반경 내 장애물 있으면 area에 그만큼 제외
        robot_xy = [self.xy[0], self.xy[1]]
        robot_group_danger = 0
        for agent in self.model.agents:
            if(agent.dead == False and (agent.type == 0 or agent.type == 1 or agent.type == 2)) : 
                robot_xy = [self.xy[0], self.xy[1]]
                if (pow(robot_xy[0]-agent.xy[0], 2) + pow(robot_xy[1]-agent.xy[1], 2)) <= pow(robot_radius, 2) : ## 로봇 반경 내에 agent가 있다면
                    robot_group_danger += agent.danger
        
        # 로봇 영역 내 장애물과 겹치는 그리드 개수 계산
        robot_x, robot_y = int(robot_xy[0]), int(robot_xy[1])
        obstacles_grid_points = self.model.obstacles_grid_points # 장애물 좌표를 집합으로 변환하여 탐색 속도 향상
        obstacles_set = set(tuple(coord) for coord in obstacles_grid_points)
        overlap_count = 0
        for x in range(robot_x - robot_radius, robot_x + robot_radius + 1):
            for y in range(robot_y - robot_radius, robot_y + robot_radius + 1):
                if math.sqrt((x - robot_x) ** 2 + (y - robot_y) ** 2) <= robot_radius:
                    if (x, y) in obstacles_set:
                        overlap_count += 1

        area = math.pi * pow(robot_radius, 2)
        area -= overlap_count
        robot_group_danger = robot_group_danger / area
        # print("overlap_count", overlap_count)
        # print("robot_area", area)
        # print("dangerous_space", dangerous_space+1)
        # print("robot_group_danger", robot_group_danger)
        # print("max_danger", max_danger)

        # # danger 밀도 비교하여 로봇 모드 변경 여부 판단
        # if self.model.robot_mode == "NOT_GUIDE":
        #     print("not guide mode")
        # else:
        #     print("guide mode")

        if self.model.map_num == 1 or self.model.map_num == 2 or self.model.map_num == 5:
            coeff_ng2g = 1
            coeff_g2ng = 2       
            agent_count = 0
            for agent in self.model.agents:
                if (agent.dead == False  and (agent.type == 0 or agent.type == 1 or agent.type == 2)) :
                    agent_count += 1
            # print("agent_count", agent_count)
            if agent_count <= 5:
                coeff_ng2g = 0.5

        if self.model.map_num == 3 or self.model.map_num == 4:
            coeff_ng2g = 0.8
            coeff_g2ng = 3       
            agent_count = 0
            for agent in self.model.agents:
                if (agent.dead == False  and (agent.type == 0 or agent.type == 1 or agent.type == 2)) :
                    agent_count += 1
            # print("agent_count", agent_count)
            if agent_count <= 5:
                coeff_ng2g = 0.5

        if self.model.robot_mode == "NOT_GUIDE": # not guide 상태일 때
            if robot_group_danger >= coeff_ng2g * max_danger:
                self.model.robot_mode = "GUIDE"
                self.drag = 1
                # print("not guide -> guide change. ")
                # print("robot_group_danger(", robot_group_danger, ") >= coeff_ng2g(", coeff_ng2g, ") * max_danger(", max_danger, ")")
        else: # guide 상태일 때
            if max_danger >= coeff_g2ng * robot_group_danger:  # guide 포기하고 not guide 하는 건 진짜 위험한 그룹이 있다고 판단될 때만.
                self.model.robot_mode = "NOT_GUIDE"
                self.drag = 0
                # print("guide -> not guide change. ")
                # print("max_danger(", max_danger, ") >= coeff_g2ng(", coeff_g2ng, ") * robot_group_danger(", robot_group_danger, ")")

    def robot_policy_Q(self):
        time_step = 0.2
        robot_radius = 7

        if(self.robot_initialized == 0 ):
            self.robot_initialized = 1
            return (self.model.robot.xy[0], self.model.robot.xy[1]) ## 오호라... 처음에 리스폰 되는 거 피하려고 
        
        #next_action = self.select_Q(robot_xy)
        next_action = self.select_Q(self.xy)


        # if (next_action[1] == "GUIDE"):
        #     reward = self.check_reward("GUIDE")
        # else :
        #     reward = self.check_reward("NOT_GUIDE")


        # if(self.is_learning_state == 1):
        #     self.update_weight(reward)
            

        goal_x = 0
        goal_y = 0
        
        if(next_action[0] == "UP"):
            goal_x = 0 
            goal_y = 2
        elif(next_action[0] == "LEFT"):
            goal_x = -2
            goal_y = 0
        elif(next_action[0] == "RIGHT"):
            goal_x = 2
            goal_y = 0
        elif(next_action[0] == "DOWN"):
            goal_x = 0
            goal_y = -2

        goal_d = math.sqrt(pow(goal_x, 2) + pow(goal_y, 2))
        intend_force = 2
        desired_speed = 3

        if(self.model.robot_mode == "NOT_GUIDE"): ## not guide 일 때
            desired_speed = 6
        
            
        if(goal_d != 0):
            desired_force = [intend_force*(desired_speed*(goal_x/goal_d)), intend_force*(desired_speed*(goal_y/goal_d))]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        else :
            desired_force = [0, 0]
    
        
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

        for near_agent in near_agents_list:
            n_x = near_agent.xy[0]
            n_y = near_agent.xy[1]
            d_x = robot_xy[0] - n_x
            d_y = robot_xy[1] - n_y
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
                    repulsive_force[0] += 13 *np.exp(-(d/2))*(d_x/d)
                    repulsive_force[1] += 13 *np.exp(-(d/2))*(d_y/d)

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
        self.xy[0] += vel[0] * time_step
        self.xy[1] += vel[1] * time_step


        next_x = int(round(self.xy[0]))
        next_y = int(round(self.xy[1]))

        if(next_x<0):
            next_x = 0
        if(next_y<0):
            next_y = 0
        if(next_x>self.model.width-1):
            next_x = self.model.width-1
        if(next_y>self.model.height):
            next_y = self.model.height-1
            
        robot_goal = [next_x, next_y]
        #print(robot_goal)
        return (next_x, next_y)


    def robot_policy_A(self):
        time_step = 0.2
        #from model import Model
        global random_disperse ## random_disperse 는 있는데.. 2는 뭐임? 어디에도 없음 ### 원래는 2가 아니었네
        global robot_status ## robot이 no guide 일 때 0, guide 일 때 1
        global robot_radius ## 7
        global robot_ringing ## 0 ,, 이거 뭐임?
        global robot_goal 
        global past_target
        #self.drag = 1
        #robot_status = 1
        global robot_prev_xy
        self.robot_previous_goal = robot_goal
        #print("self.model.robot_mode : ", self.model.robot_mode, "self.robot_goal_mesh : ", self.robot_goal_mesh)
        now_mesh = self.choice_safe_mesh(self.xy)
        if(self.robot_goal_mesh == None or (self.robot_goal_mesh == now_mesh and self.model.robot_mode == "GUIDE")): # 로봇이 누구한테 가야할지 agent 탐색
            self.model.robot_mode = "NOT_GUIDE"
            selected_agent = None
            biggest_danger = 0
            for agent in self.model.agents:
                if (agent.type == 1 or agent.type == 0 or agent.type == 2):
                    danger = agent.danger
                    if (danger > biggest_danger):
                        biggest_danger = danger
                        selected_agent = agent
            if(selected_agent == None):
                self.robot_goal_mesh = self.choice_safe_mesh(self.choice_near_exit())
                return [50, 50]
            # self.model.robot_mode == "NOT_GUIDE"
            self.robot_goal_mesh = self.choice_safe_mesh(selected_agent.xy)

        elif(self.model.robot_mode == "NOT_GUIDE" and self.robot_goal_mesh == now_mesh): # 로봇이 Guide mode로 바뀌어야 할때
            self.model.robot_mode = "GUIDE"
            self.robot_goal_mesh = self.choice_safe_mesh(self.choice_near_exit())


        # if (now_mesh in self.model.obstacle_mesh):
        #     print("장애물에 걸림 !!")
        next_mesh = self.model.next_vertex_matrix[now_mesh][self.robot_goal_mesh]
        #print("next_mesh : ", next_mesh)
        self.now_goal = [(next_mesh[0][0]+next_mesh[1][0]+next_mesh[2][0])/3, (next_mesh[0][1]+next_mesh[1][1]+next_mesh[2][1])/3]

        goal_x = self.now_goal[0] - self.xy[0]
        goal_y = self.now_goal[1] - self.xy[1]
        goal_d = math.sqrt(pow(goal_x,2) + pow(goal_y,2))
        intend_force = 2       
        
        desired_speed = 3

        if(self.model.robot_mode == "NOT_GUIDE"): ## not guide 일 때
            desired_speed = 6
            
        if(goal_d != 0):
            desired_force = [intend_force*(desired_speed*(goal_x/goal_d)), intend_force*(desired_speed*(goal_y/goal_d))]; #desired_force : 사람이 탈출구쪽으로 향하려는 힘
        else :
            desired_force = [0, 0]
    
        
        x=int(round(self.xy[0]))
        y=int(round(self.xy[1]))
 
        temp_loc = [(x-1, y), (x+1, y), (x, y+1), (x, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
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
        repulsive_force = [0, 0]
        obstacle_force = [0, 0]

        k=4

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

                elif(near_agent.type == 1): ## agents
                    repulsive_force[0] += 0/4*np.exp(-(d/2))*(d_x/d) #반발력.. 지수함수 -> 완전 밀착되기 직전에만 힘이 강하게 작용하는게 맞다고 생각해서
                    repulsive_force[1] += 0/4*np.exp(-(d/2))*(d_y/d) 

                elif(near_agent.type == 11):## 검정벽 
                    repulsive_force[0] += 20*np.exp(-(d/2))*(d_x/d)
                    repulsive_force[1] += 20*np.exp(-(d/2))*(d_y/d)

        F_x = 0
        F_y = 0
        
        F_x += desired_force[0]
        F_y += desired_force[1]
        

        F_x += repulsive_force[0]
        F_y += repulsive_force[1]
        vel = [0,0]
        vel[0] = F_x/self.mass
        vel[1] = F_y/self.mass

        self.xy[0] += vel[0] * time_step
        self.xy[1] += vel[1] * time_step
        self.model.robot.xy[0] = self.xy[0]
        self.model.robot.xy[1] = self.xy[1]

        next_x = int(round(self.xy[0]))
        next_y = int(round(self.xy[1]))

        if(next_x<0):
            next_x = 0
        if(next_y<0):
            next_y = 0
        if(next_x>self.model.width-1):
            next_x = self.model.width-1
        if(next_y>self.model.height):
            next_y = self.model.height-1

        robot_goal = [next_x, next_y]

        

        return (next_x, next_y)

    def update_weight(self, reward):
        
        if not (self.buffer.is_half()):
            return # Replay buffer가 반 차지 않았으면 update하지 않음

        global weight_changing
        alpha = 0.005
        discount_factor = 0.01
        
        gamma = 0.99
        discounted_reward = 0 # 감쇠된 reward
        next_robot_xy=  [0, 0]
        learning_sample = self.buffer.sample(32) # 32개의 랜덤 샘플로 학습시킬 것
        for index, i in enumerate(range(len(learning_sample))):
            robot_xy = learning_sample[i][0]
            next_robot_xy[0] = robot_xy[0]
            next_robot_xy[1] = robot_xy[1]
            robot_action = learning_sample[i][1]
            if robot_action[0] == 'UP':
                next_robot_xy[1] += 1
            elif robot_action[0] == 'DOWN':
                next_robot_xy[1] -= 1
            elif robot_action[0] == 'RIGHT':
                next_robot_xy[0] += 1
            elif robot_action[0] == 'LEFT':
                next_robot_xy[0] -= 1

            f1 = learning_sample[i][2]
            f2 = learning_sample[i][3]
            f3 = learning_sample[i][4]
            f4 = learning_sample[i][5]
            if(robot_action[1] == "GUIDE"):
                next_state_max_Q = self.calculate_Max_Q(next_robot_xy, "GUIDE")
                present_state_Q = self.calculate_Q(robot_xy, robot_action)
                if(weight_changing[0]):
                    self.w1 += alpha * (discounted_reward + discount_factor * next_state_max_Q - present_state_Q) * f1
                if(weight_changing[1]):
                    self.w2 += alpha * (discounted_reward + discount_factor * next_state_max_Q - present_state_Q) * f2
                self.feature_weights_guide[0] = self.w1
                self.feature_weights_guide[1] = self.w2 
       

            elif(robot_action[1] == "NOT_GUIDE"):
                next_state_max_Q = self.calculate_Max_Q(next_robot_xy, "NOT_GUIDE")
                present_state_Q = self.calculate_Q(robot_xy, robot_action)
                if(weight_changing[2]):
                    self.w3 +=  alpha * (discounted_reward + discount_factor * next_state_max_Q - present_state_Q) * f3 
                if(weight_changing[3]):
                    self.w4 +=  alpha * (discounted_reward + discount_factor * next_state_max_Q - present_state_Q) * f4
                self.feature_weights_not_guide[0] = self.w3
                self.feature_weights_not_guide[1] = self.w4

        #self.buffer = []
        with open('weight.txt','w') as file:
            file.write(f"{self.w1}\n")
            file.write(f"{self.w2}\n")
            file.write(f"{self.w3}\n")
            file.write(f"{self.w4}\n")

        return





    # def update_weight(self,reward):  
    #     global weight_changing
    #     global robot_xy
    #     #print("self.buffer : ", self.buffer)
    #     alpha = 0.01
    #     discount_factor = 0.01
    #     next_robot_xy = [0,0]

    #     # dicounted_reward = reward * gamma ^ (100 - 해당 스텝 수)
    #     gamma = 0.99
    #     discounted_reward = 0 # 감쇠된 reward 
    
    #     for index, i in enumerate(range(len(self.buffer))):
            
    #         # discounted_reward = reward * pow( gamma, 10 - (index+ 1) ) # 감쇠된 reward 계산
    #         discounted_reward = reward
    #         robot_xy = self.buffer[i][0]
    #         next_robot_xy[0] = robot_xy[0]
    #         next_robot_xy[1] = robot_xy[1]
    #         robot_action = self.buffer[i][1]
    #         #print("robot_xy : ", robot_xy)
    #         #print("robot_action : ", robot_action)
    #         if robot_action[0] == 'UP':
    #             next_robot_xy[1] += 1
    #         elif robot_action[0] == 'DOWN':
    #             next_robot_xy[1] -= 1
    #         elif robot_action[0] == 'RIGHT':
    #             next_robot_xy[0] += 1
    #         elif robot_action[0] == 'LEFT':
    #             next_robot_xy[0] -= 1
            
    #         f1 = self.buffer[i][2]
    #         f2 = self.buffer[i][3]
    #         f3 = self.buffer[i][4]
    #         f4 = self.buffer[i][5]
            

    #         if(robot_action[1] == "GUIDE"):
    #             next_state_max_Q = self.calculate_Max_Q(next_robot_xy, "GUIDE")
    #             present_state_Q = self.calculate_Q(robot_xy, robot_action)
    #             if(weight_changing[0]):
    #                 self.w1 += alpha * (discounted_reward + discount_factor * next_state_max_Q - present_state_Q) * f1
    #             if(weight_changing[1]):
    #                 self.w2 += alpha * (discounted_reward + discount_factor * next_state_max_Q - present_state_Q) * f2
    #             self.feature_weights_guide[0] = self.w1
    #             self.feature_weights_guide[1] = self.w2 
    #             # with open ('log_guide.txt', 'a') as f:
    #             #     f.write("GUIDE learning . . .\n")
    #             #     f.write(f"w1 ( {self.w1} ) += alpha ( {alpha} ) * (reward ( {discounted_reward} ) + discount_factor ( {discount_factor} ) * next_state_max_Q({ next_state_max_Q }) - present_state_Q ( {present_state_Q})) * f1( {f1})\n")
    #             #     f.write(f"w2 ( { self.w2 } ) += alpha ( { alpha }) * (reward ( { discounted_reward }) + discount_factor ( { discount_factor }) * next_state_max_Q( { next_state_max_Q }) - present_state_Q ({ present_state_Q})) * f2({ f2})\n")
    #             #     f.write("============================================================================\n")
    #             #     f.close()
       

    #         elif(robot_action[1] == "NOT_GUIDE"):
    #             next_state_max_Q = self.calculate_Max_Q(next_robot_xy, "NOT_GUIDE")
    #             present_state_Q = self.calculate_Q(robot_xy, robot_action)
    #             if(weight_changing[2]):
    #                 self.w3 +=  alpha * (discounted_reward + discount_factor * next_state_max_Q - present_state_Q) * f3 
    #             if(weight_changing[3]):
    #                 self.w4 +=  alpha * (discounted_reward + discount_factor * next_state_max_Q - present_state_Q) * f4
    #             self.feature_weights_not_guide[0] = self.w3
    #             self.feature_weights_not_guide[1] = self.w4
    #             # with open ('log_not_guide.txt', 'a') as f:
    #             #     f.write("NOT GUIDE learning . . .\n")
    #             #     f.write(f"w3 ( { self.w3 } ) += alpha ( { alpha }) * (reward ( { discounted_reward }) + discount_factor ( { discount_factor }) * next_state_max_Q( { next_state_max_Q }) - present_state_Q ({ present_state_Q})) * f3({ f3})\n")
    #             #     f.write(f"w4 ( { self.w4 } ) += alpha ( { alpha }) * (reward ( { discounted_reward }) + discount_factor ( { discount_factor }) * next_state_max_Q( { next_state_max_Q }) - present_state_Q ({ present_state_Q})) * f4({ f4})\n")
    #             #     f.write("============================================================================\n")
    #             #     f.close()                    
          
    #     #self.buffer = []
    #     with open('weight.txt','w') as file:
    #         file.write(f"{self.w1}\n")
    #         file.write(f"{self.w2}\n")
    #         file.write(f"{self.w3}\n")
    #         file.write(f"{self.w4}\n")

    #     return

    def make_buffer(self):
        robot_xy = self.model.robot.xy
        robot_action = self.now_action
        
        # f1 f2 f3 f4 값 저장
        f1 = self.F1_distance(robot_xy, robot_action[0], robot_action[1])
        f2 = self.F2_near_agents(robot_xy, robot_action[0], robot_action[1])
        f3_f4 = self.F3_F4_direction_agents_danger(robot_xy, robot_action[0])
        f3 = f3_f4[0]
        f4 = f3_f4[1]

        self.buffer.add((robot_xy, robot_action, f1, f2, f3, f4, self.model.check_reward_danger()))

        #self.buffer.append([robot_xy, robot_action, f1, f2, f3, f4])

    def F1_distance(self, state, action, mode):

        global one_foot

        min_distance = 1000
        next_robot_position = [0, 0]
        next_robot_position[0] = self.xy[0]
        next_robot_position[1] = self.xy[1]

        if (action=="UP"):
            next_robot_position[1] += one_foot
        elif (action=="DOWN"):
            next_robot_position[1] -= one_foot
        elif (action=="LEFT"):
            next_robot_position[0] -= one_foot
        elif (action=="RIGHT"):
            next_robot_position[0] += one_foot
        

        result = 999999
        for i in self.model.exit_point:
            result = min(result, self.point_to_point_distance(next_robot_position, i))

        #print(f"next_goal : {next_goal}, {action} 일때의 space : {floyd_distance[((now_space[0][0],now_space[0][1]), (now_space[1][0], now_space[1][1]))][exit] } - {math.sqrt(pow(now_space_x_center-next_goal[0],2)+pow(now_space_y_center-next_goal[1],2))} + {math.sqrt(pow(next_goal[0]-next_robot_position[0],2)+pow(next_goal[1]-next_robot_position[1],2))} = {result}")
        #result = math.sqrt(pow(next_robot_position[0]-next_goal[0],2) + pow(next_robot_position[1]-next_goal[1],2)
        return result * 0.01


    def F2_near_agents(self, state, action, mode):
        global one_foot
        robot_xyP = [0, 0]
        robot_xyP[0] = state[0] ## robot_xyP : action 이후 로봇의 위치
        robot_xyP[1] = state[1]

        if(action == "STOP"):
            return self.agents_in_robot_area(robot_xyP) * 0.2

        if action == "UP": ## action이 UP 이면 로봇의 y좌표에 one_foot을 더함
            robot_xyP[1] += one_foot
            NumberOfAgents = self.agents_in_robot_area(robot_xyP) ##action 이후 로봇 반경 내 agents 수 구함
        elif action == "DOWN":
            robot_xyP[1] -= one_foot
            NumberOfAgents = self.agents_in_robot_area(robot_xyP)
        elif action == "RIGHT":
            robot_xyP[0] += one_foot
            NumberOfAgents = self.agents_in_robot_area(robot_xyP)
        elif action == "LEFT":
            robot_xyP[0] -= one_foot
            NumberOfAgents = self.agents_in_robot_area(robot_xyP)
        
        return NumberOfAgents * 0.2


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

        for i in self.model.agents: ##SumOfDistaces 구하는 과정
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
    
    
    def select_Q(self, state) :
        global robot_step_num
        global robot_radius
        global one_foot
        global robot_status
        global NUMBER_OF_CELLS

        consistency_mul = 1.2

        action_list = ["UP", "DOWN", "LEFT", "RIGHT"]
        robot_step_num += 1
        a = 0.1
        b = 2
        alpha = 1/self.switch_criteria
        
        beta = self.switch_criteria

        del_object = []
        # for k in action_list:
        #     if (k == "UP"):
        #         if( (self.model.match_grid_to_mesh[((int(round(self.xy[0]))), int(round(self.xy[1])+1))] not in self.model.pure_mesh)or (int(round(self.xy[1])+1)>self.model.height)):
        #             del_object.append("UP")         
        #     elif (k == "DOWN"):
        #         if( self.model.match_grid_to_mesh[((int(round(self.xy[0]))), int(round(self.xy[1])-1))] not in self.model.pure_mesh or (int(round(self.xy[1])-1)<=0)):
        #             del_object.append("DOWN")
        #     elif (k == "LEFT"):
        #         if(( self.model.match_grid_to_mesh[((int(round(self.xy[0]-1))), int(round(self.xy[1])))] not in self.model.pure_mesh) or (int(round(self.xy[0])-1)<=0)):
        #             del_object.append("LEFT")
        #     elif (k == "RIGHT"):
        #         if( self.model.match_grid_to_mesh[((int(round(self.xy[0]+1))), int(round(self.xy[1])))] not in self.model.pure_mesh or (int(round(self.xy[0])+1)>self.model.width)):
        #             del_object.append("RIGHT")

        for k in action_list:
            if (k == "UP"):
                if not ( self.model.valid_space[(int(round(self.xy[0])), int(round(self.xy[1])+1))]):
                    del_object.append("UP")         
            elif (k == "DOWN"):
                if (int(round(self.xy[1])-1))<0 : 
                    del_object.append("DOWN")
                    continue
                if not ( self.model.valid_space[(int(round(self.xy[0])), int(round(self.xy[1])-1))]):
                    del_object.append("DOWN")
            elif (k == "LEFT"):
                if (int(round(self.xy[0])-1))<0 : 
                    del_object.append("LEFT")
                    continue
                if not ( self.model.valid_space[(int(round(self.xy[0])-1), int(round(self.xy[1])))]):
                    del_object.append("LEFT")
            elif (k == "RIGHT"):
                if not ( self.model.valid_space[(int(round(self.xy[0])+1), int(round(self.xy[1])))]):
                    del_object.append("RIGHT")


        del_object= list(set(del_object))
        for i in del_object:
            action_list.remove(i)

        Q_list_guide = []
        Q_list_not_guide = []
        #print("action_list : ", action_list)
        for i in range(len(action_list)):
            Q_list_guide.append(0)
            Q_list_not_guide.append(0)
        MAX_Q =-999999999

        ## 초기 selected 값 random 선택 ##
        values = ["UP", "DOWN", "LEFT", "RIGHT"]
        selected = random.choice(values)

        exploration_rate = 0
        for j in range(len(action_list)):
            f1 = self.F1_distance(state, action_list[j], "GUIDE") 
            f2 = self.F2_near_agents(state, action_list[j], "GUIDE")
            f3_f4 = self.F3_F4_direction_agents_danger(state, action_list[j])
            f3 = f3_f4[0]
            f4 = f3_f4[1]
            # guide 모드일때 weight는 feature_weights_guide
            Q_list_guide[j] = f1 * self.feature_weights_guide[0] + f2 *self.feature_weights_guide[1] 
            Q_list_not_guide[j] = f3 * self.feature_weights_not_guide[0] + f4 * self.feature_weights_not_guide[1]
        
            if (Q_list_guide[j]>MAX_Q):
                MAX_Q= Q_list_guide[j]
                selected = action_list[j]
                self.model.robot_mode = "GUIDE"

            if (Q_list_not_guide[j]>MAX_Q):
                MAX_Q= Q_list_not_guide[j]
                selected = action_list[j]
                self.model.robot_mode = "NOT_GUIDE"

        # print("Q_list_guide : ", Q_list_guide)
        # print("Q_list_not_guide : ", Q_list_not_guide)
        # print("self.now_action : ", self.now_action)
        if random.random() <= exploration_rate:
            # print("exploration!")
            selected = random.choice(action_list)
            if self.model.robot_mode == "GUIDE":
                self.model.robot_mode = "NOT_GUIDE"
            else:
                self.model.robot_mode = "GUIDE"
            # print("self.now_action_exploration : ", self.now_action)

        self.now_action = [selected, self.model.robot_mode]
        return self.now_action
        
        
    def how_urgent_another_space_is(self):
        global robot_xy 
        dict_urgent = {}

        for key, val in self.model.space_graph.items():
            if len(val) != 0 : #닫힌 공간 제외 val 0으로 초기화
                dict_urgent[key] = 0
            else :
                dict_urgent[key] = -1
        robot_space = self.model.grid_to_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))]
        
        for agent in self.model.agents:
            if (agent.type==0 or agent.type==1) and agent.dead==False:
                space = self.model.grid_to_space[int(round(agent.xy[0]))][int(round(agent.xy[1]))]
                if(space==robot_space):

                    continue
                dict_urgent[tuple(map(tuple, space))] += agent.danger
        return dict_urgent
    def how_urgent_robot_space_is(self):
        global robot_xy
        global robot_radius
        urgent = 0
        
            
        for agent in self.model.agents:
            if ((agent.type==0 or agent.type==1) and (math.sqrt((pow(agent.xy[0]-robot_xy[0],2)+pow(agent.xy[1]-robot_xy[1],2)))<robot_radius) and agent.dead==False):
                urgent += agent.danger
        return urgent


    def four_direction_compartment(self):
        from model import space_connected_linear
        #from model import Model 
        global robot_xy 
        global one_foot
        r_x = robot_xy[0]
        r_y = robot_xy[1]
        four_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        

        del_object = []
        for k in four_actions:
            if (k[0] == "UP"):
                if(self.model.valid_space[(int(round(r_x)), int(round(r_y+one_foot)))]):
                    del_object.append("UP")
                    
            elif (k[0] == "DOWN"):
                if(self.model.valid_space[(int(round(r_x)), int(round(r_y-one_foot)))]==0 or (r_y-one_foot)<0):
                    del_object.append("DOWN")

            elif (k[0] == "LEFT"):
                if(self.model.valid_space[(int(round(max(r_x-one_foot, 0))), int(round(r_y)))]==0 or (r_x-one_foot)<0):
                    del_object.append("LEFT")
            elif (k[0] == "RIGHT"):
                if(self.model.valid_space[(int(round(min(r_x+one_foot, self.model.width))), int(round(r_y)))]==0) :
                    del_object.append("RIGHT")
        
        del_object= list(set(del_object))
        for i in del_object:
            four_actions.remove([i])
            four_actions.remove([i])

        four_compartment = {}

        for j in four_actions:
            four_compartment[j] = []
        
        floyd_distance = self.model.floyd_distance 
        next_vertex_matrix = self.model.floyd_warshall()[0]

        now_s = self.model.grid_to_space[int(round(robot_xy[0]))][int(round(robot_xy[1]))]
        
        now_s = ((now_s[0][0], now_s[0][1]), (now_s[1][0], now_s[1][1]))
        now_s_x_center = (now_s[0][0] + now_s[1][0])/2
        now_s_y_center = (now_s[1][0] + now_s[1][1])/2 
        robot_position = [0, 0]
        robot_position[0] = robot_xy[0]
        robot_position[1] = robot_xy[1]
        only_space = []
        for sp in self.model.space_list:
            if (not sp in self.model.room_list and sp != [[0,0], [10, 10]] and sp != [[]]):
                only_space.append(sp)

        for i in only_space:
            key = ((i[0][0], i[0][1]), (i[1][0], i[1][1]))
            if(key==now_s):
                continue
            next_goal = space_connected_linear(now_s, next_vertex_matrix[now_s][key])
            original_distance = floyd_distance[now_s][key] - math.sqrt(pow(now_s_x_center-next_goal[0],2)+pow(now_s_y_center-next_goal[1],2)) + math.sqrt(pow(next_goal[0]-robot_position[0],2)+pow(next_goal[1]-robot_position[1],2))
            up_direction = 99999
            down_direction = 99999
            left_direction = 99999
            right_direction = 99999

            for m in four_actions:
                if (m=="UP"):
                    up_direction = floyd_distance[now_s][key] - math.sqrt(pow(now_s_x_center-next_goal[0],2)+pow(now_s_y_center-next_goal[1],2)) + math.sqrt(pow(next_goal[0]-robot_position[0],2)+pow(next_goal[1]-(robot_position[1]+one_foot),2))        
                elif (m=="DOWN"):
                    down_direction = floyd_distance[now_s][key] - math.sqrt(pow(now_s_x_center-next_goal[0],2)+pow(now_s_y_center-next_goal[1],2)) + math.sqrt(pow(next_goal[0]-robot_position[0],2)+pow(next_goal[1]-(robot_position[1]-one_foot),2))        
                elif (m=="LEFT"):
                    left_direction = floyd_distance[now_s][key] - math.sqrt(pow(now_s_x_center-next_goal[0],2)+pow(now_s_y_center-next_goal[1],2)) + math.sqrt(pow(next_goal[0]-(robot_position[0]-one_foot),2)+pow(next_goal[1]-robot_position[1],2))     
                elif (m=="RIGHT"):
                    right_direction = floyd_distance[now_s][key] - math.sqrt(pow(now_s_x_center-next_goal[0],2)+pow(now_s_y_center-next_goal[1],2)) + math.sqrt(pow(next_goal[0]-(robot_position[0]+one_foot),2)+pow(next_goal[1]-robot_position[1],2))     

            min = up_direction 
            min_direction = "UP"

            if(min>down_direction):
                min_direction = "DOWN"
                min = down_direction
            if(min>left_direction):
                min_direction = "LEFT"
                min = left_direction
            if(min>right_direction):
                min_direction = "RIGHT"
                min = right_direction  

            four_compartment[min_direction].append(i)
        return four_compartment
    
    
    def F3_F4_direction_agents_danger(self, state, action):
        result = [1, 1] 
        x = state[0]
        y = state[1]
        after_x = x 
        after_y = y

        if (action=="UP"):
            after_y = y+1
        elif (action=="DOWN"):
            after_y = y-1 
        elif (action=="LEFT"):
            after_x = x-1
        elif (action=="RIGHT"):
            after_x = x+1
 
        if (after_x<0):
            after_x = 0
        if (after_y<0):
            after_y = 0
        count = 0
        if(self.model.valid_space[(int(round(after_x)),int(round(after_y)))]==0):
            # print("valid하지 않음")
            after_x = x
            after_y = y
        
        for i in self.model.agents:
            if(i.dead == False and (i.type==0 or i.type==1 or i.type==2)):
                d = self.point_to_point_distance([x,y], [i.xy[0], i.xy[1]])
                after_d = self.point_to_point_distance([after_x, after_y], [i.xy[0], i.xy[1]])
                # print(f"after_x : {after_x}, after_y : {after_y}, x : {x}, y : {y}")
                # print(f"after_d : {after_d}, d : {d}")
                if (after_d < d):
                    result[0] += i.danger
                    count += 1
        result[1] = count
        result[0] = result[0] * 0.001
        result[1] = result[1] * 0.02
                 
        #print(f"{action}으로 가면, {count}명의 agent와 가까워짐, F3값 : {result}")
        return result
                    
                
    
        


    def calculate_Max_Q(self,state,status): # state 집어 넣으면 max_Q 내주는 함수
        one_foot = 1.5
        action_list = []
        if(status == "GUIDE"):
            action_list = [["UP", "GUIDE"], ["DOWN", "GUIDE"], ["LEFT", "GUIDE"], ["RIGHT", "GUIDE"]]
        else :
            action_list = [["UP", "NOT_GUIDE"], ["DOWN", "NOT_GUIDE"], ["LEFT", "NOT_GUIDE"], ["RIGHT", "NOT_GUIDE"]]
        
        r_x = self.xy[0]
        r_y = self.xy[1]
        
        del_object = []
        for k in action_list:
            if (k[0] == "UP"):
                if(self.model.valid_space[(int(round(r_x)), int(round(r_y+one_foot)))]):
                    del_object.append("UP")
                    
            elif (k[0] == "DOWN"):
                if (int(round(r_y-one_foot))<0):
                    del_object.append("DOWN")
                    continue
                if(self.model.valid_space[(int(round(r_x)), int(round(r_y-one_foot)))]==0 or (r_y-one_foot)<0):
                    del_object.append("DOWN")

            elif (k[0] == "LEFT"):
                if (int(round(r_x-one_foot))<0):
                    del_object.append("LEFT")
                    continue
                if(self.model.valid_space[(int(round(max(r_x-one_foot, 0))), int(round(r_y)))]==0 or (r_x-one_foot)<0):
                    del_object.append("LEFT")
            elif (k[0] == "RIGHT"):
                if(self.model.valid_space[(int(round(min(r_x+one_foot, self.model.width))), int(round(r_y)))]==0) :
                    del_object.append("RIGHT")
  
        del_object= list(set(del_object))
        if(status=="GUIDE"):
            for i in del_object:
                action_list.remove([i, "GUIDE"])
        else :
            for i in del_object:
                action_list.remove([i, "NOT_GUIDE"])

        Q_list = []
        for i in range(len(action_list)):
            Q_list.append(0)
        MAX_Q = -9999999

        for j in range(len(action_list)):
            
            if action_list[j][1] == "GUIDE": # guide 모드일때 weight는 feature_weights_guide
                f1 = self.F1_distance(state, action_list[j][0], action_list[j][1])
                f2 = self.F2_near_agents(state, action_list[j][0], action_list[j][1])                
                Q_list[j] = (f1 * self.feature_weights_guide[0] + f2 *self.feature_weights_guide[1])
            
            else :                           # not guide 모드일때 weight는 feature_weights_not_guide 
                f3_f4 = self.F3_F4_direction_agents_danger(state, action_list[j][0])
                f3 = f3_f4[0]
                f4 = f3_f4[1]
                Q_list[j] = f3 * self.feature_weights_not_guide[0] + f4 * self.feature_weights_not_guide[1]
            if (Q_list[j]>MAX_Q):
                MAX_Q= Q_list[j]
        return MAX_Q
    
    def calculate_Q(self, state, action):
        global robot_xy
        
        f1 = self.F1_distance(state, action[0], action[1])
        f2 = self.F2_near_agents(state, action[0], action[1])
        f3_f4 = self.F3_F4_direction_agents_danger(state, action[0])
        f3 = f3_f4[0]
        f4 = f3_f4[1]
        
        Q= 0
        if(action[1] == "GUIDE"):
            #Q = f1 * self.feature_weights_guide[0] + f2*self.feature_weights_guide[1] + f3 * self.feature_weights_guide[2]
            Q = f1 * self.feature_weights_guide[0] + f2 *self.feature_weights_guide[1]
        else:
            Q = f3 * self.feature_weights_not_guide[0] + f4*self.feature_weights_not_guide[1]

        return Q

class ReplayBuffer: #replay buffer class 
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen
        
    def is_half(self):
        if len(self.buffer) >= self.buffer.maxlen*2/3:
            return True

