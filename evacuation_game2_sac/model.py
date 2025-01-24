#this source code requires Mesa==2.2.1 
#^__^
from mesa import Model
from agent import RobotAgent
from agent import CrowdAgent
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import triangulate
import matplotlib.tri as mtri

import agent
from agent import WallAgent
import random
import copy
import math
import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial import Delaunay, ConvexHull
from sklearn.cluster import DBSCAN
from matplotlib.path import Path
import triangle as tr
import os
from collections import deque
#import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ADDS_AS_reinforcement import SACAgent, ReplayBuffer, PolicyNetwork, QNetwork 

##########################################################################
# 1) Replay Buffer
##########################################################################
class ReplayBuffer:
    def __init__(self, capacity=int(1e4)):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        state: (H, W) or (C, H, W) as np array
        action: np.array of shape (4,) 
                e.g. [dx, dy, mode_onehot0, mode_onehot1]
        reward: float
        next_state: np.array
        done: float(0 or 1)
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, int(batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(states).unsqueeze(1)  # (B,1,H,W) if grayscale
        actions     = torch.FloatTensor(actions)               # (B,4)
        rewards     = torch.FloatTensor(rewards)              # (B,)
        next_states = torch.FloatTensor(next_states).unsqueeze(1)
        dones       = torch.FloatTensor(dones)                # (B,)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def are_meshes_adjacent(mesh1, mesh2):
    # 두 mesh의 공통 꼭짓점의 개수를 센다
    common_vertices = set(mesh1) & set(mesh2)
    return len(common_vertices) >= 2  # 공통 꼭짓점이 두 개 이상일 때 인접하다고 판단

# goal_list = [[0,50], [49, 50]]
hazard_id = 5000
total_crowd = 10
max_specification = [20, 20]

number_of_cases = 0 # 난이도 함수 ; 경우의 수
started = 1

def get_points_within_polygon(vertices, grid_size=1):
    polygon_path = Path(vertices)
    
    # 다각형의 bounding box 설정
    min_x = int(np.min([v[0] for v in vertices]))
    max_x = int(np.max([v[0] for v in vertices]))
    min_y = int(np.min([v[1] for v in vertices]))
    max_y = int(np.max([v[1] for v in vertices]))
    
    # 그리드 점 생성
    x_grid = np.arange(min_x, max_x + grid_size, grid_size)
    y_grid = np.arange(min_y, max_y + grid_size, grid_size)
    grid_points = np.array(np.meshgrid(x_grid, y_grid)).T.reshape(-1, 2)
    
    # 다각형 내부 점 필터링
    inside_points = grid_points[polygon_path.contains_points(grid_points)]
    
    return inside_points.tolist()

def bresenham(x0, y0, x1, y1):
    """
    Bresenham's Line Algorithm to find all grid points that a line passes through.
    
    Args:
    x0, y0: Starting point of the line.
    x1, y1: Ending point of the line.
    
    Returns:
    A list of grid coordinates that the line passes through.
    """
    points = []
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    
    err = dx - dy
    
    while True:
        points.append([x0, y0])
        
        if x0 == x1 and y0 == y1:
            break
        
        e2 = 2 * err
        
        if e2 > -dy:
            err -= dy
            x0 += sx
        
        if e2 < dx:
            err += dx
            y0 += sy
    
    return points

def find_triangle_lines(v0, v1, v2):
    """
    Finds all grid coordinates that the triangle's edges pass through.
    
    Args:
    v0, v1, v2: The three vertices of the triangle, each as [x, y].
    
    Returns:
    A list of unique grid coordinates that the triangle's edges pass through.
    """
    line_points = set()  # Using a set to avoid duplicates
    
    # Get the points for each edge of the triangle
    line_points.update(tuple(pt) for pt in bresenham(v0[0], v0[1], v1[0], v1[1]))
    line_points.update(tuple(pt) for pt in bresenham(v1[0], v1[1], v2[0], v2[1]))
    line_points.update(tuple(pt) for pt in bresenham(v2[0], v2[1], v0[0], v0[1]))
    
    return list(line_points)

# # Example usage
# v0 = [10, 10]
# v1 = [20, 15]
# v2 = [15, 25]

# # Find grid coordinates for the triangle's edges
# line_coords = find_triangle_lines(v0, v1, v2)
# print("Grid coordinates that the triangle's edges pass through:", line_coords)

def is_point_in_triangle(p, v0, v1, v2):
    """
    Determines if a point p is inside the triangle formed by v0, v1, v2 using barycentric coordinates.
    
    Args:
    p: The point to check, as [x, y].
    v0, v1, v2: The triangle's vertices, each as [x, y].
    
    Returns:
    True if the point is inside the triangle, False otherwise.
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)

def calculate_internal_coordinates_in_triangle(width, height, v0, v1, v2, D):
    """
    Finds grid points inside the triangle formed by v0, v1, v2. 
    A point is included if more than half of the grid square overlaps with the triangle.
    
    Args:
    grid: The grid of points, a 2D array where each point is a coordinate [x, y].
    v0, v1, v2: The triangle's vertices, each as [x, y].
    D: The distance between grid points (grid resolution).
    
    Returns:
    A list of grid points inside the triangle.
    """
    grid_points_in_triangle = []
    
    # Loop through all grid points
    for x in range(width):
        for y in range(height):
            grid_point = [x, y]
            
            # Check if the center of the grid point is inside the triangle
            if is_point_in_triangle(grid_point, v0, v1, v2):
                grid_points_in_triangle.append(grid_point)
            # else:
            #     # If the center is not inside, check the neighboring points (for partial inclusion)
            #     # Check the four corner points of the grid square
            #     corners = [
            #         [x - D/2, y - D/2],
            #         [x + D/2, y - D/2],
            #         [x - D/2, y + D/2],
            #         [x + D/2, y + D/2]
            #     ]
                
            #     inside_corners = sum(is_point_in_triangle(corner, v0, v1, v2) for corner in corners)
                
            #     # Include grid point if more than half of its corners are inside the triangle
            #     if inside_corners >= 2:
            #         grid_points_in_triangle.append(grid_point)

    return grid_points_in_triangle

def add_intermediate_points(p1, p2, D):
    dist = np.linalg.norm(np.array(p2) - np.array(p1))
    if dist > D:
        num_points = int(dist // D) + 1
        return np.linspace(p1, p2, num=num_points+1, endpoint = False)[1:].tolist()
    return []

def generate_segments_with_points(vertices, segments, D):
    new_vertices = vertices.copy()
    new_segments = []
    for seg in segments:
        p1 = vertices[seg[0]]
        p2 = vertices[seg[1]]
        new_points = add_intermediate_points(p1, p2, D)
        last_index = seg[0]
        for point in new_points:
            new_vertices.append(point)
            new_index = len(new_vertices) - 1
            new_segments.append([last_index, new_index])
            last_index = new_index
        new_segments.append([last_index, seg[1]])
    return new_vertices, new_segments

 
class FightingModel(Model):
    """A model with some number of agents."""

    def __init__(self, number_agents: int, width: int, height: int, model_num = -1, robot = 'Q'):
        
        self.step_n = 0
        self.checking_reward = 0
        if (model_num == -1):
            model_num = random.randint(1,5)

        self.robot_type = robot
        self.spaces_of_map = []
        self.obstacles_grid_points = []
        self.map_num = model_num # 1 : 산학협력관 + 잔디밭 / 2 : 제2 공학관 + 정원 / 3 : 공학실습동 + 제2 연구동 / 4 : 벤젠고리관 / 5 : 경영관 + 퇴계 인문관

        self.running = (
            True  # required by the MESA Model Class to start and stop the simulation
        )
        self.agent_id = 1000
        self.agent_num = 0
        self.datacollector_currents = DataCollector(
            {
                "Remained Agents": FightingModel.current_healthy_agents,
                "Non Healthy Agents": FightingModel.current_non_healthy_agents,
            }
        )

        self.using_model = False
        self.total_agents = number_agents
        self.width = width
        self.height = height      
        self.obstacle_mesh = []
        self.adjacent_mesh = {}
        # map_ran_num = 2
        self.walls = list()
        self.obstacles = list()
        self.mesh = list()
        self.mesh_list = list()
        self.extract_map(self.map_num)     
        self.distance = {}  
        self.schedule = RandomActivation(self)
        self.schedule_e = RandomActivation(self)
        self.running = (
            True
        )
        self.next_vertex_matrix = {}
        self.exit_grid = np.zeros((self.width, self.height))
        self.pure_mesh = []
        self.mesh_complexity = {}
        self.mesh_danger = {}
        self.match_grid_to_mesh = {}
        self.match_mesh_to_grid = {}
        self.valid_space = {}
        self.grid = MultiGrid(width, height, False)
        self.headingding = ContinuousSpace(width, height, False, 0, 0)
        self.fill_outwalls(width, height)
        self.mesh_map()
        self.make_random_exit()
        self.construct_map()
        self.calculate_mesh_danger()
        self.exit_list = []
        #self.random_agent_distribute_outdoor(number_agents, 1)
        self.make_robot()
        #self.visualize_danger()
        self.robot_xy = [0, 0]
        self.robot_mode = "GUIDE"
        self.step_count = 0

        # for i in range(50):
        #     for j in range(50):
        #         print("(", i, j, ")", self.valid_space[(i, j)])
    def alived_agents(self):
        alived_agents = self.total_agents
        for i in self.schedule.agents:
            if((i.type==0 or i.type==1 or i.type==2) and i.dead == 1):
                alived_agents -= 1
        return alived_agents 

    def evacuated_agents(self):
        evacuated_agents = 0
        for i in self.schedule.agents:
            if((i.type==0 or i.type==1 or i.type==2) and i.dead == 1):
                evacuated_agents += 1
        return evacuated_agents

    
    def write_log(self):
        
        evacuated_agent_num = 0
        for i in self.schedule.agents:
            if((i.type==0 or i.type==1 or i.type==2) and i.dead == 1):
                evacuated_agent_num += 1

        with open("experiment.txt", "a") as f:
            f.write(f"{self.step_count} {evacuated_agent_num}\n")
        with open("experiment2.txt", "a") as f2:
            f2.write(f"{evacuated_agent_num}\n")



    def fill_outwalls(self, w, h):
        for i in range(w):
            self.walls.append((i, 0))
            self.walls.append((i, h-1))
        for j in range(h):
            self.walls.append((0, j))
            self.walls.append((w-1, j))
    def choice_safe_mesh_visualize(self, point):
        point_grid = (int(point[0]), int(point[1]))
        x = point_grid[0]
        y = point_grid[1]
        candidates = [(x+1,y+1), (x+1, y), (x, y+1), (x-1, y-1), (x-1, y), (x, y-1)]
        for c in candidates:
            if (self.match_grid_to_mesh[c] in self.pure_mesh):
                return c

        return False

        return self.match_grid_to_mesh[point_grid]
    def visualize_danger(self):
        for mesh in self.mesh:
            for i in range(len(mesh)):
                a = WallAgent(self.agent_num, self, [mesh[i][0], mesh[i][1]], 99)
                
                corresponding_mesh = self.match_grid_to_mesh[(mesh[i][0], mesh[i][1])]
                
                if (corresponding_mesh not in self.pure_mesh):
                    check = self.choice_safe_mesh_visualize([mesh[i][0], mesh[i][1]])
                    if (check == False):
                        continue
                    corresponding_mesh = self.match_grid_to_mesh[check]

                a.danger = self.mesh_danger[corresponding_mesh]
                self.agent_num+=1
                #self.schedule_e.add(a)
                self.grid.place_agent(a, [mesh[i][0], mesh[i][1]])
    
    def calculate_mesh_danger(self):
        for mesh in self.pure_mesh:
            shortest_distance = 9999999999
            near_mesh = None
            for e in self.exit_point:
                distance = math.sqrt(pow(mesh[0][0]-e[0], 2) + pow(mesh[0][1]-e[1], 2))
                if distance < shortest_distance:
                    shortest_distance = distance
                    near_mesh = e 
            self.mesh_danger[mesh] = shortest_distance
        return 0

    def mesh_map(self):

        D = 20
        map_boundary = [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]]
        obstacle_hulls = []

        for obstacle in self.obstacles:
            if len(obstacle) == 3 or len(obstacle) == 4:
                hull = ConvexHull(obstacle)
                hull_points = np.array(obstacle)[hull.vertices]
                obstacle_hulls.append(hull_points)
            else:
                raise ValueError("Each obstacle must have either 3 or 4 points.")

        # 경계점 및 장애물의 모서리 점 추가
        vertices = map_boundary.copy()
        for hull_points in obstacle_hulls:
            vertices.extend(hull_points.tolist())
        segments = [[i, (i + 1) % 4] for i in range(4)]  # 맵의 경계
        offset = 4  # 맵 경계 포인트를 위한 오프셋

        # 장애물의 모서리 추가
        for hull_points in obstacle_hulls:
            n = len(hull_points)
            segments.extend([[i + offset, (i + 1) % n + offset] for i in range(n)])
            offset += n

        # 세그먼트 및 포인트로 메쉬화
        vertices_with_points, segments_with_points = generate_segments_with_points(vertices, segments, D)

        # 삼각형화를 위한 데이터 생성
        triangulation_data = {'vertices': np.array(vertices_with_points), 'segments': np.array(segments_with_points)}

        # 삼각형화
        t = tr.triangulate(triangulation_data, 'p')
        boundary_coords = []

        for tri in t['triangles']:
            v0, v1, v2 = t['vertices'][tri[0]], t['vertices'][tri[1]], t['vertices'][tri[2]]
            vertices_tuple = tuple(sorted([tuple(v0), tuple(v1), tuple(v2)]))
            self.mesh_list.append(vertices_tuple)
            
            # 삼각형의 내부 좌표 계산
            internal_coords = calculate_internal_coordinates_in_triangle(self.width, self.height, v0, v1, v2, D)
            # 내부 좌표 저장
            self.mesh.append(internal_coords)
        for mesh in self.mesh_list:
            internal_coords = calculate_internal_coordinates_in_triangle(self.width, self.height, mesh[0], mesh[1], mesh[2], D)
            for i in internal_coords:
                if not (i[0], i[1]) in self.match_grid_to_mesh.keys():
                    self.match_grid_to_mesh[(i[0], i[1])] = (mesh[0], mesh[1], mesh[2])


        for mesh in self.mesh_list:
            middle_point = ((mesh[0][0]+mesh[1][0]+mesh[2][0])/3, (mesh[0][1]+mesh[1][1]+mesh[2][1])/3)
            
            for obstacle in self.obstacles:
                if len(obstacle) == 4: # 사각형 obstacle
                    if is_point_in_triangle(middle_point, obstacle[0], obstacle[1], obstacle[2]) or is_point_in_triangle(middle_point, obstacle[0], obstacle[2], obstacle[3]) :
                        self.obstacle_mesh.append(mesh)
                elif len(obstacle) == 3:
                    if is_point_in_triangle(middle_point, obstacle[0], obstacle[1], obstacle[2]):
                        self.obstacle_mesh.append(mesh)            

        path = {}
        
        self.next_vertex_matrix = {start: {end: None for end in self.mesh_list} for start in self.mesh_list}
        for i, mesh1 in enumerate(self.mesh_list):
            self.distance[mesh1] = {}
            path[mesh1] = {}
            for j, mesh2 in enumerate(self.mesh_list):
                self.distance[mesh1][mesh2] = 9999999999
                if i == j:
                    self.distance[mesh1][mesh2] = 0
                    self.next_vertex_matrix[mesh1][mesh2] = mesh1
                elif (mesh1 in self.obstacle_mesh or mesh2 in self.obstacle_mesh):
                    # if mesh1 in self.obstacle_mesh:
                    #     print(mesh1, "이 obstacle_mesh에 있음")
                    # elif mesh2 in self.obstacle_mesh:
                    #     print("mesh2가 obstacle_mesh에 있음")
                    self.distance[mesh1][mesh2] = math.inf
                    path[mesh1][mesh2] = None
                elif are_meshes_adjacent(mesh1, mesh2):  # 인접한 경우에만 거리 계산
                    # print("인접함!")
                    mesh1_center = ((mesh1[0][0] + mesh1[1][0] + mesh1[2][0])/3, (mesh1[0][1]+mesh1[1][1]+mesh1[2][1])/3)
                    mesh2_center = ((mesh2[0][0] + mesh2[1][0] + mesh2[2][0])/3, (mesh2[0][1]+mesh2[1][1]+mesh2[2][1])/3)        
                    dist = math.sqrt(pow(mesh1_center[0]-mesh2_center[0], 2) + pow(mesh1_center[1]-mesh2_center[1],2))
                    self.distance[mesh1][mesh2] = dist
                    self.next_vertex_matrix[mesh1][mesh2] = mesh2 
                    if (mesh1 not in self.adjacent_mesh.keys()):
                        self.adjacent_mesh[mesh1] = []
                    self.adjacent_mesh[mesh1].append(mesh2)
                    #path[mesh1][mesh2] = [i, j] if dist < math.inf else None
                else:
                    self.distance[mesh1][mesh2] = math.inf
                    self.next_vertex_matrix[mesh1][mesh2] = None
        
        n = len(mesh)
        

        for mesh1 in self.mesh_list:
            for mesh2 in self.mesh_list:
                for mesh3 in self.mesh_list:
                    i = mesh2
                    k = mesh1
                    j = mesh3
                    if mesh1 in self.obstacle_mesh or mesh3 in self.obstacle_mesh:
                        continue
                    if self.distance[i][k] + self.distance[k][j] < self.distance[i][j]:
                        self.distance[i][j] = self.distance[i][k] + self.distance[k][j]
                        self.next_vertex_matrix[i][j] = self.next_vertex_matrix[i][k]
        for mesh in self.mesh_list:
            if mesh not in self.obstacle_mesh:
                self.pure_mesh.append(mesh)
        # print("pure mesh 개수 : ", len(self.pure_mesh))
        # print("obstacle mesh 개수 : ", len(self.obstacle_mesh))
        # print("첫번쨰 pure mesh : ", self.pure_mesh[0])

        
        boundary_coords = []
        boundary_coords = list(set(map(tuple, boundary_coords)))

        for i in range(self.width):
            for j in range(self.height):
                for mesh in self.pure_mesh:
                    if is_point_in_triangle([i, j], mesh[0], mesh[1], mesh[2]):
                        if mesh not in self.match_mesh_to_grid.keys():
                            self.match_mesh_to_grid[mesh] = []
                        self.match_mesh_to_grid[mesh].append([i, j])
        # for i in range(self.width):
        #     for j in range(self.height):
        #         for mesh in self.pure_mesh:
        #             if [i, j] in self.match_mesh_to_grid[mesh]:
        #                 self.valid_space[(i, j)] = 1
        #                 break
        #             else:
        #                 self.valid_space[(i, j)] = 0
        for i in range(self.width):
            for j in range(self.height):
                self.valid_space[(i, j)] = 1
        for i in range(self.width):
            self.valid_space[(i, 70)] = 0
            self.valid_space[(i, 71)] = 0
        for j in range(self.height):
            self.valid_space[(70, j)] = 0
            self.valid_space[(71, j)] = 0
    def get_path(self, next_vertex_matrix, start, end): #start->end까지 최단 경로로 가려면 어떻게 가야하는지 알려줌 

        if next_vertex_matrix[start][end] is None:
            return []

        path = [start]
        while start != end:
            start = next_vertex_matrix[start][end]
            path.append(start)
        return path

    def extract_map(self, map_num):
        width = 70
        height = 70 
        #좌하단 #우하단 #우상단 #좌상단 순으로 입력해주기
        if map_num == 0:
            self.obstacles.append([[10, 10], [20, 20], [10, 20]])
            self.obstacles.append([[10, 20], [20, 20], [20,50], [10, 50]])
            self.obstacles.append([[20, 40], [50, 40], [50, 50], [20, 50]])
            self.obstacles.append([[40, 10], [60, 20], [40, 20]])

        elif map_num == 1: # 산학협력관 + 잔디밭
            self.obstacles.append([[15, 20], [25, 20], [25, 40], [15, 40]])
            self.obstacles.append([[15, 45], [55, 45], [55, 55], [15, 55]])
            self.obstacles.append([[35, 15], [55, 15], [55, 35]])

            self.spaces_of_map = [[[0, 55], [15, 70]],[[15, 55], [35, 70]],[[35, 55], [55, 70]],[[55, 55], [70 ,70]]
                                ,[[0, 40], [15, 55]],[[15, 40], [35, 45]],[[35, 35], [55, 45]],[[55, 45], [70, 55]],[[55, 35], [70, 45]]
                                ,[[0, 20], [15, 40]],[[25, 20], [35, 40]],[[35, 15], [55, 35]],[[55, 15], [70, 35]]
                                ,[[0, 0], [15, 20]],[[15, 0], [35, 20]],[[35, 0], [55, 15]],[[55, 0], [70, 15]]]
            

        elif map_num == 2: # 제 1공학관
            # 윗 건물
            self.obstacles.append([[10, 52], [60, 52], [60, 60], [10, 60]])
            # 정원
            self.obstacles.append([[32, 26], [44, 26], [44, 40], [32, 40]])
            # 아래 건물
            self.obstacles.append([[10, 8], [44, 8], [44, 16], [10, 16]])
            #오른쪽 건물
            self.obstacles.append([[50, 8], [56, 8], [56, 14], [50, 14]])
            self.obstacles.append([[50, 14], [60, 14], [60, 46], [50, 46]])

            self.spaces_of_map = [[[0, 60],[10, 70]],[[10, 60],[35, 70]],[[35, 60 ],[60, 70]],[[60 ,60],[70, 70]]
                                    ,[[0, 52],[10, 60]],[[0, 40],[16, 52]],[[16, 40],[32, 52]],[[32, 40],[44, 52]],[[44, 46],[60, 52]],[[60, 46],[70, 60]]
                                    ,[[0, 26],[16, 40]],[[16, 26],[32, 40]],[[44, 26],[50, 46]],[[60, 30],[70, 46]]
                                    ,[[0, 16],[16, 26]],[[16, 16],[32, 26]],[[32, 16],[44, 26]],[[44, 8],[50, 26]],[[60, 14],[70, 30]]
                                    ,[[0, 0],[10, 16]],[[10, 0],[27, 8]],[[27, 0],[44, 8]],[[44, 0],[56, 8]],[[56, 0],[70, 14]]]

        elif map_num == 3: # 공학실습동 + 제 2 종합 연구동
            # 왼쪽 건물
            self.obstacles.append([[12, 12], [18, 12], [18, 33], [12, 33]])
            self.obstacles.append([[12, 37], [18, 37], [18, 58], [12, 58]])
            # 중간 건물
            self.obstacles.append([[26, 12], [32, 12], [32, 33], [26, 33]])
            self.obstacles.append([[26, 37], [32, 37], [32, 58], [26, 58]])
            # 오른쪽 건물
            self.obstacles.append([[38, 12], [48, 12], [48, 22], [38, 22]])
            self.obstacles.append([[38, 26], [48, 26], [48, 44], [38, 44]])
            self.obstacles.append([[38, 48], [48, 48], [48, 58], [38, 58]])
            self.obstacles.append([[48, 12], [62, 12], [62, 18], [48, 18]])
            self.obstacles.append([[48, 30], [62, 30], [62, 40], [48, 40]])
            self.obstacles.append([[48, 52], [62, 52], [62, 58], [48, 58]])

            self.spaces_of_map = [[[0, 58],[12, 70]],[[12, 58],[26, 70]],[[26, 58],[38, 70]],[[38, 58],[62, 70]],[[62, 52],[70, 70]]
                                    ,[[0, 37],[12, 58]],[[18, 37],[26, 58]],[[32, 37],[38, 58]],[[38, 44],[48, 48]],[[48, 40],[62, 52]],[[62, 30],[70, 50]]
                                    ,[[0, 33],[12, 37]],[[12, 33],[26, 37]],[[26, 33],[38, 37]]
                                    ,[[0, 12],[12, 33]],[[18, 12],[26, 33]],[[32, 12],[38, 33]],[[38, 22],[48, 26]],[[48, 18],[62, 30]],[[62, 12],[70, 30]]
                                    ,[[0, 0],[12, 12]],[[12, 0],[26, 12]],[[26, 0],[38, 12]],[[38, 0],[62, 12]],[[62, 0],[70, 12]]]
        elif map_num == 4: # 벤젠고리관
            # 아래 건물
            self.obstacles.append([[48, 10], [58, 20], [58, 32], [44, 18]])
            self.obstacles.append([[26, 10], [44, 10], [40, 18], [26, 18]])
            # 중간 건물
            self.obstacles.append([[32, 24], [50, 42], [44, 48], [26, 30]])
            # 윗 건물
            self.obstacles.append([[12, 28], [20, 28], [20, 42], [12, 46]])
            self.obstacles.append([[12, 50], [20, 46], [32, 58], [26, 64]]) 

            self.spaces_of_map = [[[0, 50],[20, 70]],[[20, 58],[32, 70]],[[32, 58],[44, 70]],[[44, 42],[70, 70]]
                                    ,[[0, 18],[12, 50]],[[12, 42],[20, 50]],[[20, 30],[32, 58]],[[32, 36],[44, 58]]
                                    ,[[12, 18],[32, 30]],[[32, 18],[44, 36]],[[44, 18],[58, 42]],[[58, 20],[70, 42]]
                                    ,[[0, 0],[12, 18]],[[12, 0],[32, 18]],[[40, 10],[48, 18]],[[32, 0],[48, 10]],[[48, 0],[70, 20]]]

        elif map_num == 5: # 경영관 + 퇴계 인문관
            # 왼쪽 건물
            self.obstacles.append([[18, 10], [24, 10], [24, 28], [18, 28]])
            self.obstacles.append([[12, 20], [18, 20], [18, 26], [12, 26]])
            # # 오른쪽 건물
            self.obstacles.append([[34, 10], [46, 10], [46, 16], [34, 16]])
            self.obstacles.append([[46, 10], [56, 10], [56, 28], [46, 28]])
            # # 윗 건물
            self.obstacles.append([[18, 34], [24, 34], [24, 60], [18, 60]])
            self.obstacles.append([[24, 54], [38, 54], [38, 60], [24, 60]]) 
            self.obstacles.append([[46, 40], [52, 40], [52, 48], [46, 48]]) 
            self.obstacles.append([[24, 34], [56, 34], [56, 40], [24, 40]])
            
            self.spaces_of_map = [[[0, 47],[18, 70]],[[18, 60],[38, 70]],[[38, 54],[70 ,70]]
                                    ,[[0, 34],[18, 47]],[[24, 40],[46, 54]],[[46, 40],[70, 54]]
                                    ,[[0, 20],[18, 34]],[[18, 28],[34, 34]],[[34, 28],[56, 34]]
                                    ,[[0, 0],[18, 20]],[[24, 10],[34, 28]],[[34, 16],[46, 28]],[[56, 10],[70, 34]]
                                    ,[[18, 0],[34, 10]],[[34, 0],[56, 10]],[[56, 0],[70, 10]]]



# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    def construct_map(self):
        for i in range(len(self.walls)):
            a = WallAgent(self.agent_num, self, self.walls[i], 9)
            self.agent_num+=1
            #self.schedule_e.add(a)
            self.grid.place_agent(a, self.walls[i])
            self.valid_space[(self.walls[i][0], self.walls[i][1])] = 0
        for i in range(len(self.obstacles)):
            for each_point in  get_points_within_polygon(self.obstacles[i], 1):
                self.obstacles_grid_points.append(each_point)
                a = WallAgent(self.agent_num, self, each_point, 9)
                self.agent_num+=1
                #self.schedule_e.add(a)
                self.valid_space[(each_point[0], each_point[1]-1)] = 0
                self.valid_space[(each_point[0], each_point[1])] = 0
                self.grid.place_agent(a, each_point)

        num = 0
        exit_grid = []
        for e in self.exit_list:
            exit_grid.append(get_points_within_polygon(e, 1))
            for each_point in get_points_within_polygon(e, 1):
                self.exit_grid[each_point[0]][each_point[1]] = 1
        for i in range(len(exit_grid)):
            for each_point in exit_grid[i]:
                a = WallAgent(self.agent_num, self, self.exit_list[i][0], 10)
                self.agent_num+=1
                self.grid.place_agent(a, each_point)

        # for mesh in self.mesh:
        #     num +=1 
        #     for i in range(len(mesh)):
        #         a = CrowdAgent(self.agent_num, self, [mesh[i][0], mesh[i][1]], 102+num%5)
        #         self.agent_num+=1
        #         self.schedule_e.add(a)
        #         self.grid.place_agent(a, [mesh[i][0], mesh[i][1]])


                                  
    def make_robot(self):
        self.robot_placement() #로봇 배치 


    def reward_distance_sum(self):
        result = 0
        for i in self.agents:
            if(i.dead == False and (i.type==0 or i.type==1)):
                result += i.danger
        return result 
          

    def make_exit(self):
        exit_width = 5
        exit_height = 5
        self.exit_list = [[(0,0), (exit_width, 0), (exit_width, exit_height), (0, exit_height)],
                         [(self.width-exit_width-1,0), (self.width-1, 0), (self.width-1, exit_height), (self.width-exit_width-1, exit_height)],
                         [(0, self.height-exit_height-2), (exit_width, self.height-exit_height-2), (exit_width, self.height-1), (0, self.height-1)],
                         [(self.width-exit_width-1, self.height-exit_height-2), (self.width-1, self.height-exit_height-2), (self.width-1, self.height-1), (self.width-exit_width-1, self.height-1)]
                        ]
        self.exit_point = [[(exit_width)/2, (exit_height)/2],
                           [(self.width-exit_width-1+self.width-1)/2, (exit_height)/2],
                           [(exit_width)/2, (self.height-exit_height-1+self.height-1)/2],
                           [(self.width-exit_width-1+self.width-1)/2, (self.height-exit_height-1+self.height-1)/2]
                           ]
        
        
        return 0

    def make_random_exit(self):
        exit_width = 5
        exit_height = 5

        # 모든 출구 목록 정의
        all_exits = [
            [(0, 0), (exit_width, 0), (exit_width, exit_height), (0, exit_height)],  # 왼쪽 위
            [(self.width-exit_width-1, 0), (self.width-1, 0), (self.width-1, exit_height), (self.width-exit_width-1, exit_height)],  # 오른쪽 위
            [(0, self.height-exit_height-2), (exit_width, self.height-exit_height-2), (exit_width, self.height-1), (0, self.height-1)],  # 왼쪽 아래
            [(self.width-exit_width-1, self.height-exit_height-2), (self.width-1, self.height-exit_height-2), (self.width-1, self.height-1), (self.width-exit_width-1, self.height-1)]  # 오른쪽 아래
        ]
        
        all_exit_points = [
            [(exit_width)/2, (exit_height)/2],  # 왼쪽 위
            [(self.width-exit_width-1+self.width-1)/2, (exit_height)/2],  # 오른쪽 위
            [(exit_width)/2, (self.height-exit_height-1+self.height-1)/2],  # 왼쪽 아래
            [(self.width-exit_width-1+self.width-1)/2, (self.height-exit_height-1+self.height-1)/2]  # 오른쪽 아래
        ]
        
        # 랜덤하게 출구 선택
        index = random.randint(0, len(all_exits) - 1)
        self.exit_list = [all_exits[index]]
        self.exit_point = [all_exit_points[index]]
        
        return 0


    def check_bridge(self, space1, space2):
        visited = {}
        for i in self.space_graph.keys():
            visited[i] = 0
        
        stack = [space1]
        while(stack):
            node = stack.pop()
            if(visited[((node[0][0], node[0][1]), (node[1][0], node[1][1]))] == 0):
                visited[((node[0][0], node[0][1]), (node[1][0], node[1][1]))] = 1
                stack.extend(self.space_graph[((node[0][0], node[0][1]), (node[1][0], node[1][1]))])
        if (visited[space2] == 0):
            return 0
        else:
            return 1
    def way_to_exit(self):
        visible_distance = 6

        # 출구를 순회하면서 각 출구에 대한 x1, x2, y1, y2를 구합니다.
        for exit_rec in self.exit_recs:
            x1, x2 = float('inf'), float('-inf')
            y1, y2 = float('inf'), float('-inf')
            
            # 출구의 경계좌표를 찾습니다.
            for i in exit_rec:
                if i[0] > x2:
                    x2 = i[0]
                if i[0] < x1:
                    x1 = i[0]
                if i[1] > y2:
                    y2 = i[1]
                if i[1] < y1:
                    y1 = i[1]

            # 좌표 범위에 대해 탐색
            for j in range(y1, y2 + 1):
                self.recur_exit(x1, j, visible_distance, "l")
                self.recur_exit(x2, j, visible_distance, "r")

            for j in range(x1, x2 + 1):
                self.recur_exit(j, y1, visible_distance, "d")
                self.recur_exit(j, y2, visible_distance, "u")

    def recur_exit(self, x, y, visible_distance, direction):
        # 기저 조건 확인
        if visible_distance < 1:
            return
        
        # 경계값 확인
        max_index = len(self.grid_to_space) - 1
        if x < 0 or y < 0 or x > max_index or y > max_index:
            return
        
        # 방문한 위치가 방 내부라면 반환
        if self.grid_to_space[x][y] in self.room_list:
            return

        # 현재 위치를 경로로 설정
        self.exit_way_rec[x][y] = 1
        
        # 방향에 따른 재귀 호출
        if direction == "l":
            self.recur_exit(x - 1, y - 1, visible_distance - 2, "l")
            self.recur_exit(x - 1, y, visible_distance - 1, "l")
            self.recur_exit(x - 1, y + 1, visible_distance - 2, "l")
        elif direction == "r":
            self.recur_exit(x + 1, y - 1, visible_distance - 2, "r")
            self.recur_exit(x + 1, y, visible_distance - 1, "r")
            self.recur_exit(x + 1, y + 1, visible_distance - 2, "r")
        elif direction == "u":
            self.recur_exit(x - 1, y + 1, visible_distance - 2, "u")
            self.recur_exit(x, y + 1, visible_distance - 1, "u")
            self.recur_exit(x + 1, y + 1, visible_distance - 2, "u")
        else:  # direction == "d"
            self.recur_exit(x + 1, y - 1, visible_distance - 2, "d")
            self.recur_exit(x, y - 1, visible_distance - 1, "d")
            self.recur_exit(x - 1, y - 1, visible_distance - 2, "d")



    def robot_placement(self): # 야외 공간에 무작위로 로봇 배치 
        # get_point = self.exit_point[random.randint(0, len(self.exit_point)-1)]
        # get_point = (int(round(get_point[0])), int(round(get_point[1])))
        # self.agent_id = self.agent_id + 10
        # self.robot = RobotAgent(self.agent_id, self, [get_point[0],get_point[1]], 3)
        # self.agent_id = self.agent_id + 10
        # self.schedule.add(self.robot)
        # self.grid.place_agent(self.robot, (get_point[0], get_point[1]))

        self.agent_id = self.agent_id + 10
        self.robot = RobotAgent(self.agent_id, self, [20, 35], 3)
        self.agent_id = self.agent_id + 10
        self.schedule.add(self.robot)
        self.grid.place_agent(self.robot, (20, 35))
    

    
    
    def random_agent_distribute_outdoor(self, agent_num, ran):
        

        space_num = len(self.pure_mesh)
        
        
        space_agent = agent_num
        agent_location = []

        for i in range(agent_num):
            assign_mesh_num = random.randint(0, space_num-1)
            assigned_mesh = self.pure_mesh[assign_mesh_num]
            assigned_coordinates = self.match_mesh_to_grid[assigned_mesh]

            assigned = assigned_coordinates[random.randint(0, len(assigned_coordinates)-1)]
            assigned = [int(assigned[0]), int(assigned[1])]
            if not assigned in agent_location:
                agent_location.append(assigned)
                a = CrowdAgent(self.agent_num, self, assigned, 1)
                self.agent_num += 1
                self.schedule.add(a)
                self.grid.place_agent(a, assigned)




    def floyd_warshall(self): #공간과 공간사이의 최단 경로를 구하는 알고리즘 

        vertices = list(self.space_graph.keys())
        n = len(vertices)
        distance_matrix = {start: {end: float('infinity') for end in vertices} for start in vertices}  
        next_vertex_matrix = {start: {end: None for end in vertices} for start in vertices}
        
    
        for start in self.space_graph.keys():
            for end in self.space_graph[start]:
                end_t = ((end[0][0], end[0][1]), (end[1][0],end[1][1]))
                start_xy = [(start[0][0]+start[1][0])/2, (start[0][1]+start[1][1])/2]
                end_xy = [(end[0][0]+end[1][0])/2, (end[0][1]+end[1][1])/2]
                distance_matrix[start][end_t] = math.sqrt(pow(start_xy[0]-end_xy[0],2)+pow(start_xy[1]-end_xy[1], 2))
                next_vertex_matrix[start][end_t] = end_t

        for k in vertices:
            for i in vertices:
                for j in vertices:
                    if distance_matrix[i][j] > distance_matrix[i][k] + distance_matrix[k][j]:
                        distance_matrix[i][j] = distance_matrix[i][k] + distance_matrix[k][j]
                        next_vertex_matrix[i][j] = next_vertex_matrix[i][k]
        return [next_vertex_matrix, distance_matrix]

    def get_path(self, next_vertex_matrix, start, end): #start->end까지 최단 경로로 가려면 어떻게 가야하는지 알려줌 
        start = ((start[0][0], start[0][1]), (start[1][0], start[1][1]))
        end = ((end[0][0], end[0][1]), (end[1][0], end[1][1]))
        if next_vertex_matrix[start][end] is None:
            return []

        path = [start]
        while start != end:
            start = next_vertex_matrix[start][end]
            path.append(start)
        return path
        


    def step(self):
        self.step_n += 1
        """Advance the model by one step."""
        global started
        max_id = 1
        if(started):
            for agent in self.agents:
                if(agent.type==1 or agent.type==0):
                    if(agent.unique_id > max_id):
                        max_id = agent.unique_id
            #self.difficulty_f()
            for agent in self.agents:
                if(max_id == agent.unique_id):
                    agent.dead = True
            started = 0
            max_id = 1
            for agent in self.agents:
                if (agent.unique_id > max_id and (agent.type==0 or agent.type==1)):
                    max_id = agent.unique_id
            for agent in self.agents:
                if(max_id == agent.unique_id):
                    agent.dead = True 
        self.step_count += 1

        state = self.return_current_image()
        if(self.using_model):
            self.checking_reward += self.reward_evacuation()
        if(self.using_model and self.step_n%3==0):
            action = self.sac_agent.select_action(state)
            self.robot.receive_action(action[0])
        if(self.using_model and self.step_n%3==2):
            print("reward : ", self.checking_reward)
            self.checking_reward = 0

        self.schedule.step()
        self.datacollector_currents.collect(self)  # passing the model

        
        
        

    def check_reward(self, reference_reward):
        if self.step_count <= len(reference_reward*100):
            return self.evacuated_agents()-reference_reward[int(self.step_count/100)]
        else :
            return self.evacuated_agents()-self.total_agents

    def reward_based_alived(self):
        reward = 0
        num = 0
        
        reward = -self.alived_agents()/self.total_agents 

    def reward_based_gain(self):
        
        reward=0
        #robot이 agent를 끌어당기면 +reward
        for agent in self.agents:
            if(agent.type == 0 or agent.type == 1 or agent.type == 2 ) and (agent.dead == False):
                if(agent.robot_tracked>0):
                    reward += agent.gain2
        #reward -= self.robot.detect_abnormal_order

        reward = reward/3

        if(reward<-100):
            reward = -100
        

        #print("tracked 되고 있는 수 : ", num)
        return reward

    def reward_evacuation(self):
        if(self.step_n<3):
            return 0
        return (self.robot.previous_danger - self.robot.danger)/10
        

    def return_agent_id(self, agent_id):
        for agent in self.agents:
            if(agent.unique_id == agent_id):
                return agent
        return None
    
    def use_model(self, file_path):
        input_shape = (70, 70)
        num_actions = 4

        self.sac_agent = SACAgent(input_shape, num_actions, start_epsilon=0)
        self.sac_agent.load_model(file_path)

        self.using_model = True


    
    def return_current_image(self):

        image = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for agent in self.agents:
            if(agent.type==9):
                image[agent.pos[0]][agent.pos[1]] = 40 # 벽
            if(agent.type==10):
                image[agent.pos[0]][agent.pos[1]] = 90 # 출구
            if(agent.type == 0 or agent.type == 1 or agent.type == 2):
                image[int(round(agent.xy[0]))][int(round(agent.xy[1]))] = 140 #agent
            if(agent.type == 3):
                image[int(round(agent.xy[0]))][int(round(agent.xy[1]))] = 200 #robot

        # for i in range(self.width):
        #     for j in range(self.height):
        #         print(f'{i}, {j} : {image[i][j]}')
        
        return image

    
    def return_robot(self):
        return self.robot



    @staticmethod
    def current_healthy_agents(model) -> int:
        """Returns the total number of healthy agents.

        Args:
            model (SimulationModel): The model instance.

        Returns:
            (Integer): Number of Agents.
        """
        return sum([1 for agent in model.schedule_e.agents if agent.health > 0]) ### agent의 health가 0이어야 cureent_healthy_agents 수에 안 들어감
                                                                               ### agent.py 에서 exit area 도착했을 때 health를 0으로 바꿈


    @staticmethod
    def current_non_healthy_agents(model) -> int:
        """Returns the total number of non healthy agents.

        Args:
            model (SimulationModel): The model instance.

        Returns:
            (Integer): Number of Agents.
        """
        return sum([1 for agent in model.schedule_e.agents if agent.health == 0])
