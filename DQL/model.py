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
from typing import List, Tuple
import math
import numpy as np
from itertools import product
import collections

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

from ADDS_AS_reinforcement import DiffusionQLAgent, ReplayBuffer, PolicyNetwork, QNetwork, ACTION_SCALE
from Start_training import REWARD_A, REWARD_B, REWARD_C, REWARD_D, REWARD_E, REWARD_F, REWARD_G, REWARD_H, REWARD_I, REWARD_J, REWARD_K, FINISHED_BONUS, MAP_NUM, MAP_NUM_RANDOM

def _point_on_segment(p: Tuple[int, int],
                      a: Tuple[int, int],
                      b: Tuple[int, int]) -> bool:
    """점 p가 선분 ab 위에 있으면 True (벽 위 스폰 방지)."""
    (px, py), (ax, ay), (bx, by) = p, a, b
    cross  = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if cross != 0:
        return False
    dot = (px - ax) * (px - bx) + (py - ay) * (py - by)
    return dot <= 0  # 가운데 있으면 0보다 작거나 같음

def _point_in_polygon(p: Tuple[int, int],
                      poly: List[Tuple[int, int]]) -> bool:
    """
    홀수-짝수(레이캐스팅) 규칙.
    경계 위에 있으면 True를 즉시 반환해 ‘안전하지 않은’ 좌표로 취급.
    """
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]

        # ① 경계 위 체크
        if _point_on_segment(p, a, b):
            return True

        # ② 내부 여부 토글
        xi, yi = a
        xj, yj = b
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
    return inside

def _sample_safe_cell(self,
                      padding: int = 5,
                      max_attempts: int = 2000) -> Tuple[int, int]:
    """
    패딩을 둔 영역([padding, width-padding), [padding, height-padding))
    안에서 장애물·기존 로봇과 충돌하지 않는 좌표를 무작위로 찾는다.
    찾지 못하면 ValueError.
    """
    for _ in range(max_attempts):
        x = random.randint(padding, self.width  - padding - 1)
        y = random.randint(padding, self.height - padding - 1)
        p = (x, y)

        # ─ 장애물 충돌 검사
        collision = any(_point_in_polygon(p, poly)
                        for poly in self.obstacles)
        if collision:
            continue

        # ─ 이미 사용된 위치인지 검사
        if p in self._occupied_cells:   # ← robot_placement에서 관리
            continue

        return p
    raise ValueError("안전 스폰 위치를 찾지 못했습니다 ‑ padding 값·장애물 배치를 확인하세요.")




def are_meshes_adjacent(mesh1, mesh2):
    # 두 mesh의 공통 꼭짓점의 개수를 센다
    common_vertices = set(mesh1) & set(mesh2)
    return len(common_vertices) >= 2  # 공통 꼭짓점이 두 개 이상일 때 인접하다고 판단R


def _dist2(p, q):         # 제곱거리
    return (p[0]-q[0])**2 + (p[1]-q[1])**2

def _pt_seg_dist(p, a, b):
    """점-선분 최소거리^2 (벡터 투영)"""
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx-ax, by-ay
    if dx == dy == 0:
        return _dist2(p, a)
    t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy)/(dx*dx+dy*dy)))
    proj = (ax + t*dx, ay + t*dy)
    return _dist2(p, proj)

def poly_poly_min_dist2(poly1, poly2):
    """두 다각형(꼭짓점 리스트) 사이의 최소 거리^2"""
    d2 = math.inf
    # 꼭짓점-선분 거리
    for p in poly1:
        for i in range(len(poly2)):
            d2 = min(d2,
                     _pt_seg_dist(p, poly2[i], poly2[(i+1)%len(poly2)]))
    for p in poly2:
        for i in range(len(poly1)):
            d2 = min(d2,
                     _pt_seg_dist(p, poly1[i], poly1[(i+1)%len(poly1)]))
    return d2

def polygon_area(poly):
    """임의 다각형(꼭짓점 시계·반시계) 넓이 – Shoelace 공식"""
    area = 0.5 * abs(sum(
        x0*y1 - x1*y0
        for (x0, y0), (x1, y1) in zip(poly, poly[1:]+[poly[0]])
    ))
    return area

def is_collision(self, pos_f):
    """
    연속 좌표 pos_f = (x, y)가
    ▸ 맵 밖 ▸ valid_space==0 ▸ obstacle polygon 내부
    중 하나면 True
    """
    x, y = pos_f
    if x < 0 or y < 0 or x >= self.width or y >= self.height:
        return True
    if not self.valid_space[(int(x), int(y))]:
        return True
    return any(_point_in_polygon((x, y), poly) for poly in self.obstacles)


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

def add_gaussian_blob(image, center, amplitude, kernel_size=5, sigma=1.0):
    cx, cy = center
    half_size = kernel_size // 2
    x_start = max(0, cx - half_size)
    y_start = max(cy - half_size, 0)
    x_end = min(cx + half_size + 1, image.shape[0])
    y_end = min(cy + half_size + 1, image.shape[1])

    k_x_start = half_size - (cx - x_start)
    k_y_start = half_size - (cy - y_start)

    k_x_start = half_size - (cx - x_start)

 
class FightingModel(Model):
    """A model with some number of agents."""

    def __init__(self, number_agents: int, width: int, height: int, model_num = -1, robot = 'Q'):
        self.crowds = []
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
        if(MAP_NUM != -2 and MAP_NUM != -1):
            self.extract_map_50(self.map_num)
        if (MAP_NUM == -1):
            map_num_candidates = MAP_NUM_RANDOM
            # self.map_num = random.choice(map_num_candidates)
            self.extract_map_50(-1)    
        self.distance = {}  
        self.schedule_e = RandomActivation(self)
        self.schedule = RandomActivation(self)
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
        self.valid_space = collections.defaultdict(lambda: 1)
        self.grid = MultiGrid(width, height, False)
        self.headingding = ContinuousSpace(width, height, False, 0, 0)
        self.fill_outwalls(width, height)
        self.mesh_map()
        self.make_random_exit()
        self.construct_map()
        self.calculate_mesh_danger()
        self.exit_list = []
        self.random_agent_distribute_outdoor(number_agents, 1)
        self.make_robot()
        #self.visualize_danger()
        self.robot_xy = [0, 0]
        self.robot_mode = "GUIDE"
        self.step_count = 0

        self.now_evacuated = 0
        self.now_evacuated_with_robot = 0

        self.previous_evacuated = 0
        self.previous_evacuated_with_robot = 0

        self.before_minimum_distance = 0
        self.minimum_distance = 0
        self.new_founded_agent_danger = 0

        # for i in range(50):
        #     for j in range(50):
        #         print("(", i, j, ")", self.valid_space[(i, j)])
    def alived_agents(self):
        alived_agents = self.total_agents
        for i in self.crowds:
            if(i.dead == 1):
                alived_agents -= 1
        return alived_agents 

    def evacuated_agents(self):
        evacuated_agents = 0
        for i in self.schedule.agents:
            if((i.type==0 or i.type==1 or i.type==2) and i.dead == 1):
                evacuated_agents += 1
        return evacuated_agents
    
    def evacuated_agents_with_robot(self):
        evacuated_agents_with_robot = 0
        for i in self.schedule.agents:
            if((i.type==0 or i.type==1 or i.type==2) and i.dead == 1 and i.is_effected_by_robot == 1):
                evacuated_agents_with_robot += 1
        return evacuated_agents_with_robot

    
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
        candidates = [(x+1,y+1), (x+1, y), (x, y+1), (x-1, y-1), (x-1, y), (x, y-1), (x-1, y+1), (x+1, y-1)]
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

    def mesh_map(self, D: int = 5):
        PADDING = 8

        self.mesh_list, self.mesh = [], []
        self.match_grid_to_mesh = {}
        self.match_mesh_to_grid = {}      # ← 여기서 바로 채움
        tri_centers, square_of = {}, {}

        # ── ① 삼각형 생성
        for gx in range(PADDING, self.width - PADDING, D):
            for gy in range(PADDING, self.height - PADDING, D):
                p00, p10 = (gx, gy), (gx + D, gy)
                p01, p11 = (gx, gy + D), (gx + D, gy + D)
                for tri in ((p00, p10, p11), (p00, p11, p01)):  # ↘, ↙
                    cx = sum(v[0] for v in tri) / 3
                    cy = sum(v[1] for v in tri) / 3
                    if any(_point_in_polygon((cx, cy), obs) for obs in self.obstacles):
                        self.obstacle_mesh.append(tri)
                        continue

                    cells = calculate_internal_coordinates_in_triangle(
                                self.width, self.height,
                                tri[0], tri[1], tri[2], D=1)
                    if not cells:
                        continue

                    # ── ★ 삼각형->셀 매핑을 즉시 저장 ★
                    self.match_mesh_to_grid[tri] = []

                    self.mesh_list.append(tri)
                    self.mesh.append(cells)
                    tri_centers[tri] = (cx, cy)
                    square_of[tri]   = (gx // D, gy // D)

                    for (x, y) in cells:
                        # 그리드→삼각형 매핑은 처음 값만 유지
                        self.match_grid_to_mesh.setdefault((x, y), tri)
                        self.match_mesh_to_grid[tri].append([x, y])

                self.pure_mesh = [m for m in self.mesh_list if m not in self.obstacle_mesh]

            # ── ② 인접‧거리 행렬
            self.adjacent_mesh, self.distance = {}, {}
            self.next_vertex_matrix = {u: {v: None for v in self.mesh_list}
                                    for u in self.mesh_list}

            for u in self.mesh_list:
                self.distance[u] = {}
                cu, su = tri_centers[u], square_of[u]
                for v in self.mesh_list:
                    if u == v:
                        self.distance[u][v] = 0
                        self.next_vertex_matrix[u][v] = u
                        continue
                    if u in self.obstacle_mesh or v in self.obstacle_mesh:
                        self.distance[u][v] = math.inf
                        continue

                    sv = square_of[v]
                    dx, dy = abs(su[0]-sv[0]), abs(su[1]-sv[1])

                    # 같은 정사각형 또는 상·하·좌·우 이웃 정사각형이면 인접
                    if (dx == 0 and dy == 0) or (dx + dy == 1):
                        dist = math.hypot(cu[0]-tri_centers[v][0], cu[1]-tri_centers[v][1])
                        self.distance[u][v] = dist
                        self.next_vertex_matrix[u][v] = v
                        self.adjacent_mesh.setdefault(u, []).append(v)
                    else:
                        self.distance[u][v] = math.inf

            # ── ③ Floyd–Warshall
            for k in self.mesh_list:
                for i in self.mesh_list:
                    dik = self.distance[i][k]
                    if dik == math.inf: continue
                    for j in self.mesh_list:
                        alt = dik + self.distance[k][j]
                        if alt < self.distance[i][j]:
                            self.distance[i][j] = alt
                            self.next_vertex_matrix[i][j] = self.next_vertex_matrix[i][k]

    def get_path(self, next_vertex_matrix, start, end): #start->end까지 최단 경로로 가려면 어떻게 가야하는지 알려줌 

        if next_vertex_matrix[start][end] is None:
            return []

        path = [start]
        while start != end:
            start = next_vertex_matrix[start][end]
            path.append(start)
        return path

    def extract_map(self, map_num):
        width = 50
        height = 50 

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


    def extract_map_50(self, map_num):
        width = 50
        height = 50 
        #좌하단 #우하단 #우상단 #좌상단 순으로 입력해주기
        scale = 5/7

        if map_num == -1:
            # ─────────────────────────────────────────
            # 파라미터
            # ─────────────────────────────────────────
            PADDING     = 8          # 외곽 여백
            max_trials  = 800         # 충돌 피하기 위한 난수 시도 한계
            min_side, max_side = 4, 20
            min_gap     = 8           # 장애물 간 최소 간격
            coverage_target = (0.60, 0.70)   # 맵 면적 대비 10~25 % 사이 될 때 정지
            ratio_low, ratio_high = coverage_target
            # ----------------------------------------

            map_area      = width * height
            used_area     = 0.0
            boxes_aabb    = []

            def aabb(poly):
                xs, ys = zip(*poly)
                return min(xs), max(xs), min(ys), max(ys)

            def overlap(b1, b2):
                return not (b1[1] + min_gap < b2[0] or b2[1] + min_gap < b1[0] or
                            b1[3] + min_gap < b2[2] or b2[3] + min_gap < b1[2])

            for _ in range(max_trials):
                # coverage 상한 초과 시 중단
                if used_area / map_area >= ratio_high:
                    break

                # ─ ① 후보 다각형 랜덤 생성
                if random.random() < 0.7:                # Rectangle
                    w = random.randint(min_side, max_side)
                    h = random.randint(min_side, max_side)
                    x0 = random.randint(PADDING, width  - PADDING - w - 1)
                    y0 = random.randint(PADDING, height - PADDING - h - 1)
                    poly = [[x0, y0], [x0+w, y0], [x0+w, y0+h], [x0, y0+h]]
                else:                                    # Triangle
                    base = random.randint(min_side, max_side)
                    x0 = random.randint(PADDING, width  - PADDING - base - 1)
                    y0 = random.randint(PADDING, height - PADDING - base - 1)
                    poly = [[x0, y0],
                            [x0+base, y0],
                            [x0+random.randint(0, base), y0+base]]

                box = aabb(poly)
                # ─ ② 충돌·간격 검사
                if any(overlap(box, b) for b in boxes_aabb):
                    continue

                poly_area = polygon_area(poly)
                # coverage 하한 도달 전이라면 면적 큰 다각형 더 환영 → 제한 X
                # coverage 상한에 근접해가면, 초과 여부 확인
                if (used_area + poly_area) / map_area > ratio_high:
                    continue                             # 넣으면 초과 → 패스

                # ─ ③ 채택
                boxes_aabb.append(box)
                self.obstacles.append(poly)
                used_area += poly_area

                # 하한을 최초로 넘겼다면, 이후 확률 70%로 즉시 종료(자연스러운 분포)
                if used_area / map_area >= ratio_low and random.random() < 0.7:
                    break

        # 무작위 생성이 끝났습니다. ─ 아래에 기존 else-if 블록들이 이어집니다.


        if map_num == 1: # 왼쪽 상단
            self.obstacles.append([[10, 10], [16, 10], [10, 16]])
            self.obstacles.append([[34, 10], [40, 10], [40, 16]])
            self.obstacles.append([[10, 34], [16, 40], [10, 40]])
            self.obstacles.append([[40, 34], [40, 40], [34, 40]])
        
        elif map_num == 2: #오른쪽 하단
            self.obstacles.append([[8, 8], [25, 8], [25, 13], [8, 13]])
            self.obstacles.append([[8, 16], [14, 16], [14, 25], [8, 25]])
            self.obstacles.append([[20, 16], [25, 16], [25, 34], [20, 34]])
            self.obstacles.append([[33, 16], [40, 16], [40, 34], [33, 34]])
            self.obstacles.append([[8, 40], [13, 40], [13, 45], [8, 45]])
            self.obstacles.append([[18, 40], [35, 40], [35, 45], [18, 45]])
        
        elif map_num == 3: #왼쪽 하단
            self.obstacles.append([[15, 8], [20, 8], [20, 15], [15, 20]])
            self.obstacles.append([[20, 21], [20, 28], [15, 28]])
            self.obstacles.append([[35, 8], [43, 8], [43, 15], [35, 15]])
            self.obstacles.append([[35, 21], [43, 21], [43, 28], [35, 28]])
            self.obstacles.append([[7, 40], [40, 40], [40, 45], [7, 45]])

        elif map_num == 4: #오른쪽 상단
            self.obstacles.append([[12, 12], [18, 12], [18, 25], [12, 25]])
            self.obstacles.append([[25, 12], [40, 12], [40, 20], [25, 20]])
            self.obstacles.append([[12, 38], [30, 38], [30, 45], [17, 45]])
        
        elif map_num == 5: #오른쪽 상단
            self.obstacles.append([[10, 10], [15, 10], [15, 35], [10, 35]])
            self.obstacles.append([[20, 10], [25, 10], [25, 35], [20, 35]])


        elif map_num == 6:
            self.obstacles.append([[10, 10], [20, 10], [20, 20], [10, 20]])
            self.obstacles.append([[10, 30], [40, 30], [40, 40], [10, 40]])


        elif map_num == 7:
            self.obstacles.append([[10, 30], [20, 30], [28, 40], [10, 40]])
            self.obstacles.append([[30, 10], [35, 10], [35, 30], [30, 30]])
            self.obstacles.append([[10, 10], [15, 10], [15, 15], [10, 15]])
            self.obstacles.append([[15, 20], [22, 20], [22, 25], [15, 25]])
            self.obstacles.append([[35, 35], [40, 35], [40, 40], [35, 40]])
            # self.obstacles.append([[30, 30], [35, 30], [35, 35]]) ## ?? 이 줄만 추가하면 세그멘테이션 오류 (코어 덤프됨) 뜸 왤까?

        
        elif map_num == 8:
            self.obstacles.append([[10, 15], [25, 15], [10, 40]])
            self.obstacles.append([[30, 20], [40, 20], [40, 35], [30, 35]])

            
        elif map_num == 9:
            self.obstacles.append([[10, 15], [35, 15], [35, 25], [10, 25]])
            self.obstacles.append([[0, 35], [20, 35], [20, 40], [0, 40]])
            self.obstacles.append([[30, 40], [40, 30], [40, 40]])
        

        elif map_num == 10:
            self.obstacles.append([[10, 30], [15, 30], [15, 40], [10, 40]])
            self.obstacles.append([[30, 20], [40, 20], [40, 35], [30, 35]])
            self.obstacles.append([[41, 10], [45, 10], [45, 25], [41, 25]])

        elif map_num == 11:
            self.obstacles.append([[10, 5], [40, 5], [40, 20]])
            self.obstacles.append([[6, 25], [10, 25], [10, 40], [6, 40]])
            self.obstacles.append([[11, 32], [40, 32], [40, 40], [11, 40]])


        elif map_num == 12:
            self.obstacles.append([[15, 10], [40, 35], [35, 40], [10, 15]])
            self.obstacles.append([[20, 0], [35, 0], [35, 15]])
            self.obstacles.append([[10, 40], [20, 49], [10, 49]])

        
            

        # elif map_num == 1:  # 산학협력관 + 잔디밭
        #     self.obstacles.append([[15 * scale, 15 * scale], [25 * scale, 15 * scale], [25 * scale, 35 * scale], [15 * scale, 35 * scale]])
        #     self.obstacles.append([[15 * scale, 45 * scale], [55 * scale, 45 * scale], [55 * scale, 55 * scale], [15 * scale, 55 * scale]])
        #     self.obstacles.append([[35 * scale, 15 * scale], [55 * scale, 15 * scale], [55 * scale, 35 * scale]])

        #     self.spaces_of_map = [
        #         [[0, 55], [15, 70]], [[15, 55], [35, 70]], [[35, 55], [55, 70]], [[55, 55], [70, 70]],
        #         [[0, 40], [15, 55]], [[15, 40], [35, 45]], [[35, 35], [55, 45]], [[55, 45], [70, 55]],
        #         [[55, 35], [70, 45]], [[0, 20], [15, 40]], [[25, 20], [35, 40]], [[35, 15], [55, 35]],
        #         [[55, 15], [70, 35]], [[0, 0], [15, 20]], [[15, 0], [35, 20]], [[35, 0], [55, 15]], [[55, 0], [70, 15]]
        #     ]

        # elif map_num = 2:  # 제 1공학관
        #     # 윗 건물
        #     self.obstacles.append([[10 * scale, 52 * scale], [60 * scale, 52 * scale], [60 * scale, 60 * scale], [10 * scale, 60 * scale]])
        #     # 정원
        #     self.obstacles.append([[32 * scale, 26 * scale], [44 * scale, 26 * scale], [44 * scale, 40 * scale], [32 * scale, 40 * scale]])
        #     # 아래 건물
        #     self.obstacles.append([[10 * scale, 8 * scale], [44 * scale, 8 * scale], [44 * scale, 16 * scale], [10 * scale, 16 * scale]])
        #     # 오른쪽 건물
        #     self.obstacles.append([[50 * scale, 8 * scale], [56 * scale, 8 * scale], [56 * scale, 14 * scale], [50 * scale, 14 * scale]])
        #     self.obstacles.append([[50 * scale, 14 * scale], [60 * scale, 14 * scale], [60 * scale, 46 * scale], [50 * scale, 46 * scale]])

        #     self.spaces_of_map = [
        #         [[0, 60], [10, 70]], [[10, 60], [35, 70]], [[35, 60], [60, 70]], [[60, 60], [70, 70]],
        #         [[0, 52], [10, 60]], [[0, 40], [16, 52]], [[16, 40], [32, 52]], [[32, 40], [44, 52]],
        #         [[44, 46], [60, 52]], [[60, 46], [70, 60]], [[0, 26], [16, 40]], [[16, 26], [32, 40]],
        #         [[44, 26], [50, 46]], [[60, 30], [70, 46]], [[0, 16], [16, 26]], [[16, 16], [32, 26]],
        #         [[32, 16], [44, 26]], [[44, 8], [50, 26]], [[60, 14], [70, 30]], [[0, 0], [10, 16]],
        #         [[10, 0], [27, 8]], [[27, 0], [44, 8]], [[44, 0], [56, 8]], [[56, 0], [70, 14]]
        #     ]

        # elif map_num == 3:  # 공학실습동 + 제 2 종합 연구동
        #     # 왼쪽 건물
        #     self.obstacles.append([[12 * scale, 12 * scale], [18 * scale, 12 * scale], [18 * scale, 33 * scale], [12 * scale, 33 * scale]])
        #     self.obstacles.append([[12 * scale, 37 * scale], [18 * scale, 37 * scale], [18 * scale, 58 * scale], [12 * scale, 58 * scale]])
        #     # 중간 건물
        #     self.obstacles.append([[26 * scale, 12 * scale], [32 * scale, 12 * scale], [32 * scale, 33 * scale], [26 * scale, 33 * scale]])
        #     self.obstacles.append([[26 * scale, 37 * scale], [32 * scale, 37 * scale], [32 * scale, 58 * scale], [26 * scale, 58 * scale]])
        #     # 오른쪽 건물
        #     self.obstacles.append([[38 * scale, 12 * scale], [48 * scale, 12 * scale], [48 * scale, 22 * scale], [38 * scale, 22 * scale]])
        #     self.obstacles.append([[38 * scale, 26 * scale], [48 * scale, 26 * scale], [48 * scale, 44 * scale], [38 * scale, 44 * scale]])
        #     self.obstacles.append([[38 * scale, 48 * scale], [48 * scale, 48 * scale], [48 * scale, 58 * scale], [38 * scale, 58 * scale]])
        #     self.obstacles.append([[48 * scale, 12 * scale], [62 * scale, 12 * scale], [62 * scale, 18 * scale], [48 * scale, 18 * scale]])
        #     self.obstacles.append([[48 * scale, 30 * scale], [62 * scale, 30 * scale], [62 * scale, 40 * scale], [48 * scale, 40 * scale]])
        #     self.obstacles.append([[48 * scale, 52 * scale], [62 * scale, 52 * scale], [62 * scale, 58 * scale], [48 * scale, 58 * scale]])

        #     self.spaces_of_map = [
        #         [[0, 58], [12, 70]], [[12, 58], [26, 70]], [[26, 58], [38, 70]], [[38, 58], [62, 70]], [[62, 52], [70, 70]],
        #         [[0, 37], [12, 58]], [[18, 37], [26, 58]], [[32, 37], [38, 58]], [[38, 44], [48, 48]], [[48, 40], [62, 52]],
        #         [[62, 30], [70, 50]], [[0, 33], [12, 37]], [[12, 33], [26, 37]], [[26, 33], [38, 37]],
        #         [[0, 12], [12, 33]], [[18, 12], [26, 33]], [[32, 12], [38, 33]], [[38, 22], [48, 26]], [[48, 18], [62, 30]],
        #         [[62, 12], [70, 30]], [[0, 0], [12, 12]], [[12, 0], [26, 12]], [[26, 0], [38, 12]], [[38, 0], [62, 12]], [[62, 0], [70, 12]]
        #     ]

        # elif map_num == 4:  # 벤젠고리관
        #     # 아래 건물
        #     self.obstacles.append([[48 * scale, 10 * scale], [58 * scale, 20 * scale], [58 * scale, 32 * scale], [44 * scale, 18 * scale]])
        #     self.obstacles.append([[26 * scale, 10 * scale], [44 * scale, 10 * scale], [40 * scale, 18 * scale], [26 * scale, 18 * scale]])
        #     # 중간 건물
        #     self.obstacles.append([[32 * scale, 24 * scale], [50 * scale, 42 * scale], [44 * scale, 48 * scale], [26 * scale, 30 * scale]])
        #     # 윗 건물
        #     self.obstacles.append([[12 * scale, 28 * scale], [20 * scale, 28 * scale], [20 * scale, 42 * scale], [12 * scale, 46 * scale]])
        #     self.obstacles.append([[12 * scale, 50 * scale], [20 * scale, 46 * scale], [32 * scale, 58 * scale], [26 * scale, 64 * scale]])

        #     self.spaces_of_map = [
        #         [[0, 50], [20, 70]], [[20, 58], [32, 70]], [[32, 58], [44, 70]], [[44, 42], [70, 70]],
        #         [[0, 18], [12, 50]], [[12, 42], [20, 50]], [[20, 30], [32, 58]], [[32, 36], [44, 58]],
        #         [[12, 18], [32, 30]], [[32, 18], [44, 36]], [[44, 18], [58, 42]], [[58, 20], [70, 42]],
        #         [[0, 0], [12, 18]], [[12, 0], [32, 18]], [[40, 10], [48, 18]], [[32, 0], [48, 10]], [[48, 0], [70, 20]]
        #     ]

        # elif map_num == 5:  # 경영관 + 퇴계 인문관
        #     # 왼쪽 건물
        #     self.obstacles.append([[18 * scale, 10 * scale], [24 * scale, 10 * scale], [24 * scale, 28 * scale], [18 * scale, 28 * scale]])
        #     self.obstacles.append([[12 * scale, 20 * scale], [18 * scale, 20 * scale], [18 * scale, 26 * scale], [12 * scale, 26 * scale]])
        #     # 오른쪽 건물
        #     self.obstacles.append([[34 * scale, 10 * scale], [46 * scale, 10 * scale], [46 * scale, 16 * scale], [34 * scale, 16 * scale]])
        #     self.obstacles.append([[46 * scale, 10 * scale], [56 * scale, 10 * scale], [56 * scale, 28 * scale], [46 * scale, 28 * scale]])
        #     # 윗 건물
        #     self.obstacles.append([[18 * scale, 34 * scale], [24 * scale, 34 * scale], [24 * scale, 60 * scale], [18 * scale, 60 * scale]])
        #     self.obstacles.append([[24 * scale, 54 * scale], [38 * scale, 54 * scale], [38 * scale, 60 * scale], [24 * scale, 60 * scale]])
        #     self.obstacles.append([[46 * scale, 40 * scale], [52 * scale, 40 * scale], [52 * scale, 48 * scale], [46 * scale, 48 * scale]])
        #     self.obstacles.append([[24 * scale, 34 * scale], [56 * scale, 34 * scale], [56 * scale, 40 * scale], [24 * scale, 40 * scale]])

        #     self.spaces_of_map = [
        #         [[0, 47], [18, 70]], [[18, 60], [38, 70]], [[38, 54], [70, 70]],
        #         [[0, 34], [18, 47]], [[24, 40], [46, 54]], [[46, 40], [70, 54]],
        #         [[0, 20], [18, 34]], [[18, 28], [34, 34]], [[34, 28], [56, 34]],
        #         [[0, 0], [18, 20]], [[24, 10], [34, 28]], [[34, 16], [46, 28]], [[56, 10], [70, 34]],
        #         [[18, 0], [34, 10]], [[34, 0], [56, 10]], [[56, 0], [70, 10]]
        #     ]


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
        for i in self.crowds:
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
            [(0, 0), (exit_width, 0), (exit_width, exit_height), (0, exit_height)],  # 왼아
            [(self.width-exit_width-1, 0), (self.width-1, 0), (self.width-1, exit_height), (self.width-exit_width-1, exit_height)],  # 오아
            [(self.width-exit_width-1, self.height-exit_height-2), (self.width-1, self.height-exit_height-2), (self.width-1, self.height-1), (self.width-exit_width-1, self.height-1)],  # 오위
            [(0, self.height-exit_height-2), (exit_width, self.height-exit_height-2), (exit_width, self.height-1), (0, self.height-1)],  #  왼위
          
        ]
        
        all_exit_points = [
                    [(exit_width)/2, (exit_height)/2],  # 왼쪽 아래
                    [(self.width-exit_width-1+self.width-1)/2, (exit_height)/2],  # 오른쪽 아래
                    [(self.width-exit_width-1+self.width-1)/2, (self.height-exit_height-1+self.height-1)/2],  # 오른쪽 위
                    [(exit_width)/2, (self.height-exit_height-1+self.height-1)/2]  # 왼쪽 위
                ]
        
        # 랜덤하게 출구 선택
        index = random.randint(0, len(all_exits) - 1)
        #print(self.map_num)
        if (self.map_num == 6): #우하단
            index = 1
        elif (self.map_num == 7): #좌하단
            index = 0
        elif (self.map_num == 8) : #우하단
            index = 1
        elif (self.map_num == 1):
            index = 0
        elif (self.map_num == 2):
            index = 1
        elif (self.map_num == 3):
            index = 0
        elif (self.map_num == 4):
            index = 3
        elif (self.map_num == 5):
            index = 2
        elif (self.map_num == 9):
            index = 2
        elif (self.map_num == 10):
            index = 0
        elif (self.map_num == 11):
            index = 0
        elif (self.map_num == 12):
            index = 1
        
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



    def robot_placement(self,
                        n_robots: int = 1,
                        padding: int = 5):
        """
        장애물(삼각형·사각형)과 경계를 피하고,
        맵 외곽에서 padding 만큼 떨어진 안전 영역 안에
        n_robots 개의 로봇을 랜덤 배치한다.
        """
        # 이미 있는 로봇 좌표도 피하도록 집합 유지
        self._occupied_cells = {tuple(a.pos) for a in getattr(self, "robots", [])}

        self.robots = []  # 새로 배치할 로봇 보관
        for _ in range(n_robots):
            spawn = _sample_safe_cell(self, padding)
            self._occupied_cells.add(spawn)

            self.robot = RobotAgent(self.agent_id, self, list(spawn), 3)
            self.agent_id += 10

            self.robots.append(self.robot)
            self.schedule.add(self.robot)
            self.grid.place_agent(self.robot, spawn)

    
    
    def random_agent_distribute_outdoor(self, agent_num, ran):
        

        space_num = len(self.pure_mesh)
        
        
        space_agent = agent_num
        agent_location = []

        for i in range(agent_num):
            assign_mesh_num = random.randint(0, space_num-1)
            assigned_mesh = self.pure_mesh[assign_mesh_num]
            
            while(1):
                assigned_coordinates = self.match_mesh_to_grid[assigned_mesh]
                if (len(assigned_coordinates) !=0):
                    break
                else :
                    assign_mesh_num = random.randint(0, space_num-1)
                    assigned_mesh = self.pure_mesh[assign_mesh_num]
            assigned = assigned_coordinates[random.randint(0, len(assigned_coordinates)-1)]
            assigned = [int(assigned[0]), int(assigned[1])]
            if not assigned in agent_location:
                agent_location.append(assigned)
                a = CrowdAgent(self.agent_num, self, assigned, 1)
                self.crowds.append(a)
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
        # if(started):
        #     for agent in self.agents:
        #         if(agent.type==1 or agent.type==0):
        #             if(agent.unique_id > max_id):
        #                 max_id = agent.unique_id
        #     #self.difficulty_f()
        #     for agent in self.agents:
        #         if(max_id == agent.unique_id):
        #             agent.dead = True
        #     started = 0
        #     max_id = 1
        #     for agent in self.agents:
        #         if (agent.unique_id > max_id and (agent.type==0 or agent.type==1)):
        #             max_id = agent.unique_id
        #     for agent in self.agents:
        #         if(max_id == agent.unique_id):
        #             agent.dead = True 
        self.step_count += 1

        state = self.return_current_image()
        # if(self.using_model):
        #     self.checking_reward += self.reward_based_evacuated_with_robot()
        if(self.using_model and self.step_n%ACTION_SCALE==0):
            if(np.random.rand() < 0.04):
                self.robot.now_exploration = 0
            action, _ = self.dql_agent.select_action(state, True)
            dx, dy = action[0], action[1]
            self.robot.receive_action([dx, dy])

        if(self.using_model and self.step_n%ACTION_SCALE==(ACTION_SCALE-1)):
            print("reward_based_alived : ", self.reward_based_alived() * REWARD_A)
            print("reward_based_all_agents_danger : ", self.reward_based_all_agents_danger() * REWARD_B)
            print("reward_based_gain : ", self.reward_based_gain() * REWARD_C)
            print("reward_penalty : ", self.reward_penalty() * REWARD_D)
            print("reward_based_evacuated_with_robot : ", self.reward_based_evacuated_with_robot() * REWARD_E)
            print("reward_based_distance_from_near_agents : ", self.reward_based_distance_from_near_agents() * REWARD_F)
            print("reward_based_distance_from_near_agent_gain : ", self.reward_based_distance_from_near_agent_gain() * REWARD_G)
            print("reward_based_gain_with_time_bonus :", self.reward_based_gain_with_time_bonus() * REWARD_H)
            print("reward_based_alived_root : ", self.reward_based_alived_root() * REWARD_I)
            print("reward_based_all_agents_danger_log : ", self.reward_based_all_agents_danger_log() * REWARD_J)
            print("reward_penalty_collision : ", self.reward_penalty_collision() * REWARD_K)        
            

        self.schedule.step()
        self.datacollector_currents.collect(self)  # passing the model

        self.previous_evacuated = self.now_evacuated
        self.now_evacuated = self.evacuated_agents()

        self.previous_evacuated_with_robot = self.now_evacuated_with_robot
        self.now_evacuated_with_robot = self.evacuated_agents_with_robot()


        
        
        

    def check_reward(self, reference_reward):
        if self.step_count <= len(reference_reward*100):
            return self.evacuated_agents()-reference_reward[int(self.step_count/100)]
        else :
            return self.evacuated_agents()-self.total_agents
        
    def reward_based_evacuated_timestep_with_robot(self):
        if (self.now_evacuated >= 10 and self.previous_evacuated < 10):
            return (3000-self.step_n)/3000
    
    def reward_based_evacuated_with_robot(self):
        return (self.now_evacuated_with_robot - self.previous_evacuated_with_robot)
    
    def reward_distance_from_all_agents(self):
        reward = 0
        for agent in self.crowds:
            if(agent.type == 0 or agent.type == 1 or agent.type == 2) and (agent.dead == False):
                reward += self.robot.point_to_point_distance(agent.xy, self.robot.xy)
                print(self.robot.point_to_point_distance(agent.xy, self.robot.xy))
        return -reward

    def reward_based_alived(self):
        reward = 0
        num = 0
        
        reward = -self.alived_agents()/self.total_agents
        return reward
    
    def reward_based_all_agents_danger(self):
        
        reward = 0
        for agent in self.crowds:
            if(agent.type == 0 or agent.type == 1 or agent.type == 2) and (agent.dead == False):
                reward += agent.danger
        return -reward
    
    def reward_based_evacuated_confirmed(self):
        reward = 0
        for agent in self.crowds:
            if(agent.type == 0 and agent.is_confirmed == 1 and agent.is_confirmed_past == 0):
                reward += 1

        return reward
    def reward_based_gain(self):
        
        reward=0
        #robot이 agent를 끌어당기면 +reward
        for agent in self.crowds:
            if(agent.type == 0 or agent.type == 1 or agent.type == 2 ) and (agent.dead == False):
                if(agent.robot_tracked>0):
                    reward += agent.gain


        #print("tracked 되고 있는 수 : ", num)
        return reward
    
    def reward_penalty(self):
        reward = 0
        guided_num = 0
        for agent in self.crowds:
            if(agent.type == 0 and agent.dead == False):
                guided_num += 1
        if(guided_num == 0 and self.robot.danger<5):
            return -1
        return 0
    
    def reward_based_distance_from_near_agent_gain(self):
        guided_num = 0
        for agent in self.crowds:
            if(agent.type == 0 and agent.dead == False):
                guided_num += 1
        
        if (guided_num > 0):
            self.before_minimum_distance = 0
            return 0 
        
        minimum_distance = 999999
        for agent in self.crowds:
            if(agent.type == 0 or agent.type ==1 or agent.type == 2) and (agent.dead == False):
                distance = self.robot.point_to_point_distance(self.robot.xy, agent.xy)
                if(distance < minimum_distance):
                    minimum_distance = distance
        
        self.before_minimum_distance = self.minimum_distance
        self.minimum_distance = minimum_distance

        if(self.before_minimum_distance == 0 or self.minimum_distance == 0):
            return 0
        
        return (self.before_minimum_distance - self.minimum_distance)
        
    def reward_based_gain_with_time_bonus(self):
        reward=0
        #robot이 agent를 끌어당기면 +reward
        for agent in self.crowds:
            if(agent.type == 0 or agent.type == 1 or agent.type == 2 ) and (agent.dead == False):
                if(agent.robot_tracked>0):
                    reward += agent.gain

        if (self.alived_agents()<self.total_agents*0.4):
            reward = reward * 2

        #print("tracked 되고 있는 수 : ", num)
        return reward
    
    def reward_based_alived_root(self):
        reward = 0
        num = 0
        reward = -math.sqrt(self.alived_agents()/self.total_agents)

        return reward
    



    def reward_evacuation(self):
        if(self.step_n<3):
            return 0
        return -self.robot.danger/100
        

    def return_agent_id(self, agent_id):
        for agent in self.agents:
            if(agent.unique_id == agent_id):
                return agent
        return None
    
    def use_model(self, file_path):
        from ADDS_AS import USING_TRAINED_MODEL
        input_shape = (50, 50)
        num_actions = 4

        self.dql_agent = DiffusionQLAgent(input_shape, num_actions, start_epsilon=0)
        if (USING_TRAINED_MODEL):
            self.dql_agent.load_model(file_path)
        self.using_model = True

    
    def reward_based_distance_from_near_agents(self):
        guided_num = 0
        for agent in self.crowds:
            if(agent.type == 0 and agent.dead == False):
                guided_num += 1
        
        if (guided_num > 0):
            return 0 
        
        minimum_distance = 999999
        for agent in self.crowds:
            if(agent.type == 0 or agent.type ==1 or agent.type == 2) and (agent.dead == False):
                distance = self.robot.point_to_point_distance(self.robot.xy, agent.xy)
                if(distance < minimum_distance):
                    minimum_distance = distance
        if(minimum_distance == 999999):
            return 0
        if(minimum_distance < 8):
            return 0
        return -minimum_distance
    
    def reward_based_all_agents_danger_log(self):
        reward = 0
        for agent in self.crowds:
            if(agent.type == 0 or agent.type == 1 or agent.type == 2) and (agent.dead == False):
                reward += agent.danger
        return -math.log(reward+1)
        

    def reward_based_new_founded_agent_danger(self):
        reward = self.new_founded_agent_danger
        self.new_founded_agent_danger = 0
        return reward
    
    def reward_based_near_agents_exist(self):
        
        for agent in self.crowds:
            if(agent.dead == False):
                distance = self.robot.point_to_point_distance(self.robot.xy, agent.xy)
                if (distance<20):
                    return 0
        return -2

    def reward_penalty_collision(self):
        if self.robot.collision_check :
            return -1
        else :
            return 0
    
    def return_current_image(self):
        # Create a 2D NumPy array with zeros of type uint8
        # shape: (height, width)
        image = np.zeros((self.width, self.height), dtype=np.uint8)
        
        # Fill in wall
        for agent in self.agents:
            if agent.type == 9:
                image[agent.pos[0], agent.pos[1]] = 20  # 벽 #50?
        
        # Fill in exit
        for agent in self.agents:
            if agent.type == 10:
                image[agent.pos[0], agent.pos[1]] = 60  # 출구 #100?

        # Fill crowd agents
        for agent in self.crowds:
            if (agent.type == 1 or agent.type == 2) and not agent.dead:
                image[int(round(agent.xy[0])), int(round(agent.xy[1]))] = 100  # agent #150?
        for agent in self.crowds:
            if agent.type == 0 and not agent.dead:
                image[int(round(agent.xy[0])), int(round(agent.xy[1]))] = 140 #200?
        
        # Fill robot
        for agent in self.agents:
            if agent.type == 3:
                image[int(round(agent.xy[0])), int(round(agent.xy[1]))] = 200  # robot #255?
        
        return image
    
    def choice_random_waypoint(self):
        return [random.randint(0, self.width-1), random.randint(0, self.height-1)]

    
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
