from agent import RobotAgent
from agent import CrowdAgent

from core import RandomActivation, DataCollector
from space import ContinuousSpace

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.strtree import STRtree
from shapely.ops import triangulate
import matplotlib.tri as mtri

from core import Model, Agent
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
from typing import List, Tuple
from visibility_atlas import VisibilityAtlas
#import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ADDS_AS_reinforcement import SACAgent, ReplayBuffer, PolicyNetwork, QNetwork, ACTION_SCALE, FrameStack
from config import *


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

# 안전 스폰 헬퍼(연속 좌표). 기존 _sample_safe_cell 대체/신규
def _sample_safe_pos(self, padding: float = 1.0, max_attempts: int = 2000) -> Tuple[float, float]:
    xmin, ymin = padding, padding
    xmax, ymax = self.width - padding, self.height - padding
    for _ in range(max_attempts):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        p = Point(x, y)
        # 장애물/경계 충돌 금지
        if any(Polygon(poly).contains(p) or Polygon(poly).touches(p) for poly in self.obstacles):
            continue
        # 이미 사용된 위치 근접 금지(로봇/군중 최소 거리)
        if any(math.hypot(x - ax, y - ay) < 0.5 for (ax, ay) in getattr(self, "_occupied_positions", [])):
            continue
        return (x, y)
    raise ValueError("안전 스폰 위치를 찾지 못했습니다(continuous).")



def are_meshes_adjacent(mesh1, mesh2):
    # 두 mesh의 공통 꼭짓점의 개수를 센다
    common_vertices = set(mesh1) & set(mesh2)
    return len(common_vertices) >= 2  # 공통 꼭짓점이 두 개 이상일 때 인접하다고 판단R

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

def normalize_map_to_50(obs, target=50):
    x = torch.from_numpy(obs).float()
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        y = F.adaptive_max_pool2d(x, (target, target))
        return y.squeeze().numpy()
    else:  # (C,H,W)
        x = x.unsqueeze(0)  # (1,C,H,W)
        y = F.adaptive_max_pool2d(x, (target, target))
        return y.squeeze(0).numpy()
 


class FightingModel(Model):
    """A model with some number of agents."""

    def __init__(self, number_agents: int, width: int, height: int, model_num = -1, robot = 'Q'):
        #print("model_num :", model_num)
        super().__init__()
        self.width = width
        self.height = height      
        self.space = ContinuousSpace(self.width, self.height, cell_size=10.0, torus=False)  
        self.frame_stack = FrameStack(stack_len=4)
        self._first_step = True
        self.robot_version = robot
        
        self.crowds = []
        self.step_n = 0

        self.robot_type = robot
        self.spaces_of_map = []
        self.map_num = model_num # 1 : 산학협력관 + 잔디밭 / 2 : 제2 공학관 + 정원 / 3 : 공학실습동 + 제2 연구동 / 4 : 벤젠고리관 / 5 : 경영관 + 퇴계 인문관
        self.running = (
            True  
        )
        
        self.agent_id = 1000
        self.agent_num = 0


        self.using_model = False
        self.total_agents = number_agents
        self.obstacle_mesh = []
        self.adjacent_mesh = {}
        # map_ran_num = 2
        self.walls = list()
        self.obstacles = list()
        self.mesh = list()
        self.mesh_list = list()
        if (self.map_num == -1):
            if(MAP_NUM != -2 and MAP_NUM != -1):
                self.extract_map(self.map_num)
            if (MAP_NUM == -1):
                map_num_candidates = MAP_NUM_RANDOM
                self.map_num = random.choice(map_num_candidates)
                self.extract_map(self.map_num)    
        else :
            self.extract_map(self.map_num)

        self.distance = {}  
        self.schedule = RandomActivation(self)
        self.running = (
            True
        )
        self.next_vertex_matrix = {}
        self.pure_mesh = []
        self.mesh_danger = {}
        self.match_grid_to_mesh = {}
        self.match_mesh_to_grid = {}
        self.valid_space = {}
        self.blocked = np.zeros((self.height, self.width), dtype=bool)
        self.obstacles_grid_points = []
        self.fill_outwalls(width, height)
        self.mesh_map()
        self.make_random_exit()
        self.construct_map()
        self.calculate_mesh_danger()
        self.random_agent_distribute_outdoor(number_agents, 1)
        if (self.robot_version != 'N'):
            self.make_robot()
        self.step_count = 0

        self.now_evacuated = 0
        self.now_evacuated_with_robot = 0

        self.previous_evacuated = 0
        self.previous_evacuated_with_robot = 0

        self.before_minimum_distance = 0
        self.minimum_distance = 0
        self.new_founded_agent_danger = 0

            
        self._obstacle_polys = [Polygon(ob) for ob in self.obstacles]
        #print(self._obstacle_polys)
        self._exit_polys = [Polygon(poly) for poly in self.exit_list]
        self._mesh_polys = [Polygon(t) for t in self.mesh_list]
        self._mesh_poly2tri = {poly: tri for poly, tri in zip(self._mesh_polys, self.mesh_list)}
        self._mesh_index = STRtree(self._mesh_polys)

        self._build_blocked_grid()

        self.obstacles_version = 0
        self.vision_atlas = VisibilityAtlas(world_w=width, world_h=height, region_cells=4.0)
        
        SENSOR_R_AGENT = AGENT_VISION
        SENSOR_R_ROBOT = ROBOT_VISION  
        self.vision_atlas.rebuild_obstacles(self._obstacle_polys, self.obstacles_version)
        self.vision_atlas.set_radii([AGENT_VISION])
        self.vision_atlas.precompute(rays_per_poly=64, bsearch_iters=6)

        #self.shadow_fov = ShadowFOV(self.blocked)


        self.exit_meta = [
        {"idx":i,
         "center": tuple(self.exit_point[i]),
         "width" : 5,             # exit_width = 5
         } for i in range(len(self.exit_point))
        ]

        
        

        self.stats = DataCollector(
            model_reporters = {
                "Step"           : lambda m: m.step_n,
                "Evacuated"      : lambda m: m.evacuated_agents(),
                "EvacWithRobot"  : lambda m: m.evacuated_agents_with_robot(),
                "AvgDanger"      : lambda m: np.mean([ag.danger for ag in m.crowds if not ag.dead]),
            },
            agent_reporters = {
                "x"      : lambda a: a.xy[0],
                "y"      : lambda a: a.xy[1],
                "speed"  : lambda a: np.linalg.norm(a.vel),
                "type"   : "type",
                "danger" : "danger",
            }
        )

        self.static_grid = np.zeros((self.height, self.width), dtype = np.uint8)
        self._render_static_map(self.height, self.width)

    def is_free(self, xy):
        x, y = xy
        if x < 0 or y < 0 or x > self.width or y > self.height:
            return False
        p = Point(x, y)
        return not any(poly.contains(p) or poly.touches(p) for poly in self._obstacle_polys)

    def in_exit(self, xy, tol=0.0):
        p = Point(xy[0], xy[1])
        for poly in self._exit_polys:
            #print(f"{p}가 {poly}에 있는지 확인합니다")    
            if (poly.buffer(tol).contains(p)) if tol > 0 else poly.contains(p):
                return True
        return False

    def find_mesh(self, xy):
        p = Point(float(xy[0]), float(xy[1]))

        hits = self._mesh_index.query(p)  # Shapely 2.x: indices (np.array), 1.8: list of geometries
        if hits is None:
            return None

        # 호환 처리: 결과가 정수(인덱스)인지, geometry인지 구분
        def _is_index(x):
            try:
                import numpy as _np
                return isinstance(x, (int, _np.integer))
            except Exception:
                return isinstance(x, int)

        if isinstance(hits, (list, tuple)) and hits and not _is_index(hits[0]):
            # Shapely 1.8 스타일: geometry 리스트
            for poly in hits:
                # 경계도 포함하려면 covers 권장 (contains는 경계 제외)
                if poly.covers(p):
                    # poly -> tri 매핑
                    return self._mesh_poly2tri[poly]
        else:
            # Shapely 2.x 스타일: 인덱스 배열/리스트
            hits = np.atleast_1d(hits)
            for i in hits:
                i = int(i)
                poly = self._mesh_polys[i]
                if poly.covers(p):
                    # poly를 키로 쓰는 게 불안하면, 인덱스 기반 매핑을 따로 유지하세요.
                    return self._mesh_poly2tri[poly]

        return None

    def next_mesh_from_to(self, m_from, m_to):
        return self.next_vertex_matrix.get(m_from, {}).get(m_to, None)



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

        
        boundary_coords = []
        boundary_coords = list(set(map(tuple, boundary_coords)))

        for i in range(self.width):
            for j in range(self.height):
                for mesh in self.pure_mesh:
                    if is_point_in_triangle([i, j], mesh[0], mesh[1], mesh[2]):
                        if mesh not in self.match_mesh_to_grid.keys():
                            self.match_mesh_to_grid[mesh] = []
                        self.match_mesh_to_grid[mesh].append([i, j])
        for i in range(-10, self.width + 10):      # self.width가 50이라고 가정
            for j in range(-10, self.height + 10):   # self.height가 50이라고 가정
                if 0 <= i < self.width and 0 <= j < self.height:
                    self.valid_space[(i, j)] = 1
                else:
                    self.valid_space[(i, j)] = 0

    def get_path(self, next_vertex_matrix, start, end): #start->end까지 최단 경로로 가려면 어떻게 가야하는지 알려줌 

        if next_vertex_matrix[start][end] is None:
            return []

        path = [start]
        while start != end:
            start = next_vertex_matrix[start][end]
            path.append(start)
        return path

    def extract_map(self, map_num):

        #좌하단 #우하단 #우상단 #좌상단 순으로 입력해주기


        if map_num == 6:
            self.obstacles.append([[10, 10], [20, 10], [20, 20], [10, 20]])
            self.obstacles.append([[10, 30], [40, 30], [40, 40], [10, 40]])
            
            
            self.obstacles.append([[60, 60], [70, 60], [70, 70], [60, 70]])
            self.obstacles.append([[10, 80], [40, 80], [40, 90], [10, 90]])
            self.obstacles.append([[60, 30], [90, 30], [90, 40], [60, 40]])



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

        elif map_num == 26:
            self.obstacles.append([[20, 15], [35, 30], [30, 35], [15, 20]])
            self.obstacles.append([[20, 0], [30, 0], [30, 10]])
            self.obstacles.append([[10, 40], [20, 40], [20, 35], [10, 35]])

        
        elif map_num == 100:
            #self.obstacles.append([[20, 10], [30, 10], [30, 20]])
            self.obstacles.append([[10, 30], [15, 30], [15, 40], [10, 40]])
            self.obstacles.append([[10, 50], [20, 50], [20, 60], [10, 60]])
            self.obstacles.append([[30, 50], [50, 50], [50, 60], [30, 60]])
            #self.obstacles.append([[20, 50], [50, 50], [50, 60], [20, 60]])
            self.obstacles.append([[15, 80], [30, 80], [30, 90], [15, 90]])
            self.obstacles.append([[50, 80], [60, 80], [60, 90], [50, 90]])
            #self.obstacles.append([[70, 50], [80, 50], [80, 60], [70, 60]])
            self.obstacles.append([[85, 55], [100, 55], [100, 60], [85, 60]])
            #self.obstacles.append([[70, 70], [80, 70], [80, 80], [70, 80]])
            self.obstacles.append([[70, 70], [75, 70],[75, 80], [70, 80]])
            self.obstacles.append([[25, 20], [30, 20], [30, 30], [25, 25]])
            self.obstacles.append([[5, 15], [10, 15], [10,20], [5, 20]])
            #self.obstacles.append([[15, 10], [25, 10], [25, 15], [20, 15]])
            self.obstacles.append([[85, 85], [90, 85], [90, 90], [85, 90]])
            self.obstacles.append([[60, 30], [80, 30], [80, 40], [60, 40]])
            #self.obstacles.append([[85, 30], [90, 30], [90, 40], [85, 40]])
            #self.obstacles.append([[60, 10], [70, 10], [70, 20], [60, 20]])
            self.obstacles.append([[80, 10], [90, 10], [90, 20], [80, 20]])
            self.obstacles.append([[30, 0], [50, 0], [50, 10], [30, 10]])
            self.obstacles.append([[40, 25], [50, 25], [50, 30], [40, 30]])
            #self.obstacles.append([[50, 10], [60, 10], [50, 20]])
            #self.obstacles.append([[90, 30], [90, 40], [80, 40]])
            #self.obstacles.append([[50, 30], [60, 40], [50, 40]])
            #self.obstacles.append([[80, 10], [90, 10], [90, 20]])

        elif map_num == 101:
            self.obstacles.append([[20, 5], [30, 5], [30, 15], [20, 15]])
            #self.obstacles.append([[10, 20], [30, 20], [30, 30], [10, 30]])
            self.obstacles.append([[20, 90], [30, 90], [30, 100], [20, 100]])
            self.obstacles.append([[10, 45], [20, 45], [20, 55], [10, 55]])
            self.obstacles.append([[20, 65], [30, 65], [30, 75], [20, 75]])
            
            self.obstacles.append([[0, 10], [10, 10], [10, 30], [0, 30]])
            self.obstacles.append([[40, 60], [60, 60], [60, 70], [40, 70]])
            self.obstacles.append([[40, 30], [60, 30], [60, 40], [40, 40]])
            self.obstacles.append([[70, 5], [80, 5], [80, 15], [70, 15]])
            self.obstacles.append([[0, 10], [10, 10], [10, 30], [0, 30]])
            self.obstacles.append([[80, 30], [100, 30], [100, 50], [80, 50]])

            #self.obstacles.append([[70, 65], [80, 65], [80, 75], [70, 75]])
            #self.obstacles.append([[70, 80], [90, 80], [90, 90], [70, 90]])
    
        elif map_num == 102:
            self.obstacles.append([[10, 10], [20, 10], [20, 20], [10, 20]])
            
            self.obstacles.append([[30, 20], [40, 40], [30, 45], [10, 40]])
            self.obstacles.append([[30, 55], [40, 60], [30, 80], [10, 60]])
            
            self.obstacles.append([[10, 80], [20, 80], [20, 90], [10, 90]])
            #self.obstacles.append([[45, 60], [55, 60], [60, 80], [40, 80]])
            #self.obstacles.append([[40, 20], [60, 20], [55, 40], [45, 40]])
            self.obstacles.append([[80, 10], [90, 10], [90, 20], [80, 20]])
            
            self.obstacles.append([[70, 20], [90, 40], [70, 45], [60, 40]])
            self.obstacles.append([[70, 55], [90, 60], [70, 80], [60, 60]])
            
            self.obstacles.append([[80, 80], [90, 80], [90, 90], [80, 90]])
        
        elif map_num == 103:
            self.obstacles.append([[10, 20], [20, 20], [20, 30], [10, 30]])
            self.obstacles.append([[40, 10], [50, 10], [50, 20], [40, 20]])
            self.obstacles.append([[60, 0], [70, 0], [70, 10], [60, 10]])
            self.obstacles.append([[70, 20], [80, 20], [80, 30], [70, 30]])
            #self.obstacles.append([[40, 0], [70, 0], [70, 30], [40, 30]])
            #self.obstacles.append([[50, 50], [60, 50], [60, 60], [50, 60]])
            #elf.obstacles.append([[0, 50], [30, 50], [30, 60], [0, 90]])
            self.obstacles.append([[60, 70], [100, 70], [100, 100], [30, 100]])

        elif map_num == 104:
            #self.obstacles.append([[20, 0], [40, 0], [40, 10], [30, 10]])
            self.obstacles.append([[0, 20], [20, 20], [20, 50], [0, 70]])
            #self.obstacles.append([[20, 50], [30, 50], [30, 60], [20, 60]])
            self.obstacles.append([[50, 20], [60, 20], [60, 30], [50, 30]])
            self.obstacles.append([[50, 40], [50, 50], [60, 50], [60, 40]])
            self.obstacles.append([[80, 20], [90, 20], [90, 30], [80, 30]])
            self.obstacles.append([[80, 40], [90, 40], [90, 50]])
            self.obstacles.append([[10, 70], [15, 70], [15, 80], [10, 80]])
            self.obstacles.append([[40, 70], [100, 70], [100, 100], [10, 100]])

        elif map_num == 105:
 
            self.obstacles.append([[30, 20], [35, 25], [20, 60], [10, 50]])
            self.obstacles.append([[40, 30], [50, 40], [30, 70], [25, 65]])
            self.obstacles.append([[10, 80], [15, 80], [15, 90], [10, 90]])
            self.obstacles.append([[10, 10], [20, 10], [20, 20], [10, 20]])
            self.obstacles.append([[25, 90], [40, 90], [40, 95], [25, 95]])
            self.obstacles.append([[50, 70], [55, 70], [55, 80], [50, 80]])
            self.obstacles.append([[65, 40], [70, 40], [70, 50], [65, 50]])
            self.obstacles.append([[70, 30], [90, 30], [90, 35], [70, 35]])
            self.obstacles.append([[80, 5], [85, 5], [85, 10], [80, 10]])
            self.obstacles.append([[90, 15], [95, 15], [95, 20], [90, 20]])
            self.obstacles.append([[50, 0], [60, 0], [60, 30], [50, 20]])
            self.obstacles.append([[100, 50], [100, 100], [70, 100], [70, 80]])

        elif map_num == 106:
            self.obstacles.append([[45, 25], [55, 35], [35, 65], [20, 50]])
            self.obstacles.append([[60, 40], [75, 55], [50, 80], [40, 70]])
            self.obstacles.append([[20, 70], [30, 80], [20, 90], [10, 80]])
            self.obstacles.append([[70, 80], [80, 80], [80, 90], [70, 90]])
            self.obstacles.append([[90, 60], [90, 65], [85, 65], [85, 60]])
            self.obstacles.append([[80, 20], [100, 20], [100, 50], [80, 30]])
            self.obstacles.append([[0, 30], [10, 30], [10, 40], [0, 40]])
            self.obstacles.append([[60, 0], [60, 20], [50, 10], [50, 0]])

        elif map_num == 107:
            self.obstacles.append([[0, 0], [30, 0], [20, 30], [0, 30]])
            self.obstacles.append([[80, 50], [100, 50], [100, 100], [80, 100]])
            self.obstacles.append([[40, 0], [50, 0], [50, 10], [40, 10]])
            self.obstacles.append([[80, 10], [90, 10], [90, 20], [80, 20]])
            #self.obstacles.append([[40, 20], [50, 20], [50, 30], [40, 30]])
            self.obstacles.append([[20, 40], [40, 40], [40, 50], [20, 50]])
            #self.obstacles.append([[10, 60], [20, 60], [20, 70], [10, 70]])
            self.obstacles.append([[10, 80], [15, 80], [15, 85], [10, 90]])
            #self.obstacles.append([[25, 85], [30, 85], [30, 90], [25, 90]])
            #self.obstacles.append([[45, 90], [60, 90], [60, 95], [45, 95]])
            #self.obstacles.append([[55, 60], [60, 60], [60, 70], [55, 70]])
        
        elif map_num == 108:
            #self.obstacles.append([[10, 10], [15, 10], [15, 20], [10, 20]])
            #self.obstacles.append([[25, 15], [30, 15], [30, 20], [25, 20]])

            self.obstacles.append([[0, 25], [10, 25], [10, 30], [0, 30]])
            self.obstacles.append([[25, 10], [30, 10], [30, 15], [25, 15]])
            self.obstacles.append([[20, 35], [30, 35], [30, 40], [20, 40]])
            self.obstacles.append([[10, 50], [20, 50], [20, 70], [10, 70]])
            self.obstacles.append([[10, 80], [20, 80], [20, 90], [10, 90]])
            self.obstacles.append([[40, 70], [50, 70], [50, 90], [40, 90]])
            self.obstacles.append([[60, 70], [80, 90], [60, 90]])
            self.obstacles.append([[70, 60], [90, 60], [90, 80]])
            self.obstacles.append([[50, 40], [90, 40], [90, 50], [60, 50]])
            self.obstacles.append([[75, 25], [80, 25], [80, 30], [75, 30]])
            self.obstacles.append([[85, 10], [90, 10], [90, 15], [85, 15]])
            self.obstacles.append([[40, 50], [50, 60], [40, 60]])
            self.obstacles.append([[50, 10], [60, 10], [60, 20], [50, 20]])

            
        elif map_num == 50:
            self.obstacles.append([[20, 0], [30, 0], [30, 15], [20, 15]])
            self.obstacles.append([[0, 30], [20, 30], [20, 50], [0, 50]])
            self.obstacles.append([[30, 30], [40, 30], [40, 40], [30, 40]])
            self.obstacles.append([[30, 50], [40, 50], [60, 70], [30, 70]])
            self.obstacles.append([[50, 0], [70, 0], [70, 50], [50, 30]])

        elif map_num == 51:
            #self.obstacles.append([[0, 0], [20, 0], [20, 10], [0, 30]])
            #self.obstacles.append([[40, 5], [50, 10], [40, 20], [30, 15]])
            self.obstacles.append([[30, 10], [40, 10], [40, 20], [30, 20]])
            self.obstacles.append([[20, 30], [30, 40], [20, 50], [10, 40]])
            self.obstacles.append([[50, 30], [60, 30], [60, 50], [50, 50]])
            self.obstacles.append([[40, 60], [50, 60], [50, 70], [40, 70]])
        

        elif map_num == 53:
            self.obstacles.append([[20, 5], [30, 5], [30, 15], [20, 15]])
            self.obstacles.append([[10, 20], [30, 20], [30, 30], [10, 30]])
            self.obstacles.append([[10, 45], [20, 45], [20, 55], [10, 55]])
            self.obstacles.append([[20, 65], [30, 65], [30, 75], [20, 75]])
            self.obstacles.append([[10, 80], [30, 80], [30, 90], [10, 90]])
            self.obstacles.append([[40, 60], [60, 60], [60, 70], [40, 70]])
            self.obstacles.append([[40, 30], [60, 30], [60, 40], [40, 40]])
            #self.obstacles.append([[40, 10], [60, 10], [60, 20], [40, 20]])
            self.obstacles.append([[70, 5], [80, 5], [80, 15], [70, 15]])
            self.obstacles.append([[70, 20], [90, 20], [90, 30], [70, 30]])
            self.obstacles.append([[80, 45], [90, 45], [90, 55], [80, 55]])
            self.obstacles.append([[70, 65], [80, 65], [80, 75], [70, 75]])
            self.obstacles.append([[70, 80], [90, 80], [90, 90], [70, 90]])

        elif map_num == 54:
            self.obstacles.append([[20, 0], [30, 0], [30, 5], [20, 5]])
            self.obstacles.append([[0, 10], [10, 10], [10, 20], [0, 20]])
            self.obstacles.append([[30, 20], [35, 20], [35, 30], [30, 30]])
            self.obstacles.append([[10, 35], [30, 35], [30, 40], [10, 40]])
            self.obstacles.append([[10, 60], [30, 60], [30, 70], [10, 70]])
            self.obstacles.append([[55, 45], [65, 55], [60, 60], [50, 50]])
            self.obstacles.append([[55, 25], [70, 25], [70, 30], [55, 30]])
            self.obstacles.append([[60, 10], [65, 10], [65, 15], [60, 15]])
            self.obstacles.append([[45, 35], [50, 40], [45, 45], [40, 40]])
            self.obstacles.append([[25, 45], [30, 45], [30, 55], [25, 55]])
            self.obstacles.append([[40, 10], [45, 10], [45, 15], [40, 15]])
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    def construct_map(self):
        for i in range(len(self.obstacles)):
            for each_point in  get_points_within_polygon(self.obstacles[i], 1):
                self.obstacles_grid_points.append(each_point)
                a = WallAgent(self.agent_num, self, each_point, 9)
                self.agent_num+=1
                #self.schedule_e.add(a)
                self.valid_space[(each_point[0], each_point[1]-1)] = 0
                self.valid_space[(each_point[0], each_point[1])] = 0
                #self.grid.place_agent(a, each_point)

    def is_free_point(self, x: float, y: float, padding: float = 0.0) -> bool:
        if not (0+padding <= x <= self.width-padding and 0+padding <= y <= self.height-padding):
            return False
        p = Point(x, y)
        for poly in self.obstacles:
            P = Polygon(poly)
            if P.contains(p) or P.touches(p):
                return False
        return True


                                  
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
        # self.exit_list = [[(0,0), (exit_width, 0), (exit_width, exit_height), (0, exit_height)],
        #                  [(self.width-exit_width-1,0), (self.width-1, 0), (self.width-1, exit_height), (self.width-exit_width-1, exit_height)],
        #                  [(0, self.height-exit_height-2), (exit_width, self.height-exit_height-2), (exit_width, self.height-1), (0, self.height-1)],
        #                  [(self.width-exit_width-1, self.height-exit_height-2), (self.width-1, self.height-exit_height-2), (self.width-1, self.height-1), (self.width-exit_width-1, self.height-1)]
        #                 ]
        # self.exit_list = [[(0,0), (exit_width, 0), (exit_width, exit_height), (0, exit_height)],
        #                   [(self.width-exit_width-1,0), (self.width-1, 0), (self.width-1, exit_height), (self.width-exit_width-1, exit_height)],
        #                   [(0, self.height-exit_height-1), (exit_width, self.height-exit_height-2), (exit_width, self.height-1), (0, self.height-1)],
        #                   [(self.width-exit_width-1, self.height-exit_height-2), (self.width-1, self.height-exit_height-2), (self.width-1, self.height-1), (self.width-exit_width-1, self.height-1)]
        #                  ]
        # self.exit_point = [[(exit_width)/2, (exit_height)/2],
        #                    [(self.width-exit_width-1+self.width-1)/2, (exit_height)/2],
        #                    [(exit_width)/2, (self.height-exit_height-1+self.height-1)/2],
        #                    [(self.width-exit_width-1+self.width-1)/2, (self.height-exit_height-1+self.height-1)/2]
        #                    ]
        
        
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
            (exit_width/2,                 exit_height/2                ),  # 왼아  0
            (self.width - exit_width/2,    exit_height/2                ),  # 오아  1
            (self.width - exit_width/2,    self.height - exit_height/2  ),  # 오위  2
            (exit_width/2,                 self.height - exit_height/2  ),  # 왼위  3
        ]
        
        # 랜덤하게 출구 선택

        index = []
        #print(self.map_num)
        if (self.map_num == 6): #우하단
            index = [1]
        elif (self.map_num == 7): #좌하단
            index = [0]
        elif (self.map_num == 8) : #우하단
            index = [1]
        elif (self.map_num == 26) :
            index = [1]
        elif (self.map_num == 100):
            index = [0, 2]
        elif (self.map_num == 101):
            index = [1, 3]
        elif (self.map_num == 102):
            index = [0, 2]
        elif (self.map_num == 103):
            index = [1, 3]
        elif (self.map_num == 104):
            index = [1, 3]
        elif (self.map_num == 105):
            index = [1, 3]
        elif (self.map_num == 106):
            index = [0, 1]
        elif (self.map_num == 108):
            index = [0, 2]
        elif (self.map_num == 50):
            index = [2]
        elif (self.map_num == 51):
            index = [3]

        elif (self.map_num == 53):
            index = [1]
        elif (self.map_num == 54):
            index = [1]
        
        self.exit_point = []
        self.exit_list = []
        for i in range(len(index)):
            self.exit_list.append(all_exits[index[i]])

        for i in range(len(index)):
            self.exit_point.append(all_exit_points[index[i]])    

        
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


    def robot_placement(self, n_robots: int = 1, padding: float = 1.0):
        self._occupied_positions = [(getattr(a, "xy", a.pos)[0], getattr(a, "xy", a.pos)[1])
                                    for a in getattr(self, "robots", [])]
        self.robots = []
        for _ in range(n_robots):
            x, y = _sample_safe_pos(self, padding=padding)
            self._occupied_positions.append((x, y))

            self.robot = RobotAgent(self.agent_id, self, [x, y], 3)
            self.agent_id += 10
            self.robots.append(self.robot)
            self.schedule.add(self.robot)
            # ✨ 연속 공간에 배치
            self.space.place_agent(self.robot, (x, y))
            self.agents.append(self.robot)
    

    # 군중 배치 교체 (연속 좌표 사용)
    def random_agent_distribute_outdoor(self, agent_num: int, ran: int, padding: float = 1.0):
        self._occupied_positions = getattr(self, "_occupied_positions", [])
        for _ in range(agent_num):
            while True:
                # 메시 기반 선택은 유지하되 최종 좌표는 연속 무작위(삼각형 내부 샘플)로
                mesh = random.choice(self.pure_mesh)
                tri = Polygon([mesh[0], mesh[1], mesh[2]])
                minx, miny, maxx, maxy = tri.bounds
                for _try in range(200):
                    x = random.uniform(minx, maxx)
                    y = random.uniform(miny, maxy)
                    if not tri.contains(Point(x, y)):  # 삼각형 내부만
                        continue
                    if not self.is_free_point(x, y, padding=padding):
                        continue
                    if any(math.hypot(x-ax, y-ay) < 0.5 for (ax, ay) in self._occupied_positions):
                        continue
                    break
                else:
                    # 삼각형 샘플 실패 → 다른 mesh
                    continue
                break

            a = CrowdAgent(self.agent_num, self, [x, y], 1)
            self.crowds.append(a)
            self.agent_num += 1
            self.schedule.add(a)
            self.space.place_agent(a, (x, y))
            self.agents.append(a)
            self._occupied_positions.append((x, y))



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
    
    def exit_score(self, agent, exit_idx, alpha):
        
        """식 (12)를 그대로 구현 – distance·density·width 3요소."""
        # (i) 거리
        d_s  = agent.point_to_point_distance(agent.xy, self.exit_point[exit_idx])

        # (ii) 밀도 = 동일 출구를 향해 ‘현재’ 이동 중인 인원 수
        people_to_exit = sum(
                            bool(ag.exit_belief and ag.exit_belief["idx"] == exit_idx)
                            for ag in self.crowds if not ag.dead
                        )
        d_e = people_to_exit

        # (iii) 폭 – 이미 exit_meta에 내장됨
        d_w = self.exit_meta[exit_idx]["width"]
        base = (K1*np.exp(-d_s) + K2*np.exp(-d_e) + K3*(1-np.exp(-d_w)))
        score = np.exp(-alpha) * base/(K1+K2+K3)
        return score
        


    def step(self):
        self.stats.collect(self)        # 매 스텝 자동 누적
        self.step_n += 1
        """Advance the model by one step."""
        global started
        max_id = 1

        self.step_count += 1

        #state = self.return_current_image()
        #print(self.return_current_robot_state())
        raw_frame = self.return_current_image(self.height, self.width)
        frame = normalize_map_to_50(raw_frame)
        frame = frame.astype(np.float32) / 255.0

        state = None

        is_boundary = ((self.step_n - 1) % ACTION_SCALE == 0)

        if self._first_step:
            state = self.frame_stack.reset(frame)
            self._first_step = False

        elif is_boundary:
            state = self.frame_stack.append(frame)

        if(self.robot_version == 'Q'):
            
            if self.using_model and ((self.step_n-1) % ACTION_SCALE == 0):
                
                if state is None:
                    state = self.frame_stack.peek_with(frame)

                robot_state = np.array(self.return_current_robot_state(), dtype=np.float32)

                action, _ = self.sac_agent.select_action(state, robot_state, deterministic=True)
                
                dx, dy = action[0], action[1]
                self.robot.receive_action([dx, dy])

            if(self.using_model and self.step_n%ACTION_SCALE==(ACTION_SCALE-1) or SCALE_CHECK):
                print("reward_based_alived : ", self.reward_based_alived() * REWARD_A)
                print("reward_based_all_agents_danger : ", self.reward_based_all_agents_danger() * REWARD_B)
                #print("reward_based_gain : ", self.reward_based_gain() * REWARD_C)
                print("reward_penalty : ", self.reward_penalty() * REWARD_D)
                print("reward_fixed : ", REWARD_FIXED)
                #print("reward_based_evacuated_with_robot : ", self.reward_based_evacuated_with_robot() * REWARD_E)
                #print("reward_based_distance_from_near_agents : ", self.reward_based_distance_from_near_agents() * REWARD_F)
                #print("reward_based_distance_from_near_agent_gain : ", self.reward_based_distance_from_near_agent_gain() * REWARD_G)
                #print("reward_based_gain_with_time_bonus :", self.reward_based_gain_with_time_bonus() * REWARD_H)
                #print("reward_based_alived_root : ", self.reward_based_alived_root() * REWARD_I)
                #print("reward_based_all_agents_danger_log : ", self.reward_based_all_agents_danger_log() * REWARD_J)
                #print("reward_penalty_collision : ", self.reward_penalty_collision() * REWARD_K)          
                print("reward_based_farthest_agent_distance : ", self.reward_based_farthest_agent_distance() * REWARD_L)


        elif (self.robot_version == 'T'):
            self.robot.robot_policy_going_exit()      

        elif (self.robot_version == 'R'):
            self.robot.robot_policy_go_and_back()  

        #print(self.alived_agents())
        self.schedule.step()
        

        self.previous_evacuated = self.now_evacuated
        self.now_evacuated = self.evacuated_agents()

        self.previous_evacuated_with_robot = self.now_evacuated_with_robot
        self.now_evacuated_with_robot = self.evacuated_agents_with_robot()


        #print("farthest reward : ", self.reward_based_farthest_agent_distance())
        
        

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
    
    def agents_near_robot_num(self):
        agent_total = 0
        for agent in self.crowds:
            if (agent.type == 0 and agent.dead == False):
                agent_total += 1
        return agent_total

    
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
    
    def reward_based_alived(self):
        reward = 0
        num = 0
        
        reward = -self.alived_agents()/self.total_agents
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
        from config import USING_TRAINED_MODEL
        input_shape = (50, 50)
        num_actions = 4

        self.sac_agent = SACAgent(input_shape, start_epsilon=0)
        if (USING_TRAINED_MODEL):
            self.sac_agent.load_model(file_path)
        self.using_model = True
        self.sac_agent.policy.eval()

    
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

    def reward_based_all_agents_danger_root(self):
        reward = 0
        for agent in self.crowds:
            if(agent.type == 0 or agent.type == 1 or agent.type == 2) and (agent.dead == False):
                reward += agent.danger
        return -math.sqrt(reward)
        

    def reward_based_new_founded_agent_danger(self):
        reward = self.new_founded_agent_danger
        self.new_founded_agent_danger = 0
        return reward

    def reward_based_farthest_agent_distance(self):
        farthest_distance = 0
        for agent in self.crowds:
            if(agent.type == 0 or agent.type ==1 or agent.type == 2) and (agent.dead == False):
                distance = self.robot.point_to_point_distance(self.robot.xy, agent.xy)
                if (distance>farthest_distance):
                    farthest_distance = distance
        return -farthest_distance/((self.width**2 + self.height**2)**0.5)
    
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

    def return_current_robot_state(self):
        if (self.agents_near_robot_num() == 0):
            agent_num = 0
        else:
            agent_num = 1

        return (self.robot.xy[0]/MAP_H, self.robot.xy[1]/MAP_W, agent_num)
    
    # 연속 → 라스터 (RL 프레임) 교체
    def return_current_image(self, H: int = 100, W: int = 100):
        
        # 나중에 벽과 장애물은 자정되어있는 것을 쓰게 교체할 것임 (시간이슈)

        img = self.static_grid.copy()

        def to_px(x, y):
            # (0,width)×(0,height) → (0..W-1, 0..H-1)
            ix = int(np.clip(x / self.width  * W, 0, W-1))
            iy = int(np.clip(y / self.height * H, 0, H-1))
            return ix, iy

        # 군중
        for ag in self.crowds:
            if ag.dead:
                continue
            ix, iy = to_px(ag.xy[0], ag.xy[1])
            img[iy, ix] = 150 if ag.type == 0 else 150
            

        # 로봇
        for rb in getattr(self, "robots", []):
            ix, iy = to_px(rb.xy[0], rb.xy[1])
            img[iy, ix] = 255

        return img
    
    def update_obstacles(self, new_polys):
        # self._obstacle_polys = new_polys
        # self.obstacles_version += 1
        # self.vision_atlas.rebuild_obstacles(self._obstacle_polys, self.obstacles_version)
        # print("[vision] obs polys:", len(self._polys))
        
        self.obstacles = [list(poly.exterior.coords) for poly in new_polys]
        self._obstacle_polys = [Polygon(ob) for ob in self.obstacles]

        self.obstacles_grid_points.clear()
        self.construct_map()
        self._build_blocked_grid()
        #self.shadow_fov = ShadowFOV(self.valid_spaces)

        #self.vision_atlas.precompute(rays_per_poly=32, bsearch_iters=8)
    def choice_random_waypoint(self):
        return [random.randint(0, self.width-1), random.randint(0, self.height-1)]

    
    def return_robot(self):
        return self.robot

    def calculate_all_agents_life_time(self):
        total_life_time = 0
        for agent in self.crowds:
            if(agent.type == 0 or agent.type == 1 or agent.type == 2):
                total_life_time += agent.life_time
        return total_life_time


    def _build_blocked_grid(self):
        """
        장애물/외벽 정보를 기반으로 FOV용 blocked[y, x] 맵 생성.
        True = 시야 막힘.
        """
        self.blocked = np.zeros((self.height, self.width), dtype=bool)

        # 1) 외곽 벽
        for x in range(self.width):
            self.blocked[0, x] = True
            self.blocked[self.height - 1, x] = True
        for y in range(self.height):
            self.blocked[y, 0] = True
            self.blocked[y, self.width - 1] = True

        # 2) construct_map 에서 만든 장애물 grid 포인트
        for (gx, gy) in self.obstacles_grid_points:
            if 0 <= gx < self.width and 0 <= gy < self.height:
                self.blocked[gy, gx] = True


    def _render_static_map(self, H : int=100, W : int=100):

        def to_px(x, y):
            # (0,width)×(0,height) → (0..W-1, 0..H-1)
            ix = int(np.clip(x / self.width  * W, 0, W-1))
            iy = int(np.clip(y / self.height * H, 0, H-1))
            return ix, iy

        # 벽/장애물: 폴리곤을 rasterize (경량화: 경계 bbox만 순회)
        for poly in self.obstacles:
            P = Polygon(poly)
            minx, miny, maxx, maxy = P.bounds
            gx0, gy0 = to_px(minx, miny)
            gx1, gy1 = to_px(maxx, maxy)
            for ix in range(min(gx0, gx1), max(gx0, gx1)+1):
                for iy in range(min(gy0, gy1), max(gy0, gy1)+1):
                    # 픽셀 중심 좌표를 월드 좌표로 역변환
                    x = (ix + 0.5) * self.width / W
                    y = (iy + 0.5) * self.height / H
                    if P.contains(Point(x, y)):
                        self.static_grid[iy, ix] = 50  # 벽/장애물

        # 출구
        for epoly in self.exit_list:
            P = Polygon(epoly)
            minx, miny, maxx, maxy = P.bounds
            gx0, gy0 = to_px(minx, miny)
            gx1, gy1 = to_px(maxx, maxy)
            for ix in range(min(gx0, gx1), max(gx0, gx1)+1):
                for iy in range(min(gy0, gy1), max(gy0, gy1)+1):
                    x = (ix + 0.5) * self.width / W
                    y = (iy + 0.5) * self.height / H
                    if P.contains(Point(x, y)):
                        self.static_grid[iy, ix] = max(self.static_grid[iy, ix], 100)
        



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
