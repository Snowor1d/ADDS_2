from agent import RobotAgent
from agent import CrowdAgent

from core import RandomActivation, DataCollector
from space import ContinuousSpace

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.strtree import STRtree
from shapely.ops import triangulate
from shapely.ops import unary_union
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
from typing import Optional
#import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ADDS_AS_reinforcement import ACTION_SCALE, ACTION_STEP, FrameStack, FrameStack2, ego_crop_from_full_map, FrameStackWithStep, make_dreamer_config
from dreamer_v3 import DreamerAgent
from config import *
import json
import ast

PointT = Tuple[int, int]
MAP_W = 10

def load_map_size_from_file(
    map_num: int,
    base_dir: str = "map_infos"
) -> Tuple[int, int]:
    """
    map_infos/map_{map_num}.json에서 width, height를 반드시 읽어온다.

    width/height를 읽을 수 없으면 fallback 없이 즉시 에러를 발생시킨다.

    주의:
    - map_num == -1은 허용하지 않는다.
      FightingModel.__init__()에서 먼저 실제 map_num으로 확정해야 한다.
    - map_num == 0은 procedural random map이므로 이 함수에서 처리하지 않는다.
      map_num == 0을 계속 쓸 거라면 별도 width/height 정책이 필요하다.
    """
    if map_num == -1:
        raise ValueError(
            "[map size] map_num == -1 is not allowed here. "
            "Choose a concrete map_num before loading map size."
        )

    if map_num == 0:
        raise ValueError(
            "[map size] map_num == 0 is procedural random map. "
            "No map_infos/map_0.json size can be loaded."
        )

    fname = f"map_{map_num}.json"

    candidate_paths = [
        os.path.join(os.getcwd(), base_dir, fname),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), base_dir, fname),
    ]

    last_error = None

    for fpath in candidate_paths:
        if not os.path.exists(fpath):
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                obj = json.load(f)

            if "width" not in obj or "height" not in obj:
                raise KeyError(
                    f"[map size] {fpath} must contain both 'width' and 'height'."
                )

            width = int(obj["width"])
            height = int(obj["height"])

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"[map size] width and height must be positive. "
                    f"Got width={width}, height={height} from {fpath}"
                )

            return width, height

        except Exception as e:
            last_error = e
            break

    if last_error is not None:
        raise RuntimeError(
            f"[map size] Failed to load width/height for map_{map_num}.json"
        ) from last_error

    raise FileNotFoundError(
        "[map size] map json file not found. Tried:\n"
        + "\n".join(candidate_paths)
    )


#USE_EGO = (MODEL_VERSION != "ME")
DEBUG_SAVE = False


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


def _dedup_pslg(vertices, segments, ndigits=6):
    # 1) vertex dedup
    key_to_new = {}
    new_vertices = []
    old_to_new = {}

    for i, (x, y) in enumerate(vertices):
        k = (round(float(x), ndigits), round(float(y), ndigits))
        if k not in key_to_new:
            key_to_new[k] = len(new_vertices)
            new_vertices.append([k[0], k[1]])
        old_to_new[i] = key_to_new[k]

    # 2) segment remap + remove zero-length + remove duplicates (undirected)
    seen = set()
    new_segments = []
    for a, b in segments:
        ia = old_to_new[int(a)]
        ib = old_to_new[int(b)]
        if ia == ib:
            continue
        e = (ia, ib) if ia < ib else (ib, ia)
        if e in seen:
            continue
        seen.add(e)
        new_segments.append([ia, ib])

    return new_vertices, new_segments


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



def downsample_full_map(full_map: np.ndarray, target: int) -> np.ndarray:
    """
    full_map: (H, W) uint8
    return: (target, target) uint8 (adaptive pool)
    """
    x = torch.from_numpy(full_map).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    y = F.adaptive_max_pool2d(x, (target, target))
    return y.squeeze(0).squeeze(0).byte().numpy()


def union_obstacles_to_polygons(raw_obstacles, min_area=1e-8):
    """
    raw_obstacles: self.obstacles 같은 형태 (각 obstacle은 [[x,y],...])
    return: (outer_polys, hole_rings)
      - outer_polys: [ [(x,y),...], ... ]   (각각 exterior ring, 마지막 중복점 제거)
      - hole_rings : [ [(x,y),...], ... ]   (각각 interior ring, 마지막 중복점 제거)
    """
    parts = []
    for ob in raw_obstacles:
        if ob is None or len(ob) < 3:
            continue
        P = Polygon(ob)
        if not P.is_valid:
            P = P.buffer(0)  # self-intersection 등 보정
        if P.is_empty:
            continue
        # MultiPolygon이면 쪼개서 parts에 넣기
        if isinstance(P, MultiPolygon):
            parts.extend(list(P.geoms))
        else:
            parts.append(P)

    if not parts:
        return [], []

    U = unary_union(parts)

    # union 결과가 GeometryCollection / MultiPolygon일 수 있음
    polys = []
    if isinstance(U, Polygon):
        polys = [U]
    elif isinstance(U, MultiPolygon):
        polys = list(U.geoms)
    else:
        # GeometryCollection 등: polygon만 추출
        try:
            polys = [g for g in U.geoms if isinstance(g, Polygon)]
        except Exception:
            polys = []

    outer_polys = []
    hole_rings = []

    for P in polys:
        if P.area <= min_area:
            continue

        ext = list(P.exterior.coords)[:-1]
        if len(ext) >= 3:
            outer_polys.append([(float(x), float(y)) for x, y in ext])

        # holes(내부 링)도 segment로 넣어야 "구멍"이 장애물로 유지됨
        for ring in P.interiors:
            h = list(ring.coords)[:-1]
            if len(h) >= 3:
                hole_rings.append([(float(x), float(y)) for x, y in h])

    return outer_polys, hole_rings



class FightingModel(Model):
    """A model with some number of agents."""

    def __init__(self, number_agents: int, model_num=-1, robot='Q', robot_num=1):
        super().__init__()

        # --------------------------------------------------
        # 1) 실제 사용할 map_num을 가장 먼저 확정
        # --------------------------------------------------
        requested_map_num = model_num

        if requested_map_num == -1:
            if MAP_NUM != -2 and MAP_NUM != -1:
                self.map_num = int(MAP_NUM)
            else:
                self.map_num = int(random.choice(MAP_NUM_RANDOM))
        else:
            self.map_num = int(requested_map_num)

        # --------------------------------------------------
        # 2) 확정된 map_num 기준으로 JSON에서 width/height를 반드시 읽음
        #    읽을 수 없으면 fallback 없이 에러 발생
        # --------------------------------------------------
        self.width, self.height = load_map_size_from_file(
            self.map_num,
            base_dir="map_infos"
        )

        self.width = int(self.width)
        self.height = int(self.height)

        self.map_scale_x = self.width / DOWNSAMPLE_MAP_SIZE
        self.map_scale_y = self.height / DOWNSAMPLE_MAP_SIZE


        # --------------------------------------------------
        # 3) width/height 확정 후 space 생성
        # --------------------------------------------------
        self.space = ContinuousSpace(
            self.width,
            self.height,
            cell_size=10.0,
            torus=False
        )
        self.frame_stack = FrameStack(stack_len=4)
        self._first_step = True
        self.robot_version = robot
        
        self.crowds = []
        self.step_n = 0
        self.robot_num = robot_num

        self.robot_type = robot
        self.spaces_of_map = []
        # self.map_num = model_num # 1 : 산학협력관 + 잔디밭 / 2 : 제2 공학관 + 정원 / 3 : 공학실습동 + 제2 연구동 / 4 : 벤젠고리관 / 5 : 경영관 + 퇴계 인문관
        self.running = (
            True  
        )
        
        self.agent_id = 1000
        self.agent_num = 0
        self.robots = []

        self.using_model = False
        self.total_agents = number_agents
        self.obstacle_mesh = []
        self.adjacent_mesh = {}
        # map_ran_num = 2
        self.walls = list()
        self.obstacles = list()
        self.exit_list = list()
        self.mesh = list()
        self.mesh_list = list()
        self.RANDOM_MAP_RANGES = {
            "grid_step": 2,

            # 출구: 두 개 + 거리 확보 느낌
            "exit_count_range": (2, 2),
            "exit_w_range": (4, 8),
            "exit_h_range": (4, 8),
            "exit_keepout_range": (12, 20),

            # 장애물 간격(겹침 + 너무 붙는 것 방지)
            "wall_clearance_range": (5, 10),

            # 장애물 수
            "obstacle_target_range": (5, 12),

            # 현실형 분위기 핵심 (NEW)
            "p_main_range": (0.3, 0.60),
            "p_corr_range": (0.20, 0.40),

            # 기존 도형들은 과하지 않게
            "p_rect_range": (0.10, 0.25),
            "p_L_range": (0.08, 0.18),
            "p_U_range": (0.05, 0.15),

            # main block 크기
            "main_w_range": (30, 50),
            "main_h_range": (30, 50),

            # corridor
            "corr_thickness_range": (4, 10),
            "corr_length_range": (25, 75),
            "max_corridor_aspect": 6.0,

            # deadend
            "deadend_bias_range": (0.25, 0.60),
        }
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
        self.fill_outwalls(self.width, self.height)
        self.mesh_map()
        # if(self.map_num != 0):
        #     self.make_random_exit()
        self.construct_map()
        self.random_agent_distribute_outdoor(number_agents, 1)
        if (self.robot_version != 'N'):
            self.make_robot(self.robot_num)
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
        self._exit_union = unary_union(self._exit_polys) if self._exit_polys else None
        self._mesh_polys = [Polygon(t) for t in self.mesh_list]
        self._mesh_poly2tri = {poly: tri for poly, tri in zip(self._mesh_polys, self.mesh_list)}
        self._mesh_index = STRtree(self._mesh_polys)
        self.calculate_mesh_danger()   
        self._build_blocked_grid()

        self.obstacles_version = 0
        self.vision_atlas = VisibilityAtlas(world_w=self.width, world_h=self.height, region_cells=4.0)
        
        SENSOR_R_AGENT = AGENT_VISION
        SENSOR_R_ROBOT = ROBOT_VISION  
        self.vision_atlas.rebuild_obstacles(self._obstacle_polys, self.obstacles_version)
        self.vision_atlas.set_radii(sorted(set([AGENT_VISION, ROBOT_VISION])))
        self.vision_atlas.precompute(rays_per_poly=64, bsearch_iters=6)

        #self.shadow_fov = ShadowFOV(self.blocked)


        self.exit_meta = [
        {"idx":i,
         "width" : 5,             # exit_width = 5
         } for i in range(len(self.exit_list))
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
        #print("static grid height : ", self.height)

        self.static_grid = np.zeros((self.height, self.width), dtype = np.uint8)
        self._render_static_map(self.height, self.width)

        self.glob_stack = FrameStackWithStep(4, FRAME_STEP)
    


    def load_map_from_file(self, map_num: int, base_dir: str = "main_infos"):
        """
        현재 작업 디렉토리 기준:
        ./main_infos/map_{map_num}.json

        JSON 포맷:
        {
        "width": 100,
        "height": 100,
        "obstacles": [ [[x,y],...], ... ],
        "exits":     [ [[x,y],...], ... ]   # json에는 tuple이 없으니 list로 저장됨
        }
        """
        fname = f"map_{map_num}.json"
        fpath = os.path.join(os.getcwd(), base_dir, fname)

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"[map] not found: {fpath}")

        with open(fpath, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # --- width/height ---
        w = obj.get("width", None)
        h = obj.get("height", None)
        if w is not None and h is not None:
            self.width = int(w)
            self.height = int(h)

        # --- obstacles: [[[x,y],...], ...] 형태 유지 ---
        obstacles_raw = obj.get("obstacles", []) or []
        obstacles: List[List[List[int]]] = []
        for poly in obstacles_raw:
            if not poly or len(poly) < 3:
                continue
            norm_poly: List[List[int]] = []
            for pt in poly:
                # pt가 [x,y] 형태라고 가정 (혹시 dict/tuple 섞여도 대비)
                x, y = pt[0], pt[1]
                norm_poly.append([int(x), int(y)])
            obstacles.append(norm_poly)
        self.obstacles = obstacles

        # --- exits: 내부에서는 (x,y) 튜플 리스트로 맞춤 ---
        exits_raw = obj.get("exits", []) or []
        #print(exits_raw)
        exits: List[List[PointT]] = []
        for poly in exits_raw:
            if not poly or len(poly) < 3:
                continue
            norm_poly: List[PointT] = []
            for pt in poly:
                x, y = pt[0], pt[1]
                norm_poly.append((int(x), int(y)))
            exits.append(norm_poly)
        self.exit_list = exits
        #(self.exit_list)



    # def extract_map_info(self, map_num):
    #     if map_num <= 49:
    #         self.width = 50
    #         self.height = 50
    #     elif map_num <= 99:
    #         self.width = 70
    #         self.height = 70
    #     elif map_num <= 199:
    #         self.width =100
    #         self.height = 100

    def _sanitize_obstacles(self, eps=1e-6):
        polys = []
        for ob in self.obstacles:
            if len(ob) < 3:
                continue

            P = Polygon(ob)
            # self-intersection/비정상 폴리곤 교정
            if not P.is_valid:
                P = P.buffer(0)

            if P.is_empty:
                continue
            if P.area < eps:
                continue

            polys.append(P)

        if not polys:
            self.obstacles = []
            return

        # 겹침/중복 병합
        U = unary_union(polys)

        out = []
        if isinstance(U, Polygon):
            out = [U]
        elif isinstance(U, MultiPolygon):
            out = list(U.geoms)
        else:
            # GeometryCollection이면 폴리곤만 추출
            out = [g for g in getattr(U, "geoms", []) if isinstance(g, Polygon)]

        # 좌표를 리스트로 환원 (마지막 닫힘좌표는 제거)
        new_obs = []
        for P in out:
            coords = list(P.exterior.coords)
            coords = coords[:-1]  # 닫힘 점 제거
            # 좌표 양자화(중요)
            coords = [[round(x, 6), round(y, 6)] for x, y in coords]
            # 연속 중복 제거
            cleaned = []
            for c in coords:
                if not cleaned or cleaned[-1] != c:
                    cleaned.append(c)
            if len(cleaned) >= 3:
                new_obs.append(cleaned)

        self.obstacles = new_obs
    

    def is_free(self, xy):
        x, y = xy
        if x < 0 or y < 0 or x > self.width or y > self.height:
            return False
        p = Point(x, y)
        return not any(poly.contains(p) or poly.touches(p) for poly in self._obstacle_polys)

    # def in_exit(self, xy, tol=0.0):
    #     p = Point(xy[0], xy[1])
    #     for poly in self._exit_polys:
    #         #print(f"{p}가 {poly}에 있는지 확인합니다")    
    #         if (poly.buffer(tol).contains(p)) if tol > 0 else poly.contains(p):
    #             return True
    #     return False

    def in_exit(self, xy, tol=0.3) -> bool:
        if self._exit_union is None:
            return False
        p = Point(xy[0], xy[1])
        return self._exit_union.buffer(tol).covers(p)
    

    def nearest_exit(self, xy):
        """
        xy에서 가장 가까운 exit_polygon과
        그 polygon 경계 상 최단거리 점(q) 및 거리(d) 반환.
        Returns:
            (best_idx, (qx,qy), d)
        """
        if not self._exit_polys:
            return None, (xy[0], xy[1]), float("inf")

        p = Point(xy[0], xy[1])
        best_idx = None
        best_d = float("inf")
        best_q = (xy[0], xy[1])

        for i, poly in enumerate(self._exit_polys):
            d = poly.distance(p)  # 내부면 0
            if d < best_d:
                best_d = d
                best_idx = i
                if d == 0.0:
                    best_q = (xy[0], xy[1])
                else:
                    # 경계선 위의 최단점(투영점)
                    q = poly.exterior.interpolate(poly.exterior.project(p))
                    best_q = (q.x, q.y)

        return best_idx, best_q, best_d
    

    def goal_point_into_exit(self, exit_idx: int, from_xy, eps=0.5):
        """
        exit 경계 최단점까지 도달했을 때 '안으로 들어가게' 만들기 위한 내부 목표점.
        - 경계 최단점 q에서 polygon 내부 방향으로 eps 만큼 이동.
        """
        poly = self._exit_polys[exit_idx]
        p = Point(from_xy[0], from_xy[1])

        # 경계 최단점
        q = poly.exterior.interpolate(poly.exterior.project(p))
        # 내부 대표점(centroid보다 안전한 representative_point)
        inside = poly.representative_point()

        vx = inside.x - q.x
        vy = inside.y - q.y
        norm = (vx*vx + vy*vy) ** 0.5
        if norm < 1e-9:
            return (inside.x, inside.y)

        ux, uy = vx / norm, vy / norm
        gx, gy = q.x + eps * ux, q.y + eps * uy

        # 혹시 eps 이동이 밖이면 eps 줄이기(최대 몇 번만)
        for _ in range(5):
            if poly.covers(Point(gx, gy)):
                return (gx, gy)
            gx, gy = q.x + (eps*0.5) * ux, q.y + (eps*0.5) * uy
            eps *= 0.5

        # 최후 fallback
        return (inside.x, inside.y)
    
    def distance_to_exit(self, xy):
        """exit_polygon까지 최단거리(내부면 0)."""
        _, _, d = self.nearest_exit(xy)
        return d

    def visible_exits(self, xy, radius):
        """
        xy에서 radius 내 '시야 폴리곤' 기준으로 보이는 출구 idx 리스트.
        mode:
        - "intersects": 시야폴리곤과 출구폴리곤이 겹치면 visible (관대)
        - "rep_point": 출구 대표점이 시야폴리곤에 포함되면 visible (보수)
        """
        vision_poly = self.vision_atlas.polygon_at(
            float(xy[0]), float(xy[1]), float(radius), self.obstacles_version
        )
        if vision_poly is None or vision_poly.is_empty:
            return []

        out = []
        for i, ex in enumerate(self._exit_polys):
            if ex is None or ex.is_empty:
                continue


            if vision_poly.intersects(ex):
                out.append(i)

        return out

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
        """
        mesh_danger[mesh] = 해당 mesh(대표점)에서 exit_polygon까지의 최단 '경로거리'
        """
        BIG = 1e18
        self.mesh_danger = {}

        # mesh 대표점 -> safe mesh 매칭 안정화를 위해 centroid 사용
        for mesh in self.pure_mesh:
            cx, cy = ((mesh[0][0] + mesh[1][0] + mesh[2][0]) / 3.0,
                    (mesh[0][1] + mesh[1][1] + mesh[2][1]) / 3.0)

            best = BIG
            for exit_idx in range(len(self._exit_polys)):
                q, _ = self.nearest_point_on_exit(exit_idx, (cx, cy))
                # 기존 네비메쉬 기반 거리 사용
                d = self.robots[-1].point_to_point_distance((cx, cy), q)

                if d < best:
                    best = d

            self.mesh_danger[mesh] = best

        return 0

 
    def mesh_map(self):
        self._sanitize_obstacles()
        D = 20
        outer_polys, _ = union_obstacles_to_polygons(self.obstacles)
        self.obstacles = [ [list(p) for p in poly ] for poly in outer_polys]
        map_boundary = [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]]
        obstacle_hulls = []

        for obstacle in self.obstacles:
            obstacle_hulls.append(np.array(obstacle, dtype=float))
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
        vertices_with_points, segments_with_points = _dedup_pslg(vertices_with_points, segments_with_points)
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


        obstacle_polys = [Polygon(ob) for ob in self.obstacles]

        for mesh in self.mesh_list:
            mx = (mesh[0][0] + mesh[1][0] + mesh[2][0]) / 3.0
            my = (mesh[0][1] + mesh[1][1] + mesh[2][1]) / 3.0
            p = Point(mx, my)

            if any(P.contains(p) or P.touches(p) for P in obstacle_polys):
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
        self.load_map_from_file(map_num, base_dir="map_infos")

    def make_random_exit_2(self, seed: Optional[int] = None, difficulty: int = 6):
        """
        map_num == 0 전용:
        - Shapely(unary_union) 기반 random_map.generate_map() 사용 (difficulty 단일 입력 버전)
        - difficulty(1/2/3)에 따라 출구/장애물/편향/간격 등이 내부 테이블로 자동 결정됨
        """
        from random_map import RandomMapSpec, generate_map

        spec = RandomMapSpec(
            width=self.width,
            height=self.height,
            difficulty=difficulty,   # ✅ 추가
            seed=seed,
            # attempts는 필요하면 여기서 덮어쓰기 가능 (RandomMapSpec 기본값 사용)
            # map_level_max_tries=200,
            # obstacle_level_max_tries=20000,
            # exit_max_tries=2000,
        )

        data = generate_map(spec)

        self.obstacles = data.obstacles
        self.exit_list = data.exits  # polygon list 유지
        self.random_seed = data.seed_used
        self.is_random_map = True

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


                                  
    def make_robot(self, robot_num):
        self.robot_placement(robot_num, padding=2.0) #로봇 배치 


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
            [(self.width-exit_width, 0), (self.width, 0), (self.width, exit_height), (self.width-exit_width, exit_height)],  # 오아
            [(self.width-exit_width, self.height-exit_height), (self.width, self.height-exit_height), (self.width, self.height), (self.width-exit_width, self.height)],  # 오위
            [(0, self.height-exit_height), (exit_width, self.height-exit_height), (exit_width, self.height), (0, self.height)],  #  왼위
          
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
        elif (self.map_num == 109):
            index = [0, 2]
        elif (self.map_num == 110):
            index = [1, 3]
        elif (self.map_num == 111):
            index = [0, 2]
        elif (self.map_num == 112):
            index = [0, 2]
        elif (self.map_num == 113):
            index = [0, 2]
        elif (self.map_num == 114):
            index = [1, 3]
        elif (self.map_num == 115):
            index = [0, 2]
        elif (self.map_num == 116):
            index = [0, 2]
        elif (self.map_num == 117):
            index = [1, 3]
        elif (self.map_num == 118):
            index = [0, 2]
        elif (self.map_num == 119):
            index = [0, 2]
        elif (self.map_num == 120):
            index = [1, 3]
        elif (self.map_num == 121):
            index = [0, 2]
        elif (self.map_num == 50):
            index = [2]
        elif (self.map_num == 51):
            index = [3]

        elif (self.map_num == 53):
            index = [1]
        elif (self.map_num == 54):
            index = [1]

    
        for i in range(len(index)):
            self.exit_list.append(all_exits[index[i]])

        if (self.map_num == 122):
            self.exit_list.append([(0, 75), (5, 75), (5, 100), (0, 100)])
        elif (self.map_num == 123):
            self.exit_list.append([(0,0), (20, 0), (20, 5), (0, 5)])
            self.exit_list.append([(95, 0), (100, 0), (100, 5), (95, 5)])
        elif (self.map_num == 124):
            self.exit_list.append([(0, 25), (5, 25), (5, 45), (0, 45)])
            self.exit_list.append([(95, 41), (100, 41), (100, 49), (95, 49)])
        elif (self.map_num == 125):
            self.exit_list.append([(10, 95), (30, 95), (30, 100), (10, 100)])
            self.exit_list.append([(95, 30), (100, 30), (100, 50), (95, 50)])
        elif (self.map_num == 130):
            self.exit_list.append([(0, 72), (5, 72), (5, 82), (0,82)])
            self.exit_list.append([(95, 72), (100, 72), (100, 82), (95, 82)])
        elif (self.map_num == 131):
            self.exit_list.append([(0, 65), (0, 73), (5, 73), (5, 65)])
            self.exit_list.append([(47, 95), (57, 95), (57, 100), (47, 100)])
            self.exit_list.append([(21, 0), (29, 0), (29, 5), (21, 5)])
        elif (self.map_num == 132):
            self.exit_list.append([(0, 31), (5, 31), (5, 69), (0, 69)])
        #     self.exit_list.append([(0, 70), (5, 70), (5, 80), (0, 80)])
        #     self.exit_list.append([(78, 92), (100, 92), (100, 100), (78, 100)])
        elif (self.map_num == 133):
            self.exit_list.append([(0, 30), (5, 30), (5, 40), (0, 40)])

        elif (self.map_num == 134):
            self.exit_list.append([[0, 0], [5, 0], [5, 10], [0,10]])
            self.exit_list.append([[81, 95], [92, 95], [92, 100], [81, 100]])

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
        for _ in range(n_robots):
            x, y = _sample_safe_pos(self, padding=padding)
            self._occupied_positions.append((x, y))

            robot = RobotAgent(self.agent_id, self, [x, y], 3, len(self.robots))

            robot.map_scale_x = self.map_scale_x
            robot.map_scale_y = self.map_scale_y

            robot.latest_ego_f = None
            robot.latest_glob_f = None
            robot.latest_ego_state = None
            robot.latest_glob_state = None
            robot.latest_robot_state = None
            robot.glob_stack = FrameStackWithStep(4, FRAME_STEP)

            self.agent_id += 10
            self.robots.append(robot)
            self.schedule.add(self.robots[-1])
            # ✨ 연속 공간에 배치
            self.space.place_agent(self.robots[-1], (x, y))
            self.agents.append(self.robots[-1])
    

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
        
        """식 (12): distance·density·width 3요소 (exit_polygon 버전)."""

        # (i) 거리: agent → exit_polygon(경계 최단점 q)까지의 '경로거리'
        q, _ = self.nearest_point_on_exit(exit_idx, agent.xy)
        d_s = agent.point_to_point_distance(agent.xy, q)

        # (ii) 밀도(혼잡): 동일 출구를 목표로 하는 인원 수
        people_to_exit = sum(
            bool(ag.exit_belief and ag.exit_belief["idx"] == exit_idx)
            for ag in self.crowds if not ag.dead
        )
        d_e = people_to_exit

        # (iii) 폭
        d_w = self.exit_meta[exit_idx]["width"]

        base  = (K1*np.exp(-d_s) + K2*np.exp(-d_e) + K3*(1 - np.exp(-d_w)))
        score = np.exp(-alpha) * base / (K1 + K2 + K3)
        return score
        
    def nearest_point_on_exit(self, exit_idx, xy):
        poly = self._exit_polys[exit_idx]
        p = Point(xy[0], xy[1])
        d = poly.distance(p)
        if d == 0.0:
            return (xy[0], xy[1]), 0.0
        q = poly.exterior.interpolate(poly.exterior.project(p))
        return (q.x, q.y), d


    def step(self):
        self.stats.collect(self)
        self.step_n += 1
        self.step_count += 1

        
        for robot in self.robots:
                
            if (self.robot_version == 'Q' and self.using_model):
                # 1) full map (uint8) 가져오기 (학습 때랑 동일: MAP_H, MAP_W)
                #print("MAP_H :", MAP_H)
                full_map = self.return_current_image(self.height, self.width)  # (H,W) uint8

                # 3) ego/global frame 만들기
                # _build_ego_global_frames 내부 구현에 따라 반환값이 np.array 형태여야 함

                ego_f, glob_f = self.build_ego_global_frames(full_map, robot.robot_index)

                ego_state = robot.ego_stack.update(ego_f)
                glob_state = self.glob_stack.update(glob_f)


                # 5) 로봇 상태 가져오기 (항상 필요할 수 있음)
                robot_state = np.array(self.return_current_robot_state(robot.robot_index), dtype=np.float32)
                
                robot.update_latest_state_images(ego_f, glob_f, ego_state, glob_state, robot_state)
                robot.latest_ego_f = ego_f.copy()
                robot.latest_glob_f = glob_f.copy()
                robot.latest_ego_state = ego_state.copy()
                robot.latest_glob_state = glob_state.copy()
                robot.latest_robot_state = robot_state.copy()

            # 6) 로봇 행동 결정 및 수행
            if self.robot_version == 'Q':
                if self.using_model:
                    # ---------------------------------------------------------
                    # [수정됨] Agent의 select_action과 인터페이스 통일
                    # 내부적으로 agent.select_action(ego, global, robot, deterministic=True) 호출
                    # ---------------------------------------------------------
                    if robot.new_order_need:
                        action, _ = self.sac_agent.select_action(ego_state, glob_state, robot_state, deterministic=False)
                        robot.new_order_need = False
                        robot.receive_action_from_policy(action)
                    # else:
                    #     action = robot.last_action
                    


                    # dx, dy = float(action[0]), float(action[1])

                    # # 환경에 행동 적용 (실제 이동량이나 결과 반환 가능)
                    # real_action = self.robots[robot.robot_index].receive_action([dx, dy])

            elif self.robot_version == 'T':
                self.robots[robot.robot_index].robot_policy_going_exit()
            # elif self.robot_version == 'R':
            #     self.robot.robot_policy_go_and_back()

        # 7) 환경 진행 (Mesa schedule step)
        self.schedule.step()

        # 8) 통계 업데이트 (대피 인원 등)
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
    
    def reward_penalty_robot_index(self, robot_index):
        reward = 0
        guided_num = 0
        for agent in self.crowds:
            if(agent.type == 0 and agent.dead == False):
                guided_num += 1
        if(guided_num == 0 and self.robots[robot_index].danger<5):
            return -1
        return 0

    def reward_penalty(self):
        reward = 0
        for i in range(len(self.robots)):
            reward += self.reward_penalty_robot_index(i)
        return reward
    
    def reward_penalty_collision(self):
        reward = 0
        for i in range(len(self.robots)):
            reward += self.reward_penalty_collision_robot_index(i)
        return reward
    
    def agents_near_robot_num_robot_index(self, robot_index):
        agent_total = 0
        for agent in self.crowds:
            if (agent.type == 0 and agent.dead == False and agent.following_robot_index == robot_index):
                agent_total += 1
        return agent_total

    
    
    def reward_based_alived(self):
        reward = 0
        num = 0
        
        reward = -self.alived_agents()/self.total_agents
        return reward
    
        

    def return_agent_id(self, agent_id):
        for agent in self.agents:
            if(agent.unique_id == agent_id):
                return agent
        return None
    
    def use_model(self, file_path):
        from config import USING_TRAINED_MODEL

        self.sac_agent = DreamerAgent(make_dreamer_config(device=DEVICE))
        if (USING_TRAINED_MODEL):
            self.sac_agent.load_model(file_path)
        self.using_model = True
        self.sac_agent.world_model.eval()
        self.sac_agent.actor.eval()


    def reward_penalty_collision_robot_index(self, robot_index):
        if self.robots[robot_index].collision_check :
            return -1
        else :
            return 0

    def return_current_robot_state(self, robot_index):

        angle = self.robots[robot_index].angle
        sin_a = np.sin(angle)
        cos_a = np.cos(angle)
    

        return (

            self.robots[robot_index].xy[0] / max(1, self.width),
            self.robots[robot_index].xy[1] / max(1, self.height),
            sin_a,
            cos_a,
            self.robots[robot_index].map_scale_x,
            self.robots[robot_index].map_scale_y
        )
    
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

            if self.is_partial_crowd_observation():
                if not self.is_crowd_observed_by_any_robot(ag, radius=ROBOT_VISION):
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

    
    def return_robot(self, robot_index):
        return self.robots[robot_index]

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

    def _robot_world_to_px(self, robot_index):
        rx, ry = self.robots[robot_index].xy  # world coords

        W = max(1, int(self.width))
        H = max(1, int(self.height))

        ix = int(np.clip(rx / self.width  * W, 0, W - 1))
        iy = int(np.clip(ry / self.height * H, 0, H - 1))
        return ix, iy

    def build_ego_global_frames(self, full_map_u8: np.ndarray, robot_index=0):
        ix, iy = self._robot_world_to_px(robot_index)

        ego_u8 = ego_crop_from_full_map(full_map_u8, (ix, iy), EGO_MAP_SIZE, self.robots[robot_index].angle, pad_value=50)     # (EGO,EGO)
        glob_u8 = downsample_full_map(full_map_u8, DOWNSAMPLE_MAP_SIZE)                       # (DOWN,DOWN)

        ego_f = ego_u8.astype(np.float32) / 255.0
        glob_f = glob_u8.astype(np.float32) / 255.0
        return ego_f, glob_f

    def is_partial_crowd_observation(self) -> bool :
        return not bool(BLACK_SHEEP_WALL)
    
    def robot_visibility_polygon(self, robot_index: int, radius: float = None):
        """
        robot_index번 robot의 현재 FOV polygon 반환.
        장애물 occlusion은 VisibilityAtlas가 반영한다.
        """
        if radius is None:
            radius = ROBOT_VISION

        if robot_index < 0 or robot_index >= len(self.robots):
            return None

        rb = self.robots[robot_index]

        poly = self.vision_atlas.polygon_at(
            float(rb.xy[0]),
            float(rb.xy[1]),
            float(radius),
            self.obstacles_version
        )

        expected_area = math.pi * float(radius) * float(radius)
        min_valid_area = expected_area * 0.03

        invalid_visibility = (
            poly is None
            or poly.is_empty
            or (hasattr(poly, "area") and poly.area < min_valid_area)
        )

        if invalid_visibility:
            rb.visibility_fail_count = getattr(rb, "visibility_fail_count", 0) + 1

            cached_poly = getattr(rb, "latest_visibility_poly", None)
            if cached_poly is not None and not cached_poly.is_empty:
                return cached_poly

            return None

        # 정상 polygon이면 cache 갱신
        rb.visibility_fail_count = 0
        rb.latest_visibility_poly = poly
        rb.latest_visibility_xy = (float(rb.xy[0]), float(rb.xy[1]))

        return poly


    def is_crowd_observed_by_any_robot(self, agent, radius: float = None) -> bool:
        """
        여러 robot이 관측 정보를 공유한다고 가정한다.

        BLACK_SHEEP_WALL == True:
            모든 살아있는 crowd가 관측 가능.

        BLACK_SHEEP_WALL == False:
            하나 이상의 robot FOV 안에 들어온 crowd만 관측 가능.
        """
        if getattr(agent, "dead", False):
            return False

        if not self.is_partial_crowd_observation():
            return True

        if radius is None:
            radius = ROBOT_VISION

        ax = float(agent.xy[0])
        ay = float(agent.xy[1])

        for robot_index, rb in enumerate(self.robots):
            dx = ax - float(rb.xy[0])
            dy = ay - float(rb.xy[1])

            # 1차 거리 체크
            if dx * dx + dy * dy > float(radius) * float(radius):
                continue

            # 2차 FOV polygon 체크: 장애물 occlusion 포함
            poly = self.robot_visibility_polygon(robot_index, radius=radius)
            if poly is None:
                continue

            if poly.covers(Point(ax, ay)):
                return True

        return False


    def get_shared_observable_crowds(self, radius: float = None):
        """
        현재 모든 robot이 공유해서 알고 있는 crowd 리스트.
        """
        return [
            ag for ag in self.crowds
            if self.is_crowd_observed_by_any_robot(ag, radius=radius)
        ]



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
