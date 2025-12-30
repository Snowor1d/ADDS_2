# continuous_renderer.py
from __future__ import annotations
import math, os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle, Circle, FancyArrowPatch
from matplotlib.transforms import Affine2D
from typing import Iterable, Tuple, Dict, List, Optional
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba

_EPS = 1e-6

class ContinuousRenderer:
    """
    연속공간 렌더러 (논문/데모용 벡터 스타일)
    - 외곽 벽/장애물/출구/군중/로봇을 연속 좌표계로 표현
    - 궤적: fade/persist, 대상 선택(crowd/robot/both)
    - 방향 화살표 on/off + 스케일
    - 색상 테마 전체 커스터마이즈
    - 로봇 형상: 원(circle) 또는 임의 이미지(image)
    - 출구 5x5 스냅(경계에 딱 붙게)
    - 로봇 궤적 라벨링: 숫자 / 지하철 노선도 스타일 / 프레임 아이콘 스타일
    """

    def __init__(
        self,
        # ===== Canvas =====
        world_size: Tuple[float, float] = (50.0, 50.0),
        dpi: int = 140,
        show_axes: bool = False,
        bg_color: str = "white",
        draw_outer_wall: bool = True,

        # ===== Sizes =====
        crowd_radius: float = 0.35,
        robot_radius: float = 0.6,
        wall_linewidth: float = 2.0,

        # ===== Colors =====
        crowd_colors: Dict[int, str] | None = None,  # agent.type별 (미지정: "else")
        robot_color: str = "#e15759",                # 빨강 계열
        wall_color: str = "#111111",
        exit_color: str = "#2ecc71",                 # 초록
        obstacle_fill: str = "#222222",
        single_color_edges: bool = True,             # 선/채우기 통일

        # ===== Exit (5x5 snap) =====
        exit_size: float = 5.0,                      # 맵 50x50 기준 5
        snap_exit_to_boundary: bool = True,

        # ===== Trails =====
        trail_target: str = "none",                  # "none" | "crowd" | "robot" | "both"
        trail_style: str = "fade",                   # "fade" | "persist"
        max_trail: int = 40,
        crowd_trail_alpha: float = 0.55,
        robot_trail_alpha: float = 0.7,

        # ===== Heading arrows =====
        show_agent_heading: bool = False,
        show_robot_heading: bool = False,
        agent_heading_scale: float = 1.5,
        robot_heading_scale: float = 4,

        agent_heading_color: str | Dict[int, str] | None = None,
        agent_heading_linewidth: float = 1.0,
        agent_heading_mutation_scale: float = 9,

        # ===== Filtering =====
        hide_dead: bool = True,
        exclude_types: Iterable[int] = (),

        # ===== Robot shape =====
        robot_style: str = "circle",                 # "circle" | "image"
        robot_image_path: Optional[str] = None,      # png 등
        robot_image_scale: float = 1.6,              # 반지름 대비 이미지 스케일
        robot_image_rotate_by_velocity: bool = False,

        # ===== Robot path annotations =====
        annotate_robot_path: bool = False,            # 표기 켜기/끄기
        annotate_mode: str = "every_n",              # "every_n" | "all" | "endpoints"
        annotate_every: int = 10,                    # 간격
        annotate_style: str = "number",              # "number" | "subway" | "frame"
        annotate_fontsize: int = 9,
        annotate_offset: Tuple[float, float] = (0.0, 0.0),  # (x,y) 오프셋

        annotate_text_color="black",       # 텍스트를 검정으로
        annotate_face_color="yellow",      # 배경을 노랑
        annotate_edge_color="red",         # 테두리 빨강

        annotate_face_alpha: float = 0.95,           # subway/frame 아이콘 불투명도
        annotate_edge_width: float = 1.6,            # 아이콘 테두리 두께
        annotate_min_gap: int = 1,                   # 라벨 재사용 최소 갯수(겹침 방지용 간단 옵션)



        # ==== Vision overlay ====
        show_agent_vision: bool = True,
        show_robot_vision: bool = False,
        agent_vision_color: str = "#4e79a7",
        robot_vision_color: str = "#e15759",
        vision_alpha: float = 0.15,
        vision_edge_alpha: float = 0.35,
        vision_drawing: bool = False,



        # ===== NEW: Mesh viz =====
        show_mesh: bool = False,
        mesh_use_only_pure: bool = True,          # obstacle_mesh는 제외
        mesh_face_a: str = "#8ecae6",             # 교차색 A
        mesh_face_b: str = "#219ebc",             # 교차색 B
        mesh_alpha: float = 0.18,
        mesh_edge_color: str = "#0b7285",
        mesh_edge_width: float = 0.6,

        # ===== NEW: Agent-to-exit distance viz =====
        show_agent_exit_distance: bool = False,
        distance_exit_mode: str = "nearest",      # "nearest" | "fixed"
        distance_exit_index: int = 0,             # mode="fixed"일 때 대상 출구 idx
        distance_line_alpha: float = 0.85,
        distance_line_width: float = 1.2,
        distance_line_color: str = "#222222",
        distance_label_fontsize: int = 15,
        distance_label_color: str = "black",
        distance_max_agents: int = 9999,          # 너무 복잡하면 상한으로 컷

        distance_label_colormap: str = "gradient",  # "static" | "gradient" | "colormap"
        distance_label_cmap_name: str =  "viridis",
        distance_color_min : float = 0.0,
        distance_color_max : float = 60.0
        

        
    ):
        # ===== Canvas =====
        self.W, self.H = world_size
        self.dpi = dpi
        self.show_axes = show_axes
        self.bg_color = bg_color
        self.draw_outer_wall_flag = draw_outer_wall

        # ===== Sizes =====
        self.crowd_r = crowd_radius
        self.robot_r = robot_radius
        self.wall_lw = wall_linewidth

        # ===== Colors =====
        self.colors = {
            "robot": robot_color,
            "wall": wall_color,
            "exit": exit_color,
            "obstacle_fill": obstacle_fill,
            "trail": "#ff9896",     # 기본 trail 색 (single_color_edges=False일 때 crowd trail 용)
            "dead": "#7f7f7f",
        }

        # ===== Vision overlay (store) =====
        self.show_agent_vision = show_agent_vision
        self.show_robot_vision = show_robot_vision
        self.agent_vision_color = agent_vision_color
        self.robot_vision_color = robot_vision_color
        self.vision_alpha = vision_alpha
        self.vision_edge_alpha = vision_edge_alpha

        self.agent_heading_color = agent_heading_color
        self.agent_heading_linewidth = float(agent_heading_linewidth)
        self.agent_heading_mutation_scale = float(agent_heading_mutation_scale)

        self.single_color_edges = single_color_edges
        if crowd_colors is None:
            self.crowd_colors = {0:"#4e79a7", 1:"#4e79a7", 2:"#76b7b2", "else":" #59a14f"}
        else:
            self.crowd_colors = crowd_colors

        # ===== Exit =====
        self.exit_size = exit_size
        self.snap_exit_to_boundary = snap_exit_to_boundary

        # ===== Trails =====
        self.trail_target = trail_target
        self.trail_style = trail_style
        self.max_trail = max_trail
        self.crowd_trail_alpha = crowd_trail_alpha
        self.robot_trail_alpha = robot_trail_alpha

        # ===== Heading =====
        self.show_agent_heading = show_agent_heading
        self.show_robot_heading = show_robot_heading
        self.agent_heading_scale = agent_heading_scale
        self.robot_heading_scale = robot_heading_scale

        # ===== Filtering =====
        self.hide_dead = hide_dead
        self.exclude_types = set(exclude_types)

        # ===== Robot shape =====
        self.robot_style = robot_style
        self.robot_img = None
        if self.robot_style == "image" and robot_image_path and os.path.exists(robot_image_path):
            import matplotlib.image as mpimg
            self.robot_img = mpimg.imread(robot_image_path)
        self.robot_image_scale = robot_image_scale
        self.robot_image_rotate = robot_image_rotate_by_velocity

        # ===== Robot path annotations =====
        self.annotate_robot_path = annotate_robot_path
        self.annotate_mode = annotate_mode
        self.annotate_every = max(1, int(annotate_every))
        self.annotate_style = annotate_style  # "number" | "subway" | "frame"
        self.annotate_fontsize = annotate_fontsize
        self.annotate_offset = annotate_offset
        self.annotate_face_alpha = annotate_face_alpha
        self.annotate_edge_width = annotate_edge_width
        self.annotate_min_gap = max(1, int(annotate_min_gap))

        self.annotate_text_color = annotate_text_color
        self.annotate_face_color = annotate_face_color
        self.annotate_edge_color = annotate_edge_color

        # ===== NEW: Mesh viz =====
        self.show_mesh = show_mesh
        self.mesh_use_only_pure = mesh_use_only_pure
        self.mesh_face_a = mesh_face_a
        self.mesh_face_b = mesh_face_b
        self.mesh_alpha = float(mesh_alpha)
        self.mesh_edge_color = mesh_edge_color
        self.mesh_edge_width = float(mesh_edge_width)

        # ===== NEW: Agent-to-exit distance viz =====
        self.show_agent_exit_distance = show_agent_exit_distance
        self.distance_exit_mode = distance_exit_mode
        self.distance_exit_index = int(distance_exit_index)
        self.distance_line_alpha = float(distance_line_alpha)
        self.distance_line_width = float(distance_line_width)
        self.distance_line_color = str(distance_line_color)
        self.distance_label_fontsize = int(distance_label_fontsize)
        self.distance_label_color = str(distance_label_color)
        self.distance_max_agents = int(distance_max_agents)

        self.distance_label_colormap = distance_label_colormap
        self.distance_label_cmap_name = distance_label_cmap_name
        self.distance_color_min = float(distance_color_min)
        self.distance_color_max = float(distance_color_max)


        # ===== Buffers =====
        self.trails_crowd: Dict[int, List[Tuple[float,float]]] = {}
        # 로봇은 step 함께 저장: (x, y, step)
        self.trails_robot: Dict[int, List[Tuple[float,float,int]]] = {}

        self.vision_drawing = vision_drawing

        # ===== Matplotlib fig/ax =====
        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=self.dpi, facecolor="black")
        self.ax = self.fig.add_axes([0,0,1,1], facecolor="black")
        self.fig.subplots_adjust(0, 0, 1, 1)
        self._setup_axes()
        self.show_mesh = show_mesh

    # ====================== Public ======================
    def draw(self, model, step: Optional[int] = None) -> np.ndarray:
        """현재 상태를 그림. step을 넘기면 로봇 궤적 번호에 절대 스텝을 기록."""
        self.ax.clear()
        self._setup_axes()


        self._draw_obstacles(getattr(model, "obstacles", []))
        if self.vision_drawing:
            self._draw_vision(model)

        if self.show_mesh:
            self._draw_mesh(model)

        self._draw_exits(model)
        self._draw_crowds(getattr(model, "crowds", []))
        self._draw_obstacles(getattr(model, "obstacles", []))
        self._draw_robot(self._find_robot(model), step=step)
        if self.draw_outer_wall_flag:
            self._draw_outer_wall()

        if self.show_agent_exit_distance:
            self._draw_agent_exit_distances(model)

        self.fig.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape((h, w, 3))

    # ====================== Internals ======================
    def _setup_axes(self):
        self.ax.set_xlim(0, self.W)
        self.ax.set_ylim(0, self.H)
        self.ax.set_aspect("equal", adjustable="box")
        self.fig.patch.set_facecolor(self.bg_color)
        self.ax.set_facecolor(self.bg_color)
        if not self.show_axes:
            self.ax.axis("off")
        else:
            self.ax.set_xlabel("x"); self.ax.set_ylabel("y")
            self.ax.grid(True, alpha=0.2, linewidth=0.5)

    def _draw_outer_wall(self):
        rect = Rectangle((0, 0), self.W, self.H,
                         fill=False, linewidth=self.wall_lw*1.5,
                         edgecolor=self.colors["wall"], joinstyle="miter")
        self.ax.add_patch(rect)

    def _edge_and_fill(self, key: str):
        c = self.colors[key]
        if self.single_color_edges:
            return c, c
        return c, c  # 필요하면 선/채움 분리

    def _draw_obstacles(self, obstacles):
        edge_c, fill_c = self._edge_and_fill("wall")
        for poly in obstacles or []:
            if len(poly) == 2:  # rect by two corners
                (x1, y1), (x2, y2) = poly
                rect = Rectangle((min(x1, x2), min(y1, y2)),
                                 abs(x2 - x1), abs(y2 - y1),
                                 facecolor=self.colors["obstacle_fill"],
                                 edgecolor=edge_c,
                                 linewidth=self.wall_lw, alpha=0.95,
                                 joinstyle="miter")
                self.ax.add_patch(rect)
            else:
                patch = MplPolygon(np.asarray(poly, float), closed=True,
                                   facecolor=self.colors["obstacle_fill"],
                                   edgecolor=edge_c,
                                   linewidth=self.wall_lw,
                                   alpha=0.95, antialiased=True,
                                   joinstyle="miter")
                self.ax.add_patch(patch)

    def _snap_rect_to_boundary(self, x, y, w, h):
        # 경계 근접 시 스냅 + 클램프
        if abs(x) < _EPS: x = 0
        if abs((x + w) - self.W) < _EPS: x = self.W - w
        if abs(y) < _EPS: y = 0
        if abs((y + h) - self.H) < _EPS: y = self.H - h
        x = max(0, min(self.W - w, x))
        y = max(0, min(self.H - h, y))
        return x, y

    def _draw_exits(self, model):
        size = float(self.exit_size)
        edge_c, fill_c = self._edge_and_fill("exit")

        def draw_at_center(cx, cy):
            x = cx - size/2
            y = cy - size/2
            if self.snap_exit_to_boundary:
                # 중심이 경계 근처면 변을 바로 경계에 맞춤
                if abs(cx) < size/2 + _EPS: x, y = 0, cy - size/2
                if abs(cx - self.W) < size/2 + _EPS: x, y = self.W - size, cy - size/2
                if abs(cy) < size/2 + _EPS: x, y = cx - size/2, 0
                if abs(cy - self.H) < size/2 + _EPS: x, y = cx - size/2, self.H - size
            if self.snap_exit_to_boundary:
                x, y = self._snap_rect_to_boundary(x, y, size, size)
            rect = Rectangle((x, y), size, size,
                             facecolor=fill_c, edgecolor=edge_c,
                             linewidth=1.4, joinstyle="miter")
            self.ax.add_patch(rect)

        drew = False
        if hasattr(model, "exit_point") and model.exit_point:
            for e in model.exit_point:
                draw_at_center(float(e[0]), float(e[1]))
            drew = True
        if (not drew) and hasattr(model, "exit_list") and model.exit_list:
            for poly in model.exit_list:
                p = np.asarray(poly, float)
                cx, cy = float(np.mean(p[:,0])), float(np.mean(p[:,1]))
                draw_at_center(cx, cy)

    def _crowd_color(self, t: int):
        if t in self.crowd_colors: 
            return self.crowd_colors[t]
        return self.crowd_colors.get("else", "#59a14f")

    def _draw_crowds(self, crowds):
        if not crowds: 
            return
        draw_trail = self.trail_target in ("crowd", "both")
        for ag in crowds:
            t = getattr(ag, "type", None)
            if t in self.exclude_types: 
                continue
            if self.hide_dead and getattr(ag, "dead", False): 
                continue

            if ag.dead:
                continue

            x, y = float(ag.xy[0]), float(ag.xy[1])
            body_col = self._crowd_color(t if t is not None else -1)

            # 몸체
            self.ax.add_patch(Circle((x, y), radius=self.crowd_r,
                                     facecolor=body_col, edgecolor=body_col,
                                     linewidth=0.6, antialiased=True))

            # ▼▼▼ 화살표 (색/두께/화살촉크기 외부값 사용)
            if self.show_agent_heading and hasattr(ag, "vel"):
                vx, vy = ag.vel
                head_col = self._agent_heading_col(t, body_col)
                self._arrow(
                    x, y, vx, vy,
                    scale=self.agent_heading_scale,
                    color=head_col,
                    alpha=0.9,
                    linewidth=self.agent_heading_linewidth,
                    mutation_scale=self.agent_heading_mutation_scale
                )

            if draw_trail:
                key = id(ag)
                trail = self.trails_crowd.setdefault(key, [])
                trail.append((x, y))
                if len(trail) > self.max_trail: trail.pop(0)
                self._plot_trail(trail,
                                 alpha_base=self.crowd_trail_alpha,
                                 color=(color if self.single_color_edges else self.colors["trail"]),
                                 style=self.trail_style, linewidth=1.1)

    def _draw_robot(self, robot, step: Optional[int] = None):
        if robot is None: return
        x, y = float(robot.xy[0]), float(robot.xy[1])
        rcol = self.colors["robot"]

        if self.robot_style == "image" and self.robot_img is not None:
            s = self.robot_image_scale * self.robot_r
            im = self.ax.imshow(self.robot_img, extent=[x-s, x+s, y-s, y+s], zorder=3)
            if self.robot_image_rotate and hasattr(robot, "vel"):
                vx, vy = robot.vel
                ang = math.degrees(math.atan2(vy, vx)) if (vx*vx+vy*vy)>_EPS else 0.0
                tr = Affine2D().rotate_deg_around(x, y, ang) + self.ax.transData
                im.set_transform(tr)
        else:
            self.ax.add_patch(Circle((x, y), radius=self.robot_r,
                                     facecolor=rcol, edgecolor=rcol,
                                     linewidth=0.8, antialiased=True, zorder=3))

        if self.show_robot_heading and hasattr(robot, "vel"):
            vx, vy = robot.vel
            self._arrow(x, y, vx, vy, scale=self.robot_heading_scale,
                        color=rcol, alpha=0.95, linewidth=1.5)

        if self.trail_target in ("robot", "both"):
            key = id(robot)
            trail = self.trails_robot.setdefault(key, [])
            # step 기록
            trail.append((x, y, step if step is not None else (trail[-1][2] + 1 if trail else 0)))
            if len(trail) > self.max_trail: trail.pop(0)

            # 선
            self._plot_trail([(px, py) for (px, py, _) in trail],
                             alpha_base=self.robot_trail_alpha,
                             color=rcol, style=self.trail_style, linewidth=1.5)

            # 라벨
            if self.annotate_robot_path:
                self._annotate_robot_trail(trail, color=rcol)

    def _plot_trail(self, pts: List[Tuple[float,float]], alpha_base: float,
                    color: str, style: str, linewidth: float = 1.0):
        if len(pts) < 2: return
        if style == "persist":
            xs, ys = zip(*pts)
            self.ax.plot(xs, ys, linewidth=linewidth, alpha=alpha_base, color=color)
            return
        n = len(pts)
        for i in range(n-1):
            (x1, y1), (x2, y2) = pts[i], pts[i+1]
            a = alpha_base * (i + 1) / (n - 1)  # 최근일수록 진하게
            self.ax.plot([x1, x2], [y1, y2], linewidth=linewidth, alpha=a, color=color)

    def _arrow(self, x, y, vx, vy, scale=1.0, color="black",
               alpha=0.8, linewidth=1.0, mutation_scale=9.0):
        L = math.hypot(vx, vy)
        if L < _EPS: return
        dx, dy = (vx/L)*scale, (vy/L)*scale
        self.ax.add_patch(FancyArrowPatch(
            (x, y), (x + dx, y + dy),
            arrowstyle="-|>", mutation_scale=mutation_scale,
            linewidth=linewidth, color=color, alpha=alpha
        ))

    def _annotate_robot_trail(self, trail_xyz: List[Tuple[float,float,int]], color: str):
        if not trail_xyz: return

        # 어느 인덱스를 표시할지 결정
        n = len(trail_xyz)
        if self.annotate_mode == "all":
            idxs = list(range(n))
        elif self.annotate_mode == "endpoints":
            idxs = [0, n-1] if n > 1 else [0]
        else:  # "every_n"
            idxs = list(range(0, n, self.annotate_every))
            if (n-1) not in idxs: idxs.append(n-1)

        # 간단한 중복 필터(너무 촘촘하면 건너뜀)
        idxs = [idx for k, idx in enumerate(idxs) if (k == 0 or idx - idxs[k-1] >= self.annotate_min_gap)]

        # 스타일별 그리기
        for idx in idxs:
            x, y, step = trail_xyz[idx]
            if step is None: continue
            sx, sy = x + self.annotate_offset[0], y + self.annotate_offset[1]
            label = f"{step}"

            face = self.annotate_face_color or color
            edge = self.annotate_edge_color or color
            text_c = self.annotate_text_color

            if self.annotate_style == "subway":
                self.ax.add_patch(Circle((sx, sy), radius=self.robot_r*0.85,
                                        facecolor=face, edgecolor=edge,
                                        linewidth=self.annotate_edge_width, zorder=4))
                self.ax.text(sx, sy, label,
                            color=text_c, fontsize=self.annotate_fontsize,
                            ha="center", va="center", zorder=5,
                            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
            elif self.annotate_style == "frame":
                s = self.robot_r*1.8
                self.ax.add_patch(Rectangle((sx-s/2, sy-s/2), s, s,
                                            facecolor=face, edgecolor=edge,
                                            linewidth=self.annotate_edge_width, zorder=4))
                self.ax.text(sx, sy, label,
                            color=text_c, fontsize=self.annotate_fontsize,
                            ha="center", va="center", zorder=5,
                            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
            else:
                self.ax.text(sx, sy, label,
                            color=text_c, fontsize=self.annotate_fontsize,
                            ha="center", va="center", zorder=4,
                            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])

    def _find_robot(self, model):
        if hasattr(model, "robot"): return model.robot
        if hasattr(model, "schedule") and hasattr(model.schedule, "agents"):
            for ag in model.schedule.agents:
                if getattr(ag, "type", None) == 3: return ag
        return None


   # =============== NEW: Mesh helpers & drawer ===============
    def _tri_centroid(self, tri):
        (x0,y0),(x1,y1),(x2,y2) = tri
        return ((x0+x1+x2)/3.0, (y0+y1+y2)/3.0)

    def _draw_mesh(self, model):
        """
        인접한 삼각형끼리 facecolor를 번갈아 칠해 시각화.
        - source: model.pure_mesh / model.mesh_list
        - adjacency: model.adjacent_mesh (dict: tri -> [neighbors])
        """
        tris = []
        if hasattr(model, "pure_mesh") and self.mesh_use_only_pure:
            tris = list(getattr(model, "pure_mesh", []))
        elif hasattr(model, "mesh_list"):
            tris = list(getattr(model, "mesh_list", []))

        if not tris:
            return

        adj = getattr(model, "adjacent_mesh", {})
        visited = set()
        color_side = {}  # tri -> 0/1

        # 연결성분 단위로 BFS해서 0/1 교차 배정
        for root in tris:
            if root in visited: continue
            color_side[root] = 0
            q = [root]
            visited.add(root)
            while q:
                cur = q.pop(0)
                neighs = adj.get(cur, [])
                for nb in neighs:
                    if nb not in tris:   # 일부 obstacle_mesh 등 제외 시 필터
                        continue
                    if nb not in visited:
                        color_side[nb] = 1 - color_side[cur]
                        visited.add(nb)
                        q.append(nb)

        # 그리기
        for tri in tris:
            face = self.mesh_face_a if color_side.get(tri, 0) == 0 else self.mesh_face_b
            patch = MplPolygon(np.asarray(tri, float), closed=True,
                               facecolor=face, edgecolor=self.mesh_edge_color,
                               linewidth=self.mesh_edge_width, alpha=self.mesh_alpha,
                               antialiased=True, joinstyle="miter", zorder=1.2)
            self.ax.add_patch(patch)


    # ========== NEW: Agent <-> Exit corrected distance ==========
    def _grid_point_mesh(self, model, p):
        """
        (정수 격자 좌표) -> 포함 삼각형 tri 튜플 반환. 없으면 None.
        """
        if hasattr(model, "match_grid_to_mesh"):
            key = (int(round(p[0])), int(round(p[1])))
            return model.match_grid_to_mesh.get(key)
        return None

    def _next_mesh_step(self, model, a, b):
        """
        next_vertex_matrix[a][b] 안전 접근
        """
        nvm = getattr(model, "next_vertex_matrix", None)
        if nvm is None: return None
        return nvm.get(a, {}).get(b, None)

    def _portal_midpoints_path_length(self, model, p1, p2):
        """
        '빠른 임시 보정' 방식: 삼각형 경로를 따라
        - 인접 삼각형 공유변(포털)의 '중점'들로 폴리라인 구성
        - p1 -> 첫 중점 -> ... -> 마지막 중점 -> p2 길이 합
        실패시 유클리드로 폴백
        """
        import math

        # 1) 점이 속한 삼각형
        m1 = self._grid_point_mesh(model, p1)
        m2 = self._grid_point_mesh(model, p2)
        if m1 is None or m2 is None:
            print("소속 메쉬가 없음")
            return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

        if m1 == m2:
            return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

        # 2) 삼각형 경로 복원
        nxt = getattr(model, "next_vertex_matrix", None)
        if nxt is None or nxt.get(m1, {}).get(m2) is None:
            return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

        path_meshes = [m1]
        cur = m1
        guard = 0
        N = max(1, len(getattr(model, "mesh_list", [])))
        while cur != m2 and guard <= N:
            cur = nxt[cur][m2]
            if cur is None:
                return math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            path_meshes.append(cur)
            guard += 1
        if path_meshes[-1] != m2:
            return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

        # 3) 포털 중점들
        portal_mids = []
        for a, b in zip(path_meshes, path_meshes[1:]):
            common = list(set(a) & set(b))
            if len(common) == 2:
                (x1, y1), (x2, y2) = common[0], common[1]
                portal_mids.append(((x1+x2)/2.0, (y1+y2)/2.0))
            else:
                # 폴백: 다음 삼각형의 센트로이드
                portal_mids.append(self._tri_centroid(b))

        if not portal_mids:
            return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

        # 4) 총 길이
        dist = 0.0
        # p1 -> 첫 포털
        dist += math.hypot(portal_mids[0][0]-p1[0], portal_mids[0][1]-p1[1])
        # 포털 간
        for s, t in zip(portal_mids, portal_mids[1:]):
            dist += math.hypot(t[0]-s[0], t[1]-s[1])
        # 마지막 포털 -> p2
        dist += math.hypot(p2[0]-portal_mids[-1][0], p2[1]-portal_mids[-1][1])
        return dist

    def _draw_agent_exit_distances(self, model):
        """
        각 살아있는 crowd agent에 대해, 메쉬 기반 보정 경로(포털 중점 폴리라인)를
        실제로 화면에 그린다. 길이 라벨은 해당 폴리라인 길이.
        실패 시 직선으로 폴백.
        """
        import math
        import numpy as np
        import matplotlib.patheffects as pe

        # --- exits 수집 (중심점)
        exits = []
        if hasattr(model, "exit_point") and model.exit_point:
            exits = [tuple(e) for e in model.exit_point]
        elif hasattr(model, "exit_list") and model.exit_list:
            for poly in model.exit_list:
                p = np.asarray(poly, float)
                exits.append((float(np.mean(p[:,0])), float(np.mean(p[:,1]))))
        if not exits:
            return

        # --- 대상 agent 수집
        crowds = getattr(model, "crowds", [])
        alive = [ag for ag in crowds
                if not getattr(ag, "dead", False)
                and getattr(ag, "type", 0) in (0,1,2)]
        if not alive:
            return
        alive = alive[:getattr(self, "distance_max_agents", 9999)]

        # --- 내부 헬퍼: 점→삼각형, 경계 nudge, 폴리라인 생성
        def _nudge(pt):
            x, y = float(pt[0]), float(pt[1])
            W, H = float(self.W), float(self.H)
            eps = 0.4
            if abs(x-0.0) < 0.51: x = 0.0 + eps
            if abs(x-W  ) < 0.51: x = W   - eps
            if abs(y-0.0) < 0.51: y = 0.0 + eps
            if abs(y-H  ) < 0.51: y = H   - eps
            return (x, y)

        mgm = getattr(model, "match_grid_to_mesh", {})
        def _point_mesh(p):
            x0 = int(round(p[0])); y0 = int(round(p[1]))
            tri = mgm.get((x0, y0))
            if tri is not None: return tri
            # 주변 반경 1..3 탐색
            for r in (1,2,3):
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        if dx==0 and dy==0: continue
                        tri = mgm.get((x0+dx, y0+dy))
                        if tri is not None: return tri
            # 실패
            return None

        def _tri_centroid(tri):
            (x0,y0),(x1,y1),(x2,y2) = tri
            return ((x0+x1+x2)/3.0, (y0+y1+y2)/3.0)

        def _polyline_and_length(p_start, p_goal):
            """메쉬 경로를 포털 중점 폴리라인으로 변환. 실패 시 직선."""
            p1, p2 = _nudge(p_start), _nudge(p_goal)
            m1, m2 = _point_mesh(p1), _point_mesh(p2)

            def straight():
                L = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
                return [p1, p2], L

            if m1 is None or m2 is None:
                return straight()
            if m1 == m2:
                return straight()

            nxt = getattr(model, "next_vertex_matrix", None)
            if nxt is None or nxt.get(m1, {}).get(m2) is None:
                return straight()

            # 삼각형 경로 복원 (가드 포함)
            path_meshes = [m1]
            cur = m1
            guard = 0
            N = max(1, len(getattr(model, "mesh_list", [])))
            while cur != m2 and guard <= N:
                cur = nxt[cur][m2]
                if cur is None:
                    return straight()
                path_meshes.append(cur); guard += 1
            if path_meshes[-1] != m2:
                return straight()

            # 포털 중점들
            mids = []
            for a, b in zip(path_meshes, path_meshes[1:]):
                common = list(set(a) & set(b))
                if len(common) == 2:
                    (x1,y1),(x2,y2) = common
                    mids.append(((x1+x2)/2.0, (y1+y2)/2.0))
                else:
                    mids.append(_tri_centroid(b))

            if not mids:
                return straight()

            pts = [p1] + mids + [p2]
            L = 0.0
            for s, t in zip(pts, pts[1:]):
                L += math.hypot(t[0]-s[0], t[1]-s[1])
            return pts, L

        # --- exit 선택 + 그리기
        fixed = (getattr(self, "distance_exit_mode", "nearest") == "fixed")
        fixed_idx = max(0, min(len(exits)-1, getattr(self, "distance_exit_index", 0)))

        for ag in alive:
            ax, ay = float(ag.xy[0]), float(ag.xy[1])

            # 대상 출구 선택: nearest(보정거리) 또는 fixed
            if fixed:
                tgt = exits[fixed_idx]
                pts, L = _polyline_and_length((ax, ay), tgt)
            else:
                # 모든 출구 후보 중 보정거리 최단 선택
                best = None
                for i, c in enumerate(exits):
                    p, d = _polyline_and_length((ax, ay), c)
                    if best is None or d < best[1]:
                        best = (p, d, i, c)
                pts, L, _, tgt = best

            xs, ys = zip(*pts)
            self.ax.plot(xs, ys,
                        color=self.distance_line_color if hasattr(self, "distance_line_color") else "#ff7f0e",
                        linewidth=self.distance_line_width if hasattr(self, "distance_line_width") else 1.2,
                        alpha=self.distance_line_alpha if hasattr(self, "distance_line_alpha") else 0.85,
                        zorder=3.6
                        )
            # (선택) 포털 중점 점표시
            if len(pts) > 2:
                self.ax.scatter(xs[1:-1], ys[1:-1], s=9, zorder=3.7,
                                color=self.distance_line_color if hasattr(self, "distance_line_color") else "#ff7f0e",
                                alpha=0.9, linewidths=0)

            # ▼▼▼ HERE: 라벨을 에이전트 머리(원 위쪽)에 표시
            label_text = f"{L:.1f}"

            # --- 거리 정규화
            Lmin = getattr(self, "distance_color_min", 0.0)
            Lmax = getattr(self, "distance_color_max", 100.0)
            L_clamped = max(Lmin, min(L, Lmax))
            t = (L_clamped - Lmin) / (Lmax - Lmin + 1e-8)

            mode = getattr(self, "distance_label_colormap", "gradient")

            if mode == "static":
                # 항상 동일 색
                dist_color = getattr(self, "distance_label_color", "black")

            elif mode == "gradient":
                # 단순 RGB 보간 (초록→빨강)
                r_val = int(255 * t)
                g_val = int(255 * (1 - t))
                b_val = 0
                dist_color = f"#{r_val:02x}{g_val:02x}{b_val:02x}"

            elif mode == "colormap":
                # matplotlib 컬러맵 사용 (예: viridis, inferno 등)
                import matplotlib.cm as cm
                cmap_name = getattr(self, "distance_label_cmap_name", "viridis")
                cmap = cm.get_cmap(cmap_name)
                rgba = cmap(t)
                dist_color = (rgba[0], rgba[1], rgba[2])  # RGB 튜플 (0~1)

            else:
                # 안전 폴백
                dist_color = getattr(self, "distance_label_color", "black")

            # --- 머리 위 위치
            r = getattr(self, "crowd_r", 0.35)
            offset_y = r * 1.4
            lx, ly = ax, ay + offset_y
            if hasattr(self, "H"):
                ly = min(self.H - 0.5, ly)

            # --- 텍스트 출력
            self.ax.text(lx, ly, label_text,
                        fontsize=self.distance_label_fontsize if hasattr(self, "distance_label_fontsize") else 8,
                        color=dist_color,
                        ha="center", va="bottom", zorder=5.0)


    def _agent_heading_col(self, t: int, fallback: str):
        c = self.agent_heading_color
        if c is None:
            return fallback
        if isinstance(c, dict):
            return c.get(t, fallback)
        return str(c)

    def _draw_vision(self, model):
        """
        사전계산된 vision_atlas에서 폴리곤을 '조회'해서 연하게 칠한다.
        - 장애물 버전이 맞지 않거나 atlas가 없으면 아무 것도 안 함.
        - 반경은 센서 반경(sensor_R)로: 기본은 3 + agent.vision_radius
        """
        if not hasattr(model, "vision_atlas"):
            return
        atlas = model.vision_atlas
        obs_ver = getattr(model, "obstacles_version", 0)

        # 에이전트 시야
        if self.show_agent_vision and hasattr(model, "crowds"):
            for ag in (model.crowds or []):
                if self.hide_dead and getattr(ag, "dead", False):
                    continue
                # 센서 반경(사전계산 때 등록한 값과 동일해야 함)
                R = float(getattr(ag, "vision_radius", 0.0))
                poly = atlas.polygon_at(float(ag.xy[0]), float(ag.xy[1]), R, obs_ver)
                if getattr(poly, "is_empty", True):
                    continue
                coords = list(poly.exterior.coords)
                self.ax.add_patch(
                    MplPolygon(
                        coords, closed=True,
                        facecolor=self.agent_vision_color,
                        edgecolor=to_rgba(self.agent_vision_color, self.vision_edge_alpha),  # ← 수정
                        linewidth=0.6,
                        alpha=self.vision_alpha,
                        zorder=1
                    )
                )

        # 로봇 시야
        if self.show_robot_vision:
            rb = self._find_robot(model)
            if rb is not None:
                Rr = float(getattr(rb, "vision_radius", 0.0))
                poly_r = atlas.polygon_at(float(rb.xy[0]), float(rb.xy[1]), Rr, obs_ver)
                if getattr(poly_r, "is_empty", True):
                    return
                coords_r = list(poly_r.exterior.coords)
                self.ax.add_patch(
                    MplPolygon(
                        coords_r, closed=True,
                        facecolor=self.robot_vision_color,
                        edgecolor=to_rgba(self.robot_vision_color, self.vision_edge_alpha),  # ← 수정
                        linewidth=0.8,
                        alpha=self.vision_alpha,
                        zorder=1
                    )
                )