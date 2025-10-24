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
        robot_radius: float = 0.45,
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
        trail_target: str = "both",                  # "none" | "crowd" | "robot" | "both"
        trail_style: str = "fade",                   # "fade" | "persist"
        max_trail: int = 40,
        crowd_trail_alpha: float = 0.55,
        robot_trail_alpha: float = 0.7,

        # ===== Heading arrows =====
        show_agent_heading: bool = False,
        show_robot_heading: bool = True,
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
        robot_image_rotate_by_velocity: bool = True,

        # ===== Robot path annotations =====
        annotate_robot_path: bool = True,            # 표기 켜기/끄기
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
        vision_edge_alpha: float = 0.35
        
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
        # ===== Buffers =====
        self.trails_crowd: Dict[int, List[Tuple[float,float]]] = {}
        # 로봇은 step 함께 저장: (x, y, step)
        self.trails_robot: Dict[int, List[Tuple[float,float,int]]] = {}

        # ===== Matplotlib fig/ax =====
        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=self.dpi)
        self._setup_axes()

    # ====================== Public ======================
    def draw(self, model, step: Optional[int] = None) -> np.ndarray:
        """현재 상태를 그림. step을 넘기면 로봇 궤적 번호에 절대 스텝을 기록."""
        self.ax.clear()
        self._setup_axes()

        if self.draw_outer_wall_flag:
            self._draw_outer_wall()

        self._draw_obstacles(getattr(model, "obstacles", []))

        self._draw_vision(model)

        self._draw_exits(model)
        self._draw_crowds(getattr(model, "crowds", []))
        self._draw_robot(self._find_robot(model), step=step)

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
        if t in self.crowd_colors: return self.crowd_colors[t]
        return self.crowd_colors.get("else", "#59a14f")

    def _draw_crowds(self, crowds):
        if not crowds: return
        draw_trail = self.trail_target in ("crowd", "both")
        for ag in crowds:
            t = getattr(ag, "type", None)
            if t in self.exclude_types: continue
            if self.hide_dead and getattr(ag, "dead", False): continue

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


                # self._arrow(
                #     x, y, vx, vy,
                #     scale=self.agent_heading_scale,
                #     color=head_col,
                #     alpha=0.9,
                #     linewidth=self.agent_heading_linewidth,
                #     mutation_scale=self.agent_heading_mutation_scale
                # )

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