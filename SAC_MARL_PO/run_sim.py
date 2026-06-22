#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Crowd Simulation (GUI)
- SIM_TIMESTEP = 0.25 s
- 자동 레이아웃: world (width, height)를 좌측 가용 화면에 '비율 유지'로 맞춰 스케일
- Pause/Resume, Step, Reset, Speed, Map ID textbox, Agents stepper
- 에피소드 종료 시 자동 재시작
- stride/downsampling 없음 (매 프레임 렌더)
"""

import os, math, time, sys, json
import numpy as np
import pygame

from model import FightingModel
from continuous_renderer import ContinuousRenderer
from config import *

# =========================
# 화면/월드 설정
# =========================
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 900     # 창 크기
PANEL_RIGHT_WIDTH = 300                     # 우측 패널 고정 폭
PADDING = 10                                # 좌/우/상/하 패딩
ROBOT_NUM = 3
SHOW_STATE_IMAGES = True
STATE_PREVIEW_SIZE = 100
STATE_PREVIEW_GAP = 8

SIM_TIMESTEP = 0.25
TARGET_RENDER_FPS = 10
SPEED_MIN, SPEED_MAX = 0.25, 16.0
MAX_STEPS = 3000
MAX_ACCUM_STEPS = 12
MAX_SUBSTEPS_PER_FRAME_HARD = 6
ADAPTIVE_SUBSTEP_INIT = 4

ROBOT_CONTROL_MODE = "RL"   # "RL", "Human"
ROBOT_VERSION_FOR_MODEL = "Q"
MODEL_NAME = "MARL_10000ep_3maps.pth"

USE_CONTINUOUS_RENDERER = True
def make_renderer(world_w, world_h):
    return ContinuousRenderer(
        world_size=(float(world_w), float(world_h)),
        crowd_colors={0:"#ffa500",1:"#4e79a7",2:"#4e79a7"},
        robot_color="#e15759",
        show_agent_heading=False,
        show_robot_heading=False,
        robot_heading_scale=3,
        trail_target="none",
        trail_style="fade",
        max_trail=2000,
        single_color_edges=True,
        exit_size=5.0,
        snap_exit_to_boundary=True,
    )


WHITE=(255,255,255)
BLACK=(0,0,0)
GREY=(210,210,210)
DARK=(40,40,42)
ACCENT=(18,136,255)



# ====== 간단 GUI 위젯 ======
class UIButton:
    def __init__(self, rect, label, on_click, font, bg=(240,240,240), fg=DARK):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.font = font
        self.bg = bg
        self.fg = fg
        self.hover=False
    def draw(self, surf):
        color = (248,248,248) if self.hover else self.bg
        pygame.draw.rect(surf, color, self.rect, border_radius=8)
        pygame.draw.rect(surf, (200,200,200), self.rect, 1, border_radius=8)
        txt = self.font.render(self.label() if callable(self.label) else self.label, True, self.fg)
        surf.blit(txt, txt.get_rect(center=self.rect.center))
    def handle(self, e):
        if e.type==pygame.MOUSEMOTION: self.hover=self.rect.collidepoint(e.pos)
        elif e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and self.rect.collidepoint(e.pos):
            self.on_click()

class UISlider:

    def __init__(self, rect, vmin, vmax, value, on_change):
        self.rect = pygame.Rect(rect); self.vmin=vmin; self.vmax=vmax
        self.value=float(value); self.on_change=on_change; self.drag=False

    def draw(self, surf):
        track = self.rect.inflate(0,-14); track.centery=self.rect.centery
        pygame.draw.rect(surf,(230,230,230),track,border_radius=6)
        pygame.draw.rect(surf,(200,200,200),track,1,border_radius=6)
        t=(self.value-self.vmin)/(self.vmax-self.vmin)
        knob_x = track.left + int(t*track.width)
        pygame.draw.circle(surf, ACCENT, (knob_x,track.centery), 8)
        pygame.draw.circle(surf, (255,255,255), (knob_x,track.centery), 8, 2)

    def handle(self, e):
        def set_from_x(mx):
            track = self.rect.inflate(0,-14); track.centery=self.rect.centery
            t = (mx-track.left)/max(1,track.width); t=max(0,min(1,t))
            self.value = self.vmin + t*(self.vmax-self.vmin); self.on_change(self.value)
        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1 and self.rect.collidepoint(e.pos):
            self.drag=True; set_from_x(e.pos[0])
        elif e.type==pygame.MOUSEBUTTONUP and e.button==1: 
            self.drag=False
        elif e.type==pygame.MOUSEMOTION and self.drag: 
            set_from_x(e.pos[0])

class UIStepper:
    def __init__(self,label,x,y,w,value,step,on_change,font,min_val=0):
        self.label=label; 
        self.value=int(value); 
        self.step=int(step)
        self.on_change=on_change; 
        self.font=font; 
        self.min_val=min_val
        self.rect_label=pygame.Rect(x,y,w,20)
        self.btn_minus=pygame.Rect(x,y+20,32,26)
        self.box=pygame.Rect(x+36,y+20,w-72,26)
        self.btn_plus=pygame.Rect(x+w-32,y+20,32,26)

    def draw(self,s):
        s.blit(self.font.render(self.label,True,DARK),(self.rect_label.left,self.rect_label.top))

        for btn,sym in ((self.btn_minus,"-"),(self.btn_plus,"+")):
            pygame.draw.rect(s,(242,242,242),btn,border_radius=6)
            pygame.draw.rect(s,(200,200,200),btn,1,border_radius=6)
            s.blit(self.font.render(sym,True,DARK),btn.move(10,2))

        pygame.draw.rect(s,(250,250,250),self.box,border_radius=6)
        pygame.draw.rect(s,(200,200,200),self.box,1,border_radius=6)
        s.blit(self.font.render(str(self.value),True,DARK), self.font.render(str(self.value),True,DARK).get_rect(center=self.box.center))
    
    def handle(self,e):
        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
            if self.btn_minus.collidepoint(e.pos):
                self.value=max(self.min_val,self.value-self.step)
                self.on_change(self.value)
            elif self.btn_plus.collidepoint(e.pos):
                self.value=self.value+self.step
                self.on_change(self.value)

class UIInputNumber:
    def __init__(self,label,x,y,w,value,on_change,font,min_val=0):
        self.label=label
        self.font=font
        self.text=str(int(value))
        self.rect_label=pygame.Rect(x,y,w,20)
        self.box=pygame.Rect(x,y+20,w,26)
        self.focus=False
        self.on_change=on_change
        self.min_val=min_val
        self._blink_t=0.0
        self._show_caret=True

    def draw(self,s):
        s.blit(self.font.render(self.label,True,DARK),(self.rect_label.left,self.rect_label.top))
        pygame.draw.rect(s,(250,250,250),self.box,border_radius=6)
        pygame.draw.rect(s,(120,180,255) if self.focus else (200,200,200),self.box,1,border_radius=6)
        txtsurf=self.font.render(self.text,True,DARK)
        r=txtsurf.get_rect(midleft=(self.box.left+8,self.box.centery))
        s.blit(txtsurf,r)
        if self.focus:
            self._blink_t+=1/60
            if self._blink_t>=0.5: 
                self._blink_t=0.0
            self._show_caret=not self._show_caret
            if self._show_caret:
                cx=r.right+2
                pygame.draw.line(s,DARK,(cx,self.box.top+5),(cx,self.box.bottom-5),1)

    def handle(self,e):
        if e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
            self.focus=self.box.collidepoint(e.pos)
        elif e.type==pygame.KEYDOWN and self.focus:
            if e.key==pygame.K_RETURN:
                if self.text.strip()=="": self.text=str(self.min_val)
                try: self.on_change(max(self.min_val,int(self.text))); 
                except ValueError: pass
                self.focus=False
            elif e.key==pygame.K_BACKSPACE:
                self.text=self.text[:-1]
                if self.text.strip()!="":
                    try: self.on_change(int(self.text))
                    except ValueError: pass
            else:
                ch=e.unicode
                if ch.isdigit():
                    self.text+=ch
                    try: self.on_change(int(self.text))
                    except ValueError: pass

# ====== 레이아웃 계산 (여기가 핵심!) ======
# 좌측 가용 영역 안에 world(width,height) 비율을 유지하며 딱 맞게 배치
MAP_VIEW_W = MAP_VIEW_H = 0
MAP_OFFSET_X = MAP_OFFSET_Y = 0

def compute_map_viewport(world_w:int, world_h:int):
    """우측 패널/패딩을 제외한 좌측 영역에 world 비율로 맞춰 그릴 뷰포트 계산."""
    global MAP_VIEW_W, MAP_VIEW_H, MAP_OFFSET_X, MAP_OFFSET_Y
    avail_w = max(1, SCREEN_WIDTH - PANEL_RIGHT_WIDTH - 2*PADDING)
    avail_h = max(1, SCREEN_HEIGHT - 2*PADDING)
    # world 비율 유지해 최대 크기로 맞춤
    s = min(avail_w / max(1, world_w), avail_h / max(1, world_h))
    MAP_VIEW_W = max(1, int(round(world_w * s)))
    MAP_VIEW_H = max(1, int(round(world_h * s)))
    # 좌측 여백 + 세로 중앙 정렬
    MAP_OFFSET_X = PADDING + (avail_w - MAP_VIEW_W)//2
    MAP_OFFSET_Y = PADDING + (avail_h - MAP_VIEW_H)//2

def update_layout_from_env(env):
    """
    env.width / env.height는 FightingModel이
    map_infos/map_{map_num}.json에서 읽어온 값이어야 한다.
    없으면 fallback하지 않고 에러를 낸다.
    """
    if not hasattr(env, "width") or not hasattr(env, "height"):
        raise AttributeError("[GUI] env must have width and height.")

    w = int(env.width)
    h = int(env.height)

    if w <= 0 or h <= 0:
        raise ValueError(f"[GUI] invalid env size: width={w}, height={h}")

    compute_map_viewport(w, h)

# ====== ENV / POLICY ======
def create_env_and_renderer(map_id: int, n_agents: int):
    """
    FightingModel이 map_id 기준으로 map_infos/map_{map_id}.json에서
    width/height를 직접 읽는다.

    GUI는 생성된 env.width/env.height를 사용해서 renderer와 viewport를 맞춘다.
    """
    env = FightingModel(
        number_agents=n_agents,
        model_num=map_id,
        robot=ROBOT_VERSION_FOR_MODEL,
        robot_num=THE_NUMBER_OF_ROBOTS
    )

    world_w = int(env.width)
    world_h = int(env.height)

    renderer = make_renderer(world_w, world_h) if USE_CONTINUOUS_RENDERER else None
    update_layout_from_env(env)

    return env, renderer

def policy_fn(env: FightingModel, dt: float) -> np.ndarray:
    try:
        rx, ry = env.robot.xy
        if hasattr(env,"exit_list") and env.exit_list:
            best=(None,1e9)
            for poly in env.exit_list:
                try: cx,cy = np.mean(np.array(poly),axis=0)
                except Exception: cx,cy = poly.centroid.x, poly.centroid.y
                d=(cx-rx)**2+(cy-ry)**2
                if d<best[1]: best=((cx,cy), d)
            (cx,cy),_ = best; vx,vy = cx-rx, cy-ry
            n = math.hypot(vx,vy) or 1.0
            return np.array([1.5*vx/n, 1.5*vy/n], np.float32)
    except Exception:
        pass
    return np.array([0.0,0.0], np.float32)

def alive_agents(env)->int:
    try: return int(env.alived_agents())
    except Exception: return 0

def steps_per_second(speed:float)->float: return speed / SIM_TIMESTEP
def fmt_sim_time(step_count:int)->str:
    sec = step_count*SIM_TIMESTEP; m=int(sec//60); s=int(sec%60)
    return f"{m:02d}:{s:02d} ({sec:.1f}s)"

def state_img_to_surface(img, size: int):
    """
    img:
      - ego_f / glob_f: (H, W), float32, 0~1
      - ego_state / glob_state: (C, H, W)일 수도 있음
    return:
      - pygame Surface
    """
    surf = pygame.Surface((size, size))
    surf.fill((235, 235, 235))

    if img is None:
        return surf

    arr = np.asarray(img)

    # stack state가 들어온 경우 마지막 frame만 시각화
    if arr.ndim == 3:
        arr = arr[-1]

    if arr.ndim != 2:
        return surf

    arr = arr.astype(np.float32)
    arr = np.flipud(arr)
    # 0~1이면 0~255로 변환
    if arr.size > 0 and arr.max() <= 1.0:
        arr = arr * 255.0

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # grayscale -> RGB
    rgb = np.repeat(arr[:, :, None], 3, axis=2)

    # pygame은 (W, H, 3) 형태를 기대하므로 transpose
    img_surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    img_surf = pygame.transform.scale(img_surf, (size, size))

    return img_surf
def world_to_view_px(env, x, y):
    """
    world 좌표를 현재 MAP_VIEW 안의 pixel 좌표로 변환한다.
    반환값은 screen 전체 좌표가 아니라, MAP_VIEW 내부 상대좌표이다.
    """
    px = int(np.clip(x / max(1e-9, env.width) * MAP_VIEW_W, 0, MAP_VIEW_W - 1))
    py = int(np.clip(y / max(1e-9, env.height) * MAP_VIEW_H, 0, MAP_VIEW_H - 1))

    # ContinuousRenderer에서 y축이 뒤집혀 보이면 아래 줄로 바꿔라.
    py = MAP_VIEW_H - 1 - py

    return px, py


def draw_robot_vision_dark_overlay(screen, env, alpha=150):
    """
    메인 화면에서 robot vision 밖을 어둡게 표시한다.
    - 로봇 시야 범위 자체는 바꾸지 않는다.
    - env.robot_visibility_polygon()을 그대로 사용한다.
    - 여러 로봇이면 모든 robot vision union 영역은 밝게 유지된다.
    """
    if BLACK_SHEEP_WALL:
        return

    if not hasattr(env, "robot_visibility_polygon"):
        return

    # BLACK_SHEEP_WALL == True일 때는 full observation이므로 overlay를 안 씌우고 싶으면 return
    if not getattr(env, "is_partial_crowd_observation", lambda: False)():
        return

    # 1) 전체 맵 위에 어두운 막을 만든다.
    overlay = pygame.Surface((MAP_VIEW_W, MAP_VIEW_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, alpha))

    # 2) 로봇들이 볼 수 있는 영역은 overlay에서 투명하게 뚫는다.
    for ridx, rb in enumerate(getattr(env, "robots", [])):
        poly = env.robot_visibility_polygon(ridx, radius=ROBOT_VISION)
        if poly is None or poly.is_empty:
            continue

        coords = list(poly.exterior.coords)
        pts = [world_to_view_px(env, x, y) for x, y in coords]

        if len(pts) >= 3:
            pygame.draw.polygon(overlay, (0, 0, 0, 0), pts)

    # 3) 메인 맵 위치에 overlay를 덮는다.
    screen.blit(overlay, (MAP_OFFSET_X, MAP_OFFSET_Y))


def draw_robot_state_images(screen, env, x, y, panel_w, font_small, max_y):
    if not SHOW_STATE_IMAGES:
        return y

    title = font_small.render("Robot State Images", True, DARK)
    screen.blit(title, (x, y))
    y += 22

    img_size = STATE_PREVIEW_SIZE
    gap = STATE_PREVIEW_GAP

    label_ego = font_small.render("ego_f", True, (70, 70, 70))
    label_glob = font_small.render("glob_f", True, (70, 70, 70))

    ego_x = x
    glob_x = x + img_size + gap + 18

    for rb in getattr(env, "robots", []):
        if y + img_size + 42 > max_y:
            screen.blit(
                font_small.render("... more robots hidden", True, (120, 120, 120)),
                (x, y)
            )
            y += 18
            break

        ridx = getattr(rb, "robot_index", "?")

        row_title = font_small.render(f"Robot {ridx}", True, DARK)
        screen.blit(row_title, (x, y))
        y += 18

        ego_f = getattr(rb, "latest_ego_f", None)
        glob_f = getattr(rb, "latest_glob_f", None)

        ego_surf = state_img_to_surface(ego_f, img_size)
        glob_surf = state_img_to_surface(glob_f, img_size)

        screen.blit(ego_surf, (ego_x, y))
        screen.blit(glob_surf, (glob_x, y))

        pygame.draw.rect(screen, (180, 180, 180), (ego_x, y, img_size, img_size), 1)
        pygame.draw.rect(screen, (180, 180, 180), (glob_x, y, img_size, img_size), 1)

        screen.blit(label_ego, (ego_x, y + img_size + 2))
        screen.blit(label_glob, (glob_x, y + img_size + 2))

        y += img_size + 24

    # 간단한 intensity legend
    legend = [
        "0 black: empty",
        "50/255: wall/pad",
        "100/255: exit",
        "150/255: crowd",
        "255/255: robot",
    ]

    for line in legend:
        if y + 16 > max_y:
            break
        screen.blit(font_small.render(line, True, (90, 90, 90)), (x, y))
        y += 16

    return y

# =========================
# 메인
# =========================
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Interactive Crowd Simulation (Auto-fit Viewport)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18); font_small = pygame.font.SysFont("Arial", 16)

    # 상태

    paused=False
    step_once=False
    speed=1.0

    if MAP_NUM != -1:
        pending_map = MAP_NUM
    else:
        pending_map = np.random.choice(MAP_NUM_RANDOM)

    pending_agents= np.random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX+1)

    env, renderer = create_env_and_renderer(pending_map, pending_agents)

    if( ROBOT_CONTROL_MODE == "RL"):
        env.use_model(MODEL_NAME)
    
    curr_map, curr_agents = pending_map, pending_agents

    step_count=0
    last=time.perf_counter() 
    acc_steps=0.0
    last_render_ms=0.0
    target_frame_ms=1000.0/TARGET_RENDER_FPS
    substep_limit = ADAPTIVE_SUBSTEP_INIT

    # ─ GUI 배치 ─
    panel_x = SCREEN_WIDTH - PANEL_RIGHT_WIDTH + 12
    panel_w = PANEL_RIGHT_WIDTH - 24
    y=10
    title_rect = pygame.Rect(panel_x,y,panel_w,24); 
    y+=28
    BTN_H=28 
    GAP=6
    btn_w=(panel_w - GAP*2)//3

    def toggle_pause(): 
        nonlocal paused 
        paused=not paused
    btn_pause = UIButton((panel_x,y,btn_w,BTN_H), lambda:"▶ Resume" if paused else "⏸ Pause", toggle_pause, font_small)

    def do_step(): 
        nonlocal step_once; 
    def do_step():
        nonlocal step_once
        if paused: 
            step_once=True
    btn_step  = UIButton((panel_x+btn_w+GAP,y,btn_w,BTN_H), "Step ➜", do_step, font_small)
    
    def do_reset():
        nonlocal env, renderer, curr_map, curr_agents, step_count, acc_steps, paused

        env, renderer = create_env_and_renderer(pending_map, pending_agents)

        if ROBOT_CONTROL_MODE == "RL":
            env.use_model(MODEL_NAME)

        curr_map, curr_agents = pending_map, pending_agents
        step_count = 0
        acc_steps = 0.0
        paused = False
        
    btn_reset = UIButton((panel_x+(btn_w+GAP)*2,y,btn_w,BTN_H), "Reset ⟲", do_reset, font_small)
    y += BTN_H+10

    speed_label_rect = pygame.Rect(panel_x,y,panel_w,18)
    y += 18
    slider = UISlider((panel_x,y,panel_w,28), SPEED_MIN, SPEED_MAX, speed, on_change=lambda v: None)
    y += 34

    def on_map_change(v): 
        nonlocal pending_map
        pending_map=int(v)

    input_map = UIInputNumber("Map ID (*)", panel_x, y, panel_w, pending_map, on_change=on_map_change, font=font_small, min_val=0) 
    y += 52
    
    def on_agents_change(v): 
        nonlocal pending_agents
        pending_agents=int(v)

    step_agents = UIStepper("Agents (*)", panel_x, y, panel_w, pending_agents, 5, on_change=on_agents_change, font=font_small, min_val=0)
    y += 52

    widgets=[btn_pause, btn_step, btn_reset, slider, input_map, step_agents]

    running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT: 
                running = False
            elif e.type==pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q): 
                running = False
            elif e.type==pygame.KEYDOWN and e.key==pygame.K_SPACE: 
                paused = not paused
            elif e.type==pygame.KEYDOWN and e.key==pygame.K_RIGHT:
                if paused: 
                    step_once = True
            elif e.type==pygame.KEYDOWN and e.key==pygame.K_UP:
                speed = min(SPEED_MAX,speed*2.0)
                slider.value = speed
            elif e.type==pygame.KEYDOWN and e.key==pygame.K_DOWN:
                speed = max(SPEED_MIN,speed/2.0)
                slider.value = speed
            for w in widgets:
                if hasattr(w,"handle"): w.handle(e)

        speed=slider.value

        # 누적을 'step' 단위로
        now=time.perf_counter()
        dt=now-last
        last=now
        if not paused or step_once:
            acc_steps += (dt*speed)/SIM_TIMESTEP
            acc_steps = min(acc_steps, MAX_ACCUM_STEPS)

        # 렌더 타임 기준으로 프레임당 서브스텝 한도 적응
        if last_render_ms > target_frame_ms*0.95:
            substep_limit = max(1, substep_limit-1)
        elif last_render_ms < target_frame_ms*0.60:
            substep_limit = min(MAX_SUBSTEPS_PER_FRAME_HARD, substep_limit+1)

        did_step=False
        substeps=0
        while (acc_steps>=1.0) and (not paused or step_once) and (substeps<substep_limit):

            env.step()
            step_count+=1; acc_steps-=1.0; substeps+=1; did_step=True
            if step_once: 
                step_once=False
                break

        # 자동 재시작
        alive = alive_agents(env)
        if (alive <= 0) or (step_count >= MAX_STEPS):
            env, renderer = create_env_and_renderer(curr_map, curr_agents)

            if ROBOT_CONTROL_MODE == "RL":
                env.use_model(MODEL_NAME)

            step_count = 0
            acc_steps = 0.0
            paused = False
            alive = alive_agents(env)

        # ==== DRAW (항상 화면 안에 들어오도록) ====
        screen.fill(WHITE)

        # 맵 영역 (뷰포트 크기/위치가 자동으로 계산되어 있음)
        if USE_CONTINUOUS_RENDERER and renderer is not None:
            t0=time.perf_counter()
            rgb = renderer.draw(env, step=step_count)
            surf = pygame.surfarray.make_surface(np.transpose(rgb,(1,0,2)))
            if (surf.get_width(), surf.get_height())!=(MAP_VIEW_W, MAP_VIEW_H):
                surf = pygame.transform.scale(surf, (MAP_VIEW_W, MAP_VIEW_H))
            screen.blit(surf, (MAP_OFFSET_X, MAP_OFFSET_Y))
            draw_robot_vision_dark_overlay(screen, env, alpha=150)
            last_render_ms = (time.perf_counter()-t0)*1000.0

        # 우측 패널
        pygame.draw.rect(screen,(248,248,248),(SCREEN_WIDTH-PANEL_RIGHT_WIDTH,0,PANEL_RIGHT_WIDTH,SCREEN_HEIGHT))
        pygame.draw.line(screen,GREY,(SCREEN_WIDTH-PANEL_RIGHT_WIDTH,0),(SCREEN_WIDTH-PANEL_RIGHT_WIDTH,SCREEN_HEIGHT),2)
        title = pygame.font.SysFont("Arial",20).render("Simulation Control",True,DARK)
        screen.blit(title,(title_rect.left,title_rect.top))

        # 위젯
        screen.blit(font_small.render("Speed",True,DARK),(speed_label_rect.left,speed_label_rect.top))
        for w in widgets:
            if hasattr(w,"draw"): w.draw(screen)

        status_reserved_h = 5 * 18 + 8
        state_panel_top = y + 10
        state_panel_bottom = SCREEN_HEIGHT - status_reserved_h - 12

        draw_robot_state_images(
            screen = screen,
            env = env,
            x = panel_x,
            y = state_panel_top,
            panel_w = panel_w,
            font_small = font_small,
            max_y = state_panel_bottom
        )

        # 설명(Status) — 패널 하단 고정
        info=[
            f"Paused:{paused}  Speed:{speed:.2f}x  SPS:{steps_per_second(speed):.1f}  SubstepLimit:{substep_limit}",
            f"Step:{step_count}  Sim:{fmt_sim_time(step_count)}  Alive:{alive}",
            f"World:{int(env.width)}x{int(env.height)}  Viewport:{MAP_VIEW_W}x{MAP_VIEW_H}",
            f"LastRender:{last_render_ms:.1f} ms  Target:{(1000/TARGET_RENDER_FPS):.1f} ms   (Reset to apply Pending)",
            f"Pending(*) Map:{pending_map}  Agents:{pending_agents}    Keys: Space/→/↑/↓/ESC",
        ]
        lh=18; total_h=len(info)*lh+8; info_y=SCREEN_HEIGHT-total_h
        for line in info:
            screen.blit(font_small.render(line,True,(60,60,60)),(panel_x,info_y)); info_y+=lh

        pygame.display.flip()
        clock.tick(TARGET_RENDER_FPS)

    pygame.quit()

if __name__=="__main__":
    main()
