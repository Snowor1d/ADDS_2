#this source code requires Mesa==2.2.1
#^__^
from sim.core import Agent
import socket
import time 
import math
import numpy as np
import random
import copy
import sys 
from collections import deque
from heapq import heappush, heappop
from shapely.geometry import Point
from shapely.geometry import Polygon, MultiPolygon
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import *
import config as _config
from configs import decision_max_steps as _decision_max_steps

# How far choice_safe_mesh will look for a walkable mesh, in grid cells.
# Real building footprints need more than the original radius of two; see the
# comment at the fallback.
SAFE_MESH_SEARCH_RADIUS = 12


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

    
    
class CrowdAgent(Agent):
    """An agent that fights."""

    def __init__(self, unique_id, model, pos, type_agent): 
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.next_mesh = None
        self.past_mesh = None
        self.previous_mesh = None
        self.pos = pos
        self.behavior_probability = [random.gauss(0.9, 0.1), random.gauss(0.2, 0.1), random.gauss(0.1, 0.1)] #robot #동조 #myway
        self.robot_step = 0
        self.type = type_agent

        self.dead = False

        self.danger = 0
        self.previous_danger = 0

        self.drag = 0
        self.dead_count = 0
        self.buried = False
        self.previous_stage = []
        self.now_goal = [0,0]
        self.now_pointing_mesh = None
        self.robot_previous_goal = [0, 0]
        self.robot_initialized = 0
        self.direction = [0, 0]

        # print(isinstance(pos, tuple))
        self.xy = pos
        self.vel = [0, 0]
        self.acc = [0, 0]
        # self.mass = 3
        self.mass = np.random.normal(66, 4.16) # agent의 mass, 평균 66kg, 표준 편차 4.16kg
        if self.type == 3: # robot mass는 3으로 고정
            self.mass = 30

        self.desired_speed_a = np.random.normal(AGENT_SPEED_MEAN, 0.2) # agent의 desired_speed, 평균 1.5m/s, 표준 편차 0.2m/s

        self.is_effected_by_robot = 0
        self.blocked = False


        # Stable person-level propensity, separate from mode/context/form.
        self.compliance = random.uniform(*ROBOT_COMPLIANCE_PERSON_FACTOR_RANGE)
        # Robots noticed and still remembered: unique_id -> last step seen.
        # See _perceive_robots.
        self._known_robots = {}
        self.robot_lead_mode = None

        # Where this pedestrian is going while it knows nothing about the
        # hazard, and how long it stays when it gets there. See od.py.
        self._dwell_until = 0
        self._dwelling = False
        # Once outside the ground it believes dangerous: leave the crop or
        # resume trips, drawn once (see _belief_route_goal).
        self.post_safe_intent = None  # depart | continue
        # Heading for the way out of believed danger (M3 rule 1).
        self._escaping = False
        # Visibly responding this step: following a robot, leaving ground it
        # believes dangerous, leaving the crop, or going with a grounded
        # flow. Only this is a social cue to others (see _social_cue).
        self.responding = False
        # Bumped whenever hazard memory changes; plans made against an older
        # memory are checked again.
        self._memory_version = 0
        self._plan_version = -1
        self.outflow_reason = None
        self.ever_acted = False


        # ---- hazard awareness ----------------------------------------
        #
        # Awareness is a process, not a fact. The Protective Action Decision
        # Model has people move through receiving a cue, attending to it,
        # understanding it, and only then deciding to act, and they can stop
        # at any step. A boolean "knows about the hazard" collapses all of
        # that, and the step it collapses hardest is the one the robots can
        # actually act on.
        #
        #   unaware -> milling -> acting, or unaware -> nonresponsive
        #
        # docs/behavior_model_design.md (M1). Nonresponsive people were warned
        # and do not act on it themselves; a robot can still move them.
        #
        # `milling` is the interval of seeking confirmation before moving. It
        # is where most of the delay lives in real evacuations, and where a
        # robot's presence does its work: a robot cannot make somebody who is
        # already running run faster, but it can stop somebody standing still
        # from standing still.
        self.awareness = "unaware"
        self.cued_at = None            # step the first cue arrived
        self.act_after = None          # steps of milling still to serve
        self.cue_source = None         # "sensed" | "social" | "robot"

        # Where this pedestrian has personally sensed the hazard. Not the
        # zone: the zone is the simulator's knowledge, this is the
        # pedestrian's. Evacuation studies find people navigate on local,
        # route-level knowledge rather than a map of the place, and act on
        # what they can see from where they stand. So a pedestrian avoids the
        # spots it has sensed and can walk around a block straight into a face
        # of the same hazard it has never seen. That is the situation the
        # robots exist to prevent, and assuming global knowledge deletes it.
        self.hazard_memory = []

        # Which robot this pedestrian is currently following, or None. With a
        # team, "is being guided" is not enough: each robot's observation has
        # to count its own followers, not everyone else's as well.
        self.following_robot_id = None
        self.exit_belief = None       # unused under the hazard task
        self.life_time = 0
        self.body_radius = AGENT_BODY_RADIUS
        self.vision_radius = AGENT_VISION
        self.meeting_robot = 0


    def step(self) -> None:

        """Handles the step of the model dor each agent.
        Sets the flags of each agent during the simulation.
        """
        if not self.dead:
            self.life_time += 1
            
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

        # `dead` also marks a pedestrian who has left the cropped city.
        # Being outside the local hazard is not enough: people may stay,
        # continue a trip, or leave by a street mouth. Only those who have
        # actually left the crop cease to take simulation steps.
        


        self.move()


    def choice_safe_mesh(self, point):
        # The grid runs 0..width-1 by 0..height-1, so a query on the map edge
        # has no cell of its own. The original code special-cased exactly the
        # far corner; every other boundary point fell through and raised, which
        # real footprints hit constantly because their triangulation leaves
        # slivers along the edge whose centroids round onto it. Clamping is the
        # same intent applied to all four edges.
        x = min(max(int(round(point[0])), 0), self.model.width - 1)
        y = min(max(int(round(point[1])), 0), self.model.height - 1)
        point_grid = (x, y)
        while_checking = 0

        candidates = [(x+1,y+1), (x+1, y), (x, y+1), (x-1, y-1), (x-1, y), (x, y-1), (x+1, y-1), (x-1, y+1), (x-2, y), (x+2, y), (x, y-2), (x, y+2)]
        
        grid_to_mesh = self.model.match_grid_to_mesh
        pure = self.model.pure_mesh

        if (point_grid not in grid_to_mesh) or (grid_to_mesh[point_grid] not in pure):
            #print("다른 후보 찾기")
            #print("-")
            for c in candidates:
                if (c in grid_to_mesh) and (grid_to_mesh[c] in pure):
                    return grid_to_mesh[c]

            # The twelve fixed candidates above are enough on generated maps,
            # where obstacles are few and blocky. They are not enough on real
            # building footprints: those carry many more vertices, the
            # constrained triangulation produces correspondingly larger and
            # more irregular triangles, and a triangle's own centroid can land
            # in a grid cell whose mapped triangle is classified as an obstacle
            # mesh. Giving up there raises, the episode dies, and the worker
            # retries the same map forever.
            #
            # So widen the search in rings instead of stopping at radius two.
            # The result is the nearest walkable mesh, which is what every
            # caller wanted; the old behaviour is unchanged wherever the close
            # candidates already answered.
            for radius in range(3, SAFE_MESH_SEARCH_RADIUS + 1):
                best = None
                best_d2 = None
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        # Only the new ring, not the interior already searched.
                        if max(abs(dx), abs(dy)) != radius:
                            continue
                        c = (x + dx, y + dy)
                        mesh = grid_to_mesh.get(c)
                        if mesh is None or mesh not in pure:
                            continue
                        d2 = dx * dx + dy * dy
                        if best_d2 is None or d2 < best_d2:
                            best, best_d2 = mesh, d2
                if best is not None:
                    return best

            # Nothing walkable anywhere near. Returning the mesh that actually
            # contains the point keeps the caller working with a real triangle
            # rather than aborting the episode; it is blocked ground, which the
            # distance query will report as unreachable, and that is the
            # truthful answer for a point sealed inside a building.
            if point_grid in grid_to_mesh:
                return grid_to_mesh[point_grid]

            raise Exception(f"{x}, {y} 지점에서 오류 발생, safe mesh를 찾지 못했습니다")
        return grid_to_mesh[point_grid]
        



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


    def _neighbors(self, radius):
        # 1) 기존처럼 반경 후보
        candidates = self.model.space.query_radius(self.xy, radius, predicate=None)
        if not candidates:
            return []

        # 2) 시야 폴리곤은 '사전계산된 것'을 조회만
        poly = self.model.vision_atlas.polygon_at(
            self.xy[0], self.xy[1], radius, self.model.obstacles_version
        )

        if poly.is_empty:
            return []

        # One vectorised containment test for the whole candidate set.
        #
        # This used to build a shapely Point per candidate and ask the polygon
        # about each one separately. At a thousand pedestrians that was the
        # single largest cost in the simulation: 274,818 Point constructions
        # and 222,066 `covers` calls over 20 steps, and `_neighbors` alone
        # accounted for 45 per cent of the step. The social force everyone
        # assumes is the expensive part was 7 per cent.
        #
        # `intersects_xy` rather than `contains_xy`. For a point the two
        # differ exactly on the boundary: `contains_xy` excludes it and
        # `covers` includes it, so a pedestrian standing on the edge of
        # somebody's field of view would appear or vanish depending on which
        # was used. `intersects_xy` agrees with `covers` everywhere, checked
        # against 7,200 tests on real vision polygons with real crowd
        # positions and zero disagreements.
        import numpy as np
        from shapely import intersects_xy

        refs = []
        xs = []
        ys = []
        for b in candidates:
            ref = b.ref
            if (ref is None) or (ref is self) or getattr(ref, "dead", False):
                continue
            refs.append(ref)
            xs.append(b.pos[0])
            ys.append(b.pos[1])
        if not refs:
            return []

        hit = intersects_xy(poly, np.asarray(xs, dtype=float),
                            np.asarray(ys, dtype=float))
        return [ref for ref, ok in zip(refs, hit) if ok]

    def move(self) -> None:
        """Handles the movement behavior.
        Here the agent decides   if it moves,
        drinks the heal potion,
        or attacks other agent."""
        if(self.model.robot_version != 'N'):
            cells_with_agents = []
            robot_xy = [self.model.robot.xy[0], self.model.robot.xy[1]]

        if (self.type == 3):
            self.robot_step += 1

                   
            if self.model.robot_type == "Q":
                new_position_robot = self.robot_policy_Q()
            
            elif self.model.robot_type == "T":
                new_position_robot = self.robot_policy_Q()
            elif self.model.robot_type == "R":
                new_position_robot = self.robot_policy_Q()
            else:
                raise ValueError(f"Unknown robot_type {self.model.robot_type}")
            

            self.model.space.move(self.unique_id, self.xy)
            self.pos = new_position_robot
            return
        
        if self.type in (0, 1, 2):               # (로봇이 아니면)
            # (2) 힘 계산·충돌 예측·이동 ----------
            self.pos = (round(self.xy[0]), round(self.xy[1]))
            new_pos  = self.agent_modeling()      # ← 내부에서 predict_collision() 포함
            self.pos = (self.xy[0], self.xy[1])
    
    def _wall_repulsion(self):
        from shapely.geometry import Point, box
        Fwx = Fwy = 0.0
        KN = SF_WALL_KN
        CN = SF_WALL_CN
        MU_T = SF_WALL_MU_T
        p = Point(self.xy[0], self.xy[1])
        F_contact_x = 0.0
        F_contact_y = 0.0
        F_fric_x = 0.0
        F_fric_y = 0.0
        # Only walls near enough to matter, found through the spatial index.
        # This used to copy the obstacle list, build a fresh boundary polygon
        # and ask every polygon for its exterior on every call, for every
        # agent, on every step: 92,400 distance queries over 200 steps with
        # 40 agents, almost all of them against walls metres away.
        # A pedestrian keeps its body clear of the wall plus a hand's breadth,
        # not plus a metre. The old standoff made every doorway narrower than
        # it is by three metres of clearance a real person does not keep.
        r_sum = self.body_radius + SF_WALL_MARGIN_M
        reach = max(1.0, r_sum * 2.0)
        exteriors = self.model._wall_exteriors
        # A box, not a buffered point. The buffer was building a polygon for
        # every agent on every step purely to ask the index a range question,
        # which is the one thing a box already answers; it became the largest
        # single cost once the walls themselves were indexed.
        near = self.model._wall_exterior_index.query(
            box(self.xy[0] - reach, self.xy[1] - reach,
                self.xy[0] + reach, self.xy[1] + reach))

        for idx in near:
            ring = exteriors[int(idx)]
            d = ring.distance(p)
            if d < r_sum:

                q = ring.interpolate(ring.project(p))
                dx = self.xy[0]-q.x
                dy = self.xy[1]-q.y
                dist = math.hypot(dx, dy) or 1e-9
                ux, uy = dx/dist, dy/dist

                # Standoff: a soft push while the body is merely close.
                F_contact_x += SF_WALL_SOFT_K * (r_sum - d) * ux
                F_contact_y += SF_WALL_SOFT_K * (r_sum - d) * uy
                if d >= self.body_radius:
                    # Not touching. No contact spring and, above all, no
                    # tangential friction: braking everyone who walks near a
                    # wall is what closed every doorway in the model.
                    continue

                penetration = self.body_radius - d
                Fn_k = KN * penetration
                # (b) 상대속도에 대한 법선 감쇠
                v_jx = 0
                v_jy = 0
                rel_vx, rel_vy = (self.vel[0] - v_jx), (self.vel[1] - v_jy)
                v_n = rel_vx*ux + rel_vy*uy        # 법선 성분
                
                if v_n < 0:
                    restitution = 0.2
                    drop = (1.0-restitution)*v_n
                    self.vel[0] -= drop*ux
                    self.vel[1] -= drop*uy

                Fn_c = -CN * v_n                   # 접근할수록(음수) + 방향은 법선
                Fn = max(Fn_k + Fn_c, 0.0)         # 법선 힘은 음수가 되지 않게

                F_contact_x += Fn * ux
                F_contact_y += Fn * uy

                # (c) 접선 방향(미끄럼) 마찰: v_t = rel_v - v_n n
                vt_x = rel_vx - v_n*ux
                vt_y = rel_vy - v_n*uy
                F_fric_x += -MU_T * vt_x
                F_fric_y += -MU_T * vt_y

            if d < reach:
                q = ring.interpolate(ring.project(p))
                dx = self.xy[0]-q.x
                dy = self.xy[1]-q.y
                dist = math.hypot(dx, dy) or 1e-9
                nx, ny = dx/dist, dy/dist
                mag = 200 * math.exp(-(d/0.2))
                Fwx += mag * nx
                Fwy += mag * ny
        return Fwx, Fwy

    def _navmesh_detour_heading(self):
        """Find a body-clear next portal when the direct goal hits a wall."""
        goal = getattr(self, "now_goal", None)
        if goal is None or not hasattr(self.model, "find_mesh"):
            return None
        here = self.model.find_mesh(self.xy)
        dest = self.model.find_mesh(goal)
        walkable = getattr(self.model, "pure_mesh", ())
        if here not in walkable:
            return None
        if dest not in walkable:
            # A directional goal can land inside a building. Continue its
            # bearing to the first free point beyond that building; steering
            # to the near wall would recreate the concave-corner deadlock.
            dx, dy = goal[0] - self.xy[0], goal[1] - self.xy[1]
            distance = math.hypot(dx, dy)
            if distance < 1e-6:
                return None
            ux, uy = dx / distance, dy / distance
            for k in range(1, 41):
                px = goal[0] + ux * (0.5 * k)
                py = goal[1] + uy * (0.5 * k)
                if not (self.body_radius <= px <= self.model.width - self.body_radius
                        and self.body_radius <= py <= self.model.height - self.body_radius):
                    break
                if not self.model.is_free_point(px, py, padding=self.body_radius):
                    continue
                dest = self.model.find_mesh((px, py))
                if dest in walkable:
                    break
            if dest not in walkable:
                return None
        if here == dest:
            return None
        nxt = self.model.next_mesh_from_to(here, dest)
        if nxt is None:
            return None
        shared = [p for p in here if p in nxt]
        if len(shared) != 2:
            return None
        mx = (shared[0][0] + shared[1][0]) / 2.0
        my = (shared[0][1] + shared[1][1]) / 2.0
        cx = sum(p[0] for p in nxt) / 3.0
        cy = sum(p[1] for p in nxt) / 3.0
        for fraction in (0.6, 0.3, 0.0):
            tx = mx + fraction * (cx - mx)
            ty = my + fraction * (cy - my)
            length = math.hypot(tx - self.xy[0], ty - self.xy[1])
            if length < 0.1:
                continue
            samples = max(2, int(math.ceil(length / 0.2)))
            if all(self.model.is_free_point(
                    self.xy[0] + (tx - self.xy[0]) * k / samples,
                    self.xy[1] + (ty - self.xy[1]) * k / samples,
                    padding=self.body_radius)
                   for k in range(1, samples + 1)):
                return ((tx - self.xy[0]) / length,
                        (ty - self.xy[1]) / length)
        return None

    def _wall_aware_heading(self, ux, uy):
        """Slide along a nearby wall when the desired heading points into it.

        A hazard belief or a followed pedestrian gives a *direction*, not an
        obstacle-free route. A constant drive into a building otherwise forms
        a stable equilibrium with wall repulsion, even though the pedestrian
        has plenty of room to walk around the building.
        """
        from shapely.geometry import LineString, Point, box

        x, y = self.xy
        # Begin the turn before the acceleration-limited walker reaches the
        # collision boundary, rather than after it has already stopped there.
        reach = self.body_radius + SF_WALL_LOOKAHEAD_M
        rings = self.model._wall_exteriors
        nearby = self.model._wall_exterior_index.query(
            box(x - reach, y - reach, x + reach, y + reach))
        goal = getattr(self, "now_goal", None)
        if (goal is not None and hasattr(self.model, "is_free_segment")
                and math.hypot(goal[0] - x, goal[1] - y) > reach
                and not self.model.is_free_segment(
                    x, y, goal[0], goal[1], padding=self.body_radius)):
            # In a concave recess the next 1.75 m can be clear even though
            # the full line to the goal runs through its closed end. Waiting
            # until that short lookahead blocks makes the pedestrian reverse
            # between two points forever. Commit to the route out now.
            detour = self._navmesh_detour_heading()
            if detour is not None:
                self._wall_follow_side = 0
                return detour
        if len(nearby) == 0:
            self._wall_follow_side = 0
            return ux, uy

        # Slide only when the way ahead is actually blocked.
        #
        # Without this the heading is deflected by any wall face the
        # pedestrian is pointing at, and a doorway is exactly that: the wall
        # beside the opening has its normal pointing straight back at
        # somebody walking through, so everyone approaching a door was turned
        # sideways and shuffled along the facade. Measured on a 3 m door, 90
        # people took 253 s to get out against about 25 s in the published
        # bottleneck experiments. The model had no way to notice that the wall
        # in front of it has a gap in it.
        step = max(0.4, self.body_radius)
        samples = max(2, int(reach / step))

        def clear(hx, hy):
            return all(self.model.is_free_point(x + hx * step * k,
                                                y + hy * step * k,
                                                padding=self.body_radius)
                       for k in range(1, samples + 1))

        if clear(ux, uy):
            self._wall_follow_side = 0
            return ux, uy

        detour = self._navmesh_detour_heading()
        if detour is not None:
            self._wall_follow_side = 0
            return detour

        # Blocked straight ahead: steer around it toward the goal before
        # giving up and sliding along the wall.
        #
        # This is what makes a doorway usable. Sliding was the only response,
        # so anybody approaching a door even slightly off the axis was turned
        # along the facade instead of angling into the opening, and the door
        # only passed the few who happened to be lined up with it. Turning by
        # a few tens of degrees finds the gap, which is what a person walking
        # into a doorway does.
        for deg in (20.0, -20.0, 40.0, -40.0, 60.0, -60.0):
            a = math.radians(deg)
            ca, sa = math.cos(a), math.sin(a)
            hx, hy = ux * ca - uy * sa, ux * sa + uy * ca
            if clear(hx, hy):
                self._wall_follow_side = 0
                return hx, hy

        p = Point(x, y)
        candidates = []
        for idx in nearby:
            ring = rings[int(idx)]
            dist = ring.distance(p)
            if dist >= reach:
                continue
            q = ring.interpolate(ring.project(p))
            nx, ny = x - q.x, y - q.y
            norm = math.hypot(nx, ny)
            if norm < 1e-8:
                continue
            nx, ny = nx / norm, ny / norm
            candidates.append((dist, ring, nx, ny))
        if not candidates:
            self._wall_follow_side = 0
            return ux, uy
        candidates.sort(key=lambda item: item[0])
        nearest_dist, _, nx, ny = candidates[0]
        if ux * nx + uy * ny >= -0.15:
            # In a narrow passage the closest wall can be behind the goal.
            # Only then inspect similarly close walls in the travel corridor;
            # a more distant wall must not hijack ordinary local movement.
            corridor = LineString(
                ((x, y), (x + ux * reach, y + uy * reach))
            ).buffer(self.body_radius, quad_segs=4)
            blocking = next((item for item in candidates[1:]
                             if item[0] <= nearest_dist + 0.25
                             and ux * item[2] + uy * item[3] < -0.15
                             and item[1].intersects(corridor)), None)
            if blocking is None:
                self._wall_follow_side = 0
                return ux, uy
            _, _, nx, ny = blocking

        tx, ty = -ny, nx
        alignment = ux * tx + uy * ty
        side = getattr(self, "_wall_follow_side", 0)
        if side == 0:
            side = 1 if alignment >= 0 else -1
        # Keep the chosen side until the wall clears or the wedge recovery
        # deliberately reverses it. Re-evaluating a slightly changing belief
        # each step made pedestrians shuffle back and forth along a façade.
        self._wall_follow_side = side

        # A little outward bias keeps the body clear of the wall while the
        # tangential component carries it toward an end or a doorway.
        hx, hy = side * tx + 0.3 * nx, side * ty + 0.3 * ny
        scale = math.hypot(hx, hy)
        return hx / scale, hy / scale

    # ---- 스윕 이동(터널링 방지) ----
    def swept_move(self, xy, vel, dt):
        nx, ny = xy[0], xy[1]
        max_disp = max(abs(vel[0]*dt), abs(vel[1]*dt))
        steps = max(1, int(math.ceil(max_disp / 0.5)))
        sdt = dt / steps
        # A body, not a point.
        #
        # `is_free` asks whether a coordinate is outside the obstacles, so a
        # half-metre-wide pedestrian was free to walk into a half-metre gap
        # where its body does not fit. It then cannot get out: the wall
        # repulsion cancels the drive in every direction, and the release
        # slides it along the wall only for it to come back, because the only
        # way out is the same gap it came in by. Measured in a Soho crop, one
        # pedestrian walked 58 m over six hundred steps without ever getting
        # 0.9 m from where it started, pressed against a building edge the
        # whole time, in a spot whose largest free circle is exactly its own
        # body radius.
        fits = lambda px, py: self.model.is_free_point(px, py,
                                                       padding=self.body_radius)
        for _ in range(steps):
            tx = nx + vel[0]*sdt
            ty = ny + vel[1]*sdt
            if fits(tx, ty):
                nx, ny = tx, ty
            else:
                # 축 분리
                moved = False
                if fits(nx + vel[0]*sdt, ny):
                    nx += vel[0]*sdt
                    moved = True
                if fits(nx, ny + vel[1]*sdt):
                    ny += vel[1]*sdt
                    moved = True
                if not moved:
                    # Along the facade. Splitting the move by axis slides
                    # along walls that run with the grid and stops dead at
                    # a slanted one: a pedestrian in a passage between a
                    # slanted building and the crop edge had its drive along
                    # the passage refused on both axes, every step, for the
                    # rest of the episode (maboneng crop).
                    face = self._nearest_wall_normal(nx, ny)
                    if face is not None:
                        ux, uy = face
                        dx, dy = vel[0]*sdt, vel[1]*sdt
                        into = dx*ux + dy*uy
                        if into < 0.0:
                            dx -= into*ux
                            dy -= into*uy
                        dx += 0.01*ux
                        dy += 0.01*uy
                        if fits(nx + dx, ny + dy):
                            nx, ny = nx + dx, ny + dy
        if not fits(nx, ny):
            nx, ny = self._push_out_of_wall(nx, ny)
        # Wedging is measured as a lack of progress, not as a lack of motion.
        #
        # A pedestrian pressed into a corner does not stand still: it shuffles
        # back and forth against the wall, a few tenths of a metre a step, and
        # gets nowhere. Comparing one step's displacement to a threshold calls
        # that movement and resets the counter every step, so the release
        # never fired: the one pedestrian still stuck in a Soho crop moved
        # 0.003 m a step with a wedged counter of 0 while staying inside a
        # 0.4 m circle for four hundred steps.
        #
        # So progress is measured from an anchor that only moves when the
        # pedestrian actually gets somewhere.
        if getattr(self, "_dwelling", False):
            self._progress_anchor = (nx, ny)
            self._wedged_for = 0
            return [nx, ny]

        anchor = getattr(self, "_progress_anchor", None)
        if anchor is None:
            self._progress_anchor = (nx, ny)
            self._wedged_for = 0
            return [nx, ny]
        if math.hypot(nx - anchor[0], ny - anchor[1]) >= self.WEDGE_PROGRESS_M:
            self._progress_anchor = (nx, ny)
            self._wedged_for = 0
            return [nx, ny]

        self._wedged_for = getattr(self, "_wedged_for", 0) + 1
        if self._wedged_for < self.WEDGE_PATIENCE:
            return [nx, ny]
        freed = self._unwedge([nx, ny])
        self._progress_anchor = (freed[0], freed[1])
        self._wedged_for = 0
        return freed

    def _nearest_wall_normal(self, x, y):
        """Unit normal, pointing away from it, of the nearest building face
        within a body radius and a hand's breadth; None if there is none."""
        from shapely.geometry import Point, box
        m = self.model
        if not hasattr(m, "_obstacles_for_query"):
            return None
        reach = self.body_radius + SF_WALL_MARGIN_M
        p = Point(x, y)
        index, polys = m._obstacles_for_query()
        best = None
        for idx in index.query(box(x - reach, y - reach, x + reach, y + reach)):
            ring = polys[int(idx)].exterior
            d = ring.distance(p)
            if d < reach and (best is None or d < best[0]):
                best = (d, ring)
        if best is None:
            return None
        q = best[1].interpolate(best[1].project(p))
        n = math.hypot(x - q.x, y - q.y)
        if n < 1e-9:
            return None
        return (x - q.x) / n, (y - q.y) / n

    def _push_out_of_wall(self, x, y):
        """Move a body that overlaps a building just clear of its nearest face.

        The swept move only accepts positions where the whole body fits, so a
        body already overlapping a wall, by contact forces or a crowd push,
        was refused every move, including the ones that would have reduced
        the overlap. Measured on a maboneng crop: a pedestrian 0.004 m into a
        building in a 0.8 m passage along the crop edge stood still for the
        rest of the episode while its drive pointed along the passage.
        """
        from shapely.geometry import Point, box
        m = self.model
        if not hasattr(m, "_obstacles_for_query"):
            return x, y
        r = self.body_radius
        p = Point(x, y)
        index, polys = m._obstacles_for_query()
        best = None
        for idx in index.query(box(x - r, y - r, x + r, y + r)):
            poly = polys[int(idx)]
            if poly.contains(p):
                return x, y          # inside a building: the unwedge handles it
            d = poly.exterior.distance(p)
            if d < r and (best is None or d < best[0]):
                best = (d, poly.exterior)
        if best is None:
            return x, y
        d, ring = best
        q = ring.interpolate(ring.project(p))
        n = math.hypot(x - q.x, y - q.y)
        if n < 1e-9:
            return x, y
        push = r - d + 0.01
        px = x + (x - q.x) / n * push
        py = y + (y - q.y) / n * push
        if not (0.0 < px < m.width and 0.0 < py < m.height):
            return x, y
        if (m.is_free_point(px, py, padding=r)
                or m.obstacle_clearance(px, py) > m.obstacle_clearance(x, y)):
            return px, py
        return x, y

    # Steps of not getting anywhere before a pedestrian is treated as wedged.
    WEDGE_PATIENCE = 12

    # How far a pedestrian must get from its anchor to count as making
    # progress. Larger than the shuffle a wedged pedestrian manages against a
    # wall, smaller than a walking pedestrian covers in the patience window:
    # at 1.35 m/s and a 0.25 s step that is several metres.
    WEDGE_PROGRESS_M = 1.0

    def _unwedge(self, xy):
        """Free a pedestrian whose forces have cancelled against a wall.

        Some pedestrians end up in a recess where the wall repulsion points
        exactly opposite the way they want to go. Measured in a medina: a
        pedestrian 0.21 m from a wall, well inside the 1.5 m contact
        threshold, with a goal 2.95 m ahead, a drive direction of
        (0.76, 0.65) and a wall force of (-62.8, -53.8). Anti-parallel, so the
        net force is zero, the velocity is zero, and the swept move has
        nothing to attempt. It stood still for the rest of the episode.

        The social force model has no way out of this on its own: it is a
        stable equilibrium, not a transient. So after a few steps of not
        moving, step along the wall instead of into it. Sliding rather than
        pushing is also what a person does in a doorway.

        Patience matters. A pedestrian legitimately stands still while
        milling, or when the crowd around it is packed, and shoving those
        would change the dynamics being modelled rather than fix a defect.
        """
        fx, fy = self._wall_repulsion()
        norm = math.hypot(fx, fy)
        if norm < 1e-6:
            # Not wedged against a wall at all; something else is holding it.
            return [xy[0], xy[1]]

        ux, uy = fx / norm, fy / norm
        # A wall-following side that reached a dead end must try the other end.
        side = getattr(self, "_wall_follow_side", 0)
        if side:
            side = -side
            self._wall_follow_side = side
        else:
            side = 1

        def clear_segment(cx, cy):
            # Testing only the destination used to teleport pedestrians across
            # thin walls and into gaps too narrow for their bodies.
            length = math.hypot(cx - xy[0], cy - xy[1])
            samples = max(2, int(math.ceil(length / 0.2)))
            for i in range(1, samples + 1):
                t = i / samples
                if not self.model.is_free_point(
                    xy[0] + t * (cx - xy[0]),
                    xy[1] + t * (cy - xy[1]),
                    padding=self.body_radius,
                ):
                    return False
            return True

        for step in (0.5, 1.0):
            for tx, ty in ((side * -uy, side * ux),
                           (side * uy, side * -ux)):
                cx, cy = xy[0] + tx * step, xy[1] + ty * step
                if clear_segment(cx, cy):
                    self.vel = [0.0, 0.0]
                    return [cx, cy]
            cx, cy = xy[0] + ux * step, xy[1] + uy * step
            if clear_segment(cx, cy):
                self.vel = [0.0, 0.0]
                return [cx, cy]
        return [xy[0], xy[1]]

        

    def agent_modeling(self):
        """
        Helbing + Contact (penalty) model
        - 비관통(원-원) 접촉: 탄성(스프링) + 점성 감쇠 + 접선 마찰
        - 공기저항 형태 속도 감쇠로 관성 억제
        """
        import math



        # ====== 기본 파라미터 (필요하면 수치만 조정) ======
        dt   = AGENT_TIME_STEP
        tau  = 1                  
        A_MAX = 1.5                    # 가속 클립 ↑ 약간 강화
        V_MAX_MULT = 1.00              # 목표속도보다 과속 안하게
        # One definition of the body, from config. It used to be written here
        # as well as in config, so the radius the forces used and the radius
        # the collision test used could drift apart.
        BODY_RADIUS = AGENT_BODY_RADIUS
        WALL_RADIUS = AGENT_BODY_RADIUS + SF_WALL_MARGIN_M
        KN = SF_KN
        CN = SF_CN
        MU_T = SF_MU_T
        K_AGENT = SF_K_AGENT
        K_WALL  = 500 # modified 참고
        LAMBDA_A = SF_LAMBDA_A
        # 공기저항(속도 감쇠) → 둥둥 뜨는 느낌 제거
        BETA = 0                     # F_drag = -BETA * v

        # ---- 유틸 ----
        def get_radius(agent):
            if getattr(agent, "type", None) == 3:
                return ROBOT_BODY_RADIUS
            elif getattr(agent, "type", None) in (9, 11):  # 벽/장애물
                return WALL_RADIUS
            else:
                return BODY_RADIUS

        def soft_clip_vec(x, y, lim):
            n = math.hypot(x, y)
            if n <= lim: return x, y
            s = math.tanh(n/lim) / (n/lim)
            return x*s, y*s
        
        # How far this pedestrian still has to walk to be clear of the
        # hazard, geodesically. Zero once it is out by the safety margin.
        #
        # It used to be the distance to the nearest exit, and an unreachable
        # exit reported a huge number, which was taken as a signal to remove
        # the pedestrian. That cannot happen here: being far from safety is
        # the normal state at the start of an episode, not a failure to route,
        # and a level where somebody genuinely cannot get out is rejected by
        # the generator's validator before it is ever run.
        self.danger = self.model.escape_distance(self.xy)

        # 이웃 상호작용 (사람/로봇)
        sensor_R = self.vision_radius
        near_agents = self._neighbors(sensor_R)

        # Awareness first: what this pedestrian knows decides what it wants.
        #
        # A robot only counts as a cue while it is signalling and can be seen
        # signalling. Before, any robot within range cued everybody whatever
        # it was doing and through walls, which made the control mode
        # impossible: a robot told the crowd about the hazard even when it was
        # deliberately saying nothing.
        perceived, new_ids = self._perceive_robots(near_agents)
        self._signal_robot = (min(perceived, key=lambda rb: math.hypot(
            self.xy[0] - rb.xy[0], self.xy[1] - rb.xy[1]))
            if perceived else None)
        self.update_awareness(near_agents, bool(perceived))
        # Whom to follow, decided after awareness so a robot that has just
        # warned somebody can also be followed by them (M5, M6).
        self._robot_step(near_agents, perceived, new_ids)

        self.which_goal_agent_want(near_agents)
        # Straight-line goals (fleeing along a remembered bearing, following
        # a neighbour or a robot) are walked around buildings rather than
        # into them. Measured on OSM crops: 7 of 14 pedestrians wedged against
        # a wall for 80+ steps were pressed into a concave building corner by
        # a flee heading or a neighbour on the far side of the wall. A goal
        # already in sight, which includes every navmesh waypoint, is kept.
        if (not getattr(self, "_dwelling", False)
                and getattr(self, "scripted_goal", None) is None):
            self.now_goal = self._routed_goal(self.now_goal)

        # ---- 목표 방향 ----
        gx = self.now_goal[0] - self.xy[0]
        gy = self.now_goal[1] - self.xy[1]
        gd = math.hypot(gx, gy)
        if gd > 0:
            dir_x, dir_y = gx/gd, gy/gd
            dir_x, dir_y = self._wall_aware_heading(dir_x, dir_y)
        else:
            dir_x, dir_y = 0.0, 0.0

        # ---- 원하는 속도 → Helbing desired force ----
        v_des_x = self.desired_speed_a * dir_x
        v_des_y = self.desired_speed_a * dir_y
        F_des_x = self.mass * (v_des_x - self.vel[0]) / tau
        F_des_y = self.mass * (v_des_y - self.vel[1]) / tau

        # ---- 기존의 약한(원거리) 반발력 (지수) ----
        F_rep_x = 0.0
        F_rep_y = 0.0

        # ---- 접촉(비관통) + 마찰 모델(핵심 추가) ----
        F_contact_x = 0.0
        F_contact_y = 0.0
        F_fric_x    = 0.0
        F_fric_y    = 0.0

        r_i = BODY_RADIUS
        # 자기 상태 (속도)
        v_ix, v_iy = self.vel[0], self.vel[1]
        self.meeting_robot = 0
        for nb in near_agents:
            if nb is self or getattr(nb, "dead", False):
                continue
            if nb.type == 3:
                self.meeting_robot = 1

            dx = self.xy[0] - nb.xy[0]
            dy = self.xy[1] - nb.xy[1]
            d  = math.hypot(dx, dy)
            if d < 1e-9:
                # 완전 겹침 초기 해소(랜덤 툭 치기)
                jx, jy = (1.0, -1.0) if random.random() < 0.5 else (-1.0, 1.0)
                F_contact_x += jx * KN * 0.01
                F_contact_y += jy * KN * 0.01
                continue

            ux, uy = dx/d, dy/d  # (nb -> self) 법선 방향
            nb_R = getattr(nb, "radius", BODY_RADIUS)
            r_sum = self.body_radius + nb_R

            # 원거리 지수 반발, 앞쪽 가중
            #
            # The weight runs from 1 for a neighbour directly ahead to
            # SF_ANISOTROPY for one directly behind. Without it a uniform
            # queue is symmetric and the model has no fundamental diagram at
            # all: whoever is in front pushes back exactly as hard as whoever
            # is behind pushes forward, so walking speed does not depend on
            # density. See docs/crowd_validation.md.
            mag = K_AGENT * math.exp((r_sum-d) / max(LAMBDA_A, 1e-6))
            if SF_ANISOTROPY < 1.0:
                sp = math.hypot(v_ix, v_iy)
                if sp > 1e-6:
                    # cos of the angle between where this pedestrian is going
                    # and where the neighbour is. -ux is the direction to it.
                    cos_phi = (-ux * v_ix - uy * v_iy) / sp
                    mag *= (SF_ANISOTROPY
                            + (1.0 - SF_ANISOTROPY) * (1.0 + cos_phi) / 2.0)
            F_rep_x += mag * ux
            F_rep_y += mag * uy

            # # # 1) 원거리 지수 반발(부드러운 회피)
            # if getattr(nb, "type", None) in (11, 9):
            #     mag = K_WALL * math.exp((r_sum - d) / LAMBDA_A)
            #     F_rep_x += mag * ux
            #     F_rep_y += mag * uy
            # else:
            #     mag = K_AGENT * math.exp((r_sum - d) / LAMBDA_A)
            #     F_rep_x += mag * ux
            #     F_rep_y += mag * uy

            # 2) 근거리 접촉(비관통) + 점성 감쇠 + 접선 마찰
            if d < r_sum:
                # 침투량(양수면 겹침)
                penetration = (r_sum - d)
                # (a) 법선 스프링
                Fn_k = KN * penetration
                # (b) 상대속도에 대한 법선 감쇠
                v_jx = getattr(nb, "vel", [0,0])[0] if hasattr(nb, "vel") else 0.0
                v_jy = getattr(nb, "vel", [0,0])[1] if hasattr(nb, "vel") else 0.0
                rel_vx, rel_vy = (v_ix - v_jx), (v_iy - v_jy)
                v_n = rel_vx*ux + rel_vy*uy        # 법선 성분
                
                if v_n < 0:
                    restitution = 0
                    drop = (1.0-restitution)*v_n
                    self.vel[0] -= drop*ux
                    self.vel[1] -= drop*uy

                Fn_c = -CN * v_n                   # 접근할수록(음수) + 방향은 법선
                Fn = max(Fn_k + Fn_c, 0.0)         # 법선 힘은 음수가 되지 않게

                F_contact_x += Fn * ux
                F_contact_y += Fn * uy

                # (c) 접선 방향(미끄럼) 마찰: v_t = rel_v - v_n n
                vt_x = rel_vx - v_n*ux
                vt_y = rel_vy - v_n*uy
                F_fric_x += -MU_T * vt_x
                F_fric_y += -MU_T * vt_y
        
        BETA=0
        decay = 1
        self.vel[0] *= decay
        self.vel[1] *= decay
        # ---- 공기저항(속도 감쇠) ----
        F_drag_x = -BETA * self.vel[0]
        F_drag_y = -BETA * self.vel[1]


        # ---- 외곽 지대 나가지 않게 ----

        # 🔹 (추가) 맵 outer wall 반발력
        W = self.model.width
        H = self.model.height
        # Only at contact range, like a wall. The crop edge is not a wall: the
        # city carries on past it, and the swept move already keeps bodies
        # inside the map. A 2 m band pinned pedestrians between the edge and a
        # building a metre from it, and held people back from the street
        # mouths they were leaving by (7 of 14 wedged cases measured).
        MARGIN = self.body_radius + SF_WALL_MARGIN_M
        K_BORDER = 200.0     # 경계 힘 세기 (필요하면 조절)
        F_wx = 0
        F_wy = 0

        W_x, W_y = self._wall_repulsion()
        F_wx += W_x
        F_wy += W_y

        # left 벽 (x = 0 부근)
        dx = max(0.0, MARGIN - self.xy[0])
        if dx > 0.0:
            # 왼쪽 벽에 가까우면 +x 방향으로 민다
            F_wx += K_BORDER * dx

        # right 벽 (x = W 부근)
        dx = max(0.0, self.xy[0] - (W - MARGIN))
        if dx > 0.0:
            # 오른쪽 벽에 가까우면 -x 방향으로 민다
            F_wx -= K_BORDER * dx

        # bottom 벽 (y = 0 부근)
        dy = max(0.0, MARGIN - self.xy[1])
        if dy > 0.0:
            # 아래쪽 벽에 가까우면 +y 방향
            F_wy += K_BORDER * dy

        # top 벽 (y = H 부근)
        dy = max(0.0, self.xy[1] - (H - MARGIN))
        if dy > 0.0:
            # 위쪽 벽에 가까우면 -y 방향
            F_wy -= K_BORDER * dy


        # ---- 스윕 이동(터널링 방지) ----
        def swept_move(xy, vel, dt):
            nx, ny = xy[0], xy[1]
            max_disp = max(abs(vel[0]*dt), abs(vel[1]*dt))
            steps = max(1, int(math.ceil(max_disp / 0.5)))
            sdt = dt / steps
            for _ in range(steps):
                tx = nx + vel[0]*sdt
                ty = ny + vel[1]*sdt
                ix, iy = int(round(tx)), int(round(ty))
                if self.model.valid_space.get((ix, iy), False):
                    nx, ny = tx, ty
                    continue
                # 축 분리 시도
                ix_only = int(round(nx + vel[0]*sdt))
                if self.model.valid_space.get((ix_only, int(round(ny))), False):
                    nx = nx + vel[0]*sdt
                iy_only = int(round(ny + vel[1]*sdt))
                if self.model.valid_space.get((int(round(nx)), iy_only), False):
                    ny = ny + vel[1]*sdt
            return [nx, ny]

        #self.xy = swept_move(self.xy, self.vel, dt)
        # self.xy[0] = self.xy[0] + self.vel[0] * dt
        # self.xy[1] = self.xy[1] + self.vel[1] * dt



        # ---- 총합 힘 ----
        F_x = F_des_x + F_rep_x + F_contact_x + F_fric_x + F_drag_x + F_wx
        F_y = F_des_y + F_rep_y + F_contact_y + F_fric_y + F_drag_y + F_wy

        # ---- 가속도 계산 + 클립 ----
        a_x = F_x / self.mass
        a_y = F_y / self.mass
        a_x, a_y = soft_clip_vec(a_x, a_y, A_MAX)

        self.acc[0], self.acc[1] = a_x, a_y

        # ---- 속도 업데이트 ----
        self.vel[0] += a_x * dt
        self.vel[1] += a_y * dt

        # 속도 클립 (벡터 노름)
        v_des_scalar = max(self.desired_speed_a, 1e-6)
        V_MAX = V_MAX_MULT * v_des_scalar
        spd = math.hypot(self.vel[0], self.vel[1])
        if spd > V_MAX:
            s = V_MAX / spd
            self.vel[0] *= s; self.vel[1] *= s
        #print("agent desired speed : ", v_des_scalar)
        #print("agent speed : ", self.vel[0], self.vel[1])
        self.xy = self.swept_move(self.xy, self.vel, dt)
        #self.model.space.clamp(self.xy)
        self.model.space.move(self.unique_id, self.xy)

        self.direction = [self.vel[0], self.vel[1]]

        return tuple(self.xy)

    
    # ------------------------------------------------------------------
    # hazard awareness
    # ------------------------------------------------------------------

    # ---- robots: noticing and choosing whom to follow (M5, M6) -------------
    #
    # docs/behavior_model_design.md. A robot has to be noticed before it can
    # be followed, and whom to follow is a choice among the noticed robots
    # and "my own way", made only when something changes: a robot is newly
    # noticed, or the one being followed is lost. Between those moments the
    # choice stands, so an instruction is not a coin flip every 0.5 s.

    def _signal_candidates(self, neighbors):
        """Robots whose signal this pedestrian could read now, with distance.

        The robot has to be signalling at all ("off" is a body in the way and
        nothing else), within ROBOT_SIGNAL_RADIUS_M, and in sight: the
        field-of-view test in `_neighbors` already established that, so a
        robot behind a block cannot instruct anybody.
        """
        radius = float(ROBOT_SIGNAL_RADIUS_M)
        if ROBOT_SIGNAL_REQUIRES_SIGHT:
            pool = [nb for nb in neighbors if getattr(nb, "type", None) == 3]
        else:
            pool = list(getattr(self.model, "robots", []) or [])
        out = []
        for rb in pool:
            if getattr(rb, "mode", "off") == "off":
                continue
            d = math.hypot(self.xy[0] - rb.xy[0], self.xy[1] - rb.xy[1])
            if d <= radius:
                out.append((rb, d))
        return out

    def _signalling_robot(self, neighbors):
        """The nearest readable robot, noticed or not. For diagnostics."""
        cands = self._signal_candidates(neighbors)
        return min(cands, key=lambda c: c[1])[0] if cands else None

    def _robot_by_id(self, rid):
        if rid is None:
            return None
        for rb in getattr(self.model, "robots", None) or ():
            if rb.unique_id == rid:
                return rb
        return None

    def _perceive_robots(self, neighbors):
        """(noticed robots in sight now, ids noticed for the first time now).

        A readable robot not yet known is noticed with probability
        ROBOT_SIGNAL_P_MAX * (1 - d / ROBOT_SIGNAL_RADIUS_M) per step, so
        distance mostly changes how soon. Once noticed it stays known while
        it is seen, and is forgotten after ROBOT_SIGNAL_REENCOUNTER_GAP_STEPS
        out of sight or as soon as it stops signalling: seeing it again then
        is a new encounter.
        """
        step = int(getattr(self.model, "step_count", self.life_time))
        gap = int(ROBOT_SIGNAL_REENCOUNTER_GAP_STEPS)
        known = self._known_robots
        for rid in list(known):
            rb = self._robot_by_id(rid)
            if (step - known[rid] > gap or rb is None
                    or getattr(rb, "mode", "off") == "off"):
                del known[rid]
        radius = float(ROBOT_SIGNAL_RADIUS_M)
        perceived, new = [], set()
        for rb, d in self._signal_candidates(neighbors):
            rid = rb.unique_id
            if rid not in known:
                p = float(ROBOT_SIGNAL_P_MAX) * max(0.0, 1.0 - d / radius)
                if random.random() >= p:
                    continue
                new.add(rid)
            known[rid] = step
            perceived.append(rb)
        return perceived, new

    def _instruction_direction(self, rb):
        """The way this robot is asking this pedestrian to go (unit)."""
        if getattr(rb, "mode", "guide") == "direct":
            sx, sy = getattr(rb, "signal_dir", (0.0, 0.0))
        else:
            sx, sy = rb.xy[0] - self.xy[0], rb.xy[1] - self.xy[1]
        n = math.hypot(sx, sy)
        return (sx / n, sy / n) if n > 1e-6 else (0.0, 0.0)

    def _acting_headings(self, neighbors, grounded_only: bool = False):
        """Unit headings of the visibly responding people in sight who are
        moving (see _social_cue).

        `grounded_only` keeps those whose direction has a basis: they sensed
        the hazard or are following a robot. Used for going with the flow
        (M4): letting the uninformed follow the uninformed made a flock that
        drifted on its own momentum and, with a hazard nobody could sense,
        cleared the zone by chance.
        """
        out = []
        for nb in neighbors:
            if (nb is self or getattr(nb, "type", None) == 3
                    or getattr(nb, "dead", False)
                    or not getattr(nb, "responding", False)):
                continue
            if grounded_only and not (getattr(nb, "hazard_memory", None)
                                      or getattr(nb, "type", None) == 0):
                continue
            vx, vy = float(nb.vel[0]), float(nb.vel[1])
            s = math.hypot(vx, vy)
            if s >= 0.2:
                out.append((vx / s, vy / s))
        return out

    def _crowd_alignment(self, neighbors, direction):
        """Mean agreement (-1..1) between the acting people in sight and a
        direction; 0 when none of them is moving."""
        if direction == (0.0, 0.0):
            return 0.0
        hs = self._acting_headings(neighbors)
        if not hs:
            return 0.0
        return sum(hx * direction[0] + hy * direction[1]
                   for hx, hy in hs) / len(hs)

    def _robot_utility(self, rb, neighbors, leader_id):
        """Utility of following `rb` against "my own way" at 0.

        logit(base) + CROWD_BETA * crowd alignment + INERTIA_BETA * [followed
        now] + ln(person * scenario * form): one robot, a neutral crowd and a
        first meeting give exactly the base rate.
        """
        base = float(ROBOT_DIRECT_BASE_COMPLIANCE
                     if getattr(rb, "mode", "guide") == "direct"
                     else ROBOT_GUIDE_BASE_COMPLIANCE)
        scenario = float(getattr(self.model, "robot_compliance_scenario_factor",
                                 ROBOT_COMPLIANCE_SCENARIO_FACTOR))
        form = float(getattr(rb, "compliance_form_factor",
                             ROBOT_COMPLIANCE_FORM_FACTOR))
        scale = float(self.compliance) * scenario * form
        if base <= 0.0 or scale <= 0.0:
            return -math.inf
        logit = 30.0 if base >= 1.0 else math.log(base / (1.0 - base))
        u = logit + math.log(scale)
        u += float(ROBOT_CHOICE_CROWD_BETA) * self._crowd_alignment(
            neighbors, self._instruction_direction(rb))
        if rb.unique_id == leader_id:
            u += float(ROBOT_CHOICE_INERTIA_BETA)
        return u

    @staticmethod
    def _logit_choice(utilities):
        """Index of the chosen option, or None for "my own way" (utility 0)."""
        finite = [u for u in utilities if u != -math.inf]
        top = max(finite + [0.0])
        weights = [math.exp(u - top) if u != -math.inf else 0.0
                   for u in utilities]
        own = math.exp(-top)
        r = random.random() * (sum(weights) + own)
        for i, w in enumerate(weights):
            if r < w:
                return i
            r -= w
        return None

    def _robot_leads_back_in(self, rb) -> bool:
        """Following `rb` would take this pedestrian back into ground it
        believes dangerous after leaving it. Not an option then: people weigh
        a guide against what they know of the place (Cao et al. 2026)."""
        if not self.hazard_memory:
            return False
        mode = getattr(rb, "mode", "guide")
        goal = self._robot_led_goal(rb, mode)
        m = self.model
        target = m.find_mesh(goal) or self.choice_safe_mesh(goal)
        pts = self._route_points(target, goal)
        return pts is not None and self._reexposure(pts) > 0.0

    def _robot_step(self, neighbors, perceived, new_ids) -> None:
        """Choose whom to follow when a robot is newly noticed or the robot
        being followed is lost (M6); otherwise keep the standing choice."""
        leader_id = self.following_robot_id if self.type == 0 else None
        leader_lost = (leader_id is not None
                       and leader_id not in self._known_robots)
        if not new_ids and not leader_lost:
            return
        options = [rb for rb in perceived if not self._robot_leads_back_in(rb)]
        if (leader_id is not None and not leader_lost
                and all(rb.unique_id != leader_id for rb in options)):
            # Known and still signalling, briefly out of sight this step.
            rb = self._robot_by_id(leader_id)
            if rb is not None and not self._robot_leads_back_in(rb):
                options.append(rb)
        utilities = [self._robot_utility(rb, neighbors, leader_id)
                     for rb in options]
        pick = self._logit_choice(utilities) if options else None
        if pick is None:
            if self.type == 0:
                self._stop_following()
            return
        self._start_following(options[pick])

    def _start_following(self, rb) -> None:
        """Accepting an instruction is acting on it: milling ends here."""
        self.type = 0
        self.is_effected_by_robot = 1
        self.following_robot_id = rb.unique_id
        self.robot_lead_mode = getattr(rb, "mode", "guide")
        self._dwelling = False
        if self.awareness != "acting":
            if self.cued_at is None:
                self.cued_at = int(getattr(self.model, "step_count", 0))
                self.cue_source = "robot"
            self.awareness = "acting"
            self.ever_acted = True

    def _stop_following(self) -> None:
        """Decide for itself again, from here and against what it now knows:
        a destination held from before the robot took over is dropped."""
        self.type = 1
        self.following_robot_id = None
        self.robot_lead_mode = None
        self.now_pointing_mesh = None
        self._dwell_until = 0
        self._plan_version = -1

    def _sense_hazard(self) -> bool:
        """Direct perception. Returns True if the hazard was sensed this step.

        Two ranges, because standing in smoke is not the same as seeing it
        across a street. Inside the zone the per-step chance is high; outside
        it needs line of sight, which the visibility atlas already answers, so
        a hazard behind a block is not sensed at all.

        Both scale with the hazard's perceptibility. A fire announces itself;
        a gas leak does not, and for it this channel is effectively closed and
        only word of mouth and the robots remain.
        """
        zone = getattr(self.model, "danger_zone", None)
        if zone is None:
            return False
        perceptibility = float(getattr(self.model, "danger_perceptibility",
                                       0.5))
        if perceptibility < PERCEPTIBILITY_SENSORY_FLOOR:
            # No sensory cue at all. See config.PERCEPTIBILITY_SENSORY_FLOOR:
            # an unodorised leak is not faintly smellable, it is not smellable,
            # and treating it as a small per-step chance made every hazard
            # perceptible given a long enough episode.
            return False
        # How far above the floor, so the channel opens gradually.
        salience = ((perceptibility - PERCEPTIBILITY_SENSORY_FLOOR)
                    / max(1e-6, 1.0 - PERCEPTIBILITY_SENSORY_FLOOR))

        x, y = float(self.xy[0]), float(self.xy[1])
        if zone.contains(x, y):
            p = AWARENESS_P_INSIDE * salience
            spot = (x, y)
        else:
            d = zone.signed_distance(x, y)
            if d > self.vision_radius:
                return False
            if not self.model.hazard_in_sight(self):
                return False
            p = AWARENESS_P_VISIBLE * salience
            spot = zone.nearest_safe_point(x, y, margin=0.0)

        if random.random() >= p:
            return False
        self._remember_hazard(spot, perceptibility)
        return True

    def _remember_hazard(self, spot, perceptibility: float) -> None:
        """Record a sensed hazard location in this pedestrian's own memory.

        Above PERCEPTIBILITY_GLOBAL_CUE the sighting also conveys a coarse
        sense of the hazard as a whole, the way a smoke plume seen from a
        distance does: you cannot trace its outline but you know roughly where
        it is and that it is big. Below it, only the spot touched is known.

        A spot closer than a third of HAZARD_MEMORY_RADIUS_M to one already
        held adds nothing to what the pedestrian believes and is dropped; this
        is the resolution of the memory, not a behavioural parameter.
        """
        pts = [(float(spot[0]), float(spot[1]))]
        if perceptibility >= PERCEPTIBILITY_GLOBAL_CUE:
            zone = getattr(self.model, "danger_zone", None)
            if zone is not None:
                pts.append((float(zone.cx), float(zone.cy)))
        spacing = float(HAZARD_MEMORY_RADIUS_M) / 3.0
        for p in pts:
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) > spacing
                   for q in self.hazard_memory):
                self.hazard_memory.append(p)
                self._memory_version += 1
        if len(self.hazard_memory) > HAZARD_MEMORY_MAX_POINTS:
            del self.hazard_memory[:-HAZARD_MEMORY_MAX_POINTS]

    # ---- belief and routes (M2, M3) ----------------------------------------

    def _here_mesh(self):
        """The triangle this pedestrian stands in, looked up once per step.

        Several decisions in one step ask for it; find_mesh is a spatial
        query and was a fifth of the goal-selection cost when repeated.
        """
        step = getattr(self.model, "step_count", None)
        if step is None:
            # Not a running simulation (a test double): no step to key on.
            return self.model.find_mesh(self.xy) or self.choice_safe_mesh(self.xy)
        key = (int(step), float(self.xy[0]), float(self.xy[1]))
        memo = getattr(self, "_mesh_memo", None)
        if memo is not None and memo[0] == key:
            return memo[1]
        mesh = self.model.find_mesh(self.xy) or self.choice_safe_mesh(self.xy)
        self._mesh_memo = (key, mesh)
        return mesh

    def _believes_dangerous(self, x, y) -> bool:
        """Whether this pedestrian believes (x, y) is dangerous: within
        HAZARD_MEMORY_RADIUS_M of a spot it has sensed. Its own picture, not
        the true zone."""
        r = float(HAZARD_MEMORY_RADIUS_M)
        return any(math.hypot(x - hx, y - hy) < r
                   for hx, hy in self.hazard_memory)

    def _route_points(self, target_mesh, end=None):
        """The navmesh route from here to `target_mesh` as a polyline through
        the shared portals, ending at `end` (default: its centroid). None
        when there is no route."""
        m = self.model
        now = self._here_mesh()
        if now is None or target_mesh is None:
            return None
        pts = [(float(self.xy[0]), float(self.xy[1]))]
        t = now
        for _ in range(2000):
            if t == target_mesh:
                break
            nxt = m.next_mesh_from_to(t, target_mesh)
            if nxt is None:
                return None
            pts.append(tuple(self._portal_waypoint(t, nxt)))
            t = nxt
        else:
            return None
        if end is None:
            end = (sum(p[0] for p in target_mesh) / 3.0,
                   sum(p[1] for p in target_mesh) / 3.0)
        pts.append((float(end[0]), float(end[1])))
        return pts

    def _reexposure(self, pts, step_m: float = 1.5) -> float:
        """E(P): metres of this route inside ground the pedestrian believes
        dangerous after it has first left it. Leaving from where it stands is
        escape and does not count."""
        if not self.hazard_memory or not pts:
            return 0.0
        exited = not self._believes_dangerous(*pts[0])
        total = 0.0
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            seg = math.hypot(x1 - x0, y1 - y0)
            n = max(1, int(math.ceil(seg / step_m)))
            for k in range(1, n + 1):
                inside = self._believes_dangerous(x0 + (x1 - x0) * k / n,
                                                  y0 + (y1 - y0) * k / n)
                if not exited:
                    exited = not inside
                elif inside:
                    total += seg / n
        return total

    def _safe_route(self, target_mesh, end=None):
        """The route to `target_mesh` if there is one that does not re-enter
        believed danger (E(P) = 0), else None."""
        pts = self._route_points(target_mesh, end)
        if pts is None or self._reexposure(pts) > 0.0:
            return None
        return pts

    def _escape_mesh(self):
        """The nearest triangle, by walking distance, whose centre this
        pedestrian does not believe dangerous (M3 rule 1). None if every
        triangle it can reach is."""
        import heapq
        m = self.model
        now = self._here_mesh()
        if now is None:
            return None

        def centre(t):
            return ((t[0][0] + t[1][0] + t[2][0]) / 3.0,
                    (t[0][1] + t[1][1] + t[2][1]) / 3.0)

        x, y = float(self.xy[0]), float(self.xy[1])
        c0 = centre(now)
        dist = {now: math.hypot(c0[0] - x, c0[1] - y)}
        heap = [(dist[now], 0, now)]
        tie = 1
        while heap:
            d, _, t = heapq.heappop(heap)
            if d > dist.get(t, math.inf):
                continue
            ct = centre(t)
            if not self._believes_dangerous(*ct):
                return t
            for nb in m.adjacent_mesh.get(t, ()):
                cn = centre(nb)
                nd = d + math.hypot(cn[0] - ct[0], cn[1] - ct[1])
                if nd < dist.get(nb, math.inf):
                    dist[nb] = nd
                    heapq.heappush(heap, (nd, tie, nb))
                    tie += 1
        return None

    def _social_cue(self, neighbors) -> int:
        """How many visible neighbours are visibly responding.

        The dominant channel: warning research finds informal contact is how
        most people learn of an emergency. What is seen is people evacuating
        (Kinateder & Warren 2016), not people who have heard something and
        carry on walking. Counting every "acting" neighbour let a warning heard
        before the episode spread over the whole crop and clear the zone with
        no robot, even for a hazard hardly anyone could sense.
        """
        n = 0
        for nb in neighbors:
            if nb is self or getattr(nb, "dead", False):
                continue
            if getattr(nb, "responding", False):
                n += 1
        return n

    def _draw_premovement(self) -> float:
        """Steps of milling before acting, drawn lognormally.

        Fire-safety engineering treats evacuation as pre-movement plus travel
        and reports pre-movement as a broad, right-skewed distribution rather
        than a constant. The tail is the part that matters: the last occupants
        to move are the ones who set the evacuation time, so a model that uses
        a mean loses the thing being designed against.
        """
        return max(1.0, random.lognormvariate(
            math.log(max(1.0, PREMOVEMENT_MEDIAN_STEPS)), PREMOVEMENT_SIGMA))

    def receive_cue(self, source: str) -> None:
        """First warning of any kind (M1): a robot, a sensed hazard, acting
        neighbours, or a warning heard before the episode.

        A share CROWD_NONRESPONSE_FRACTION never act on their own (PADM); the
        rest mill for a lognormal delay, shortened by MILLING_ROBOT_SPEEDUP
        when the cue was a robot.
        """
        if self.awareness not in ("unaware", "nonresponsive"):
            return
        if self.awareness == "unaware":
            self.cue_source = source
            self.cued_at = int(getattr(self.model, "step_count", 0))
        # Non-response is to being told (neighbours, a warning, a robot), not
        # to standing in the hazard: PADM ranks environmental cues as the
        # strongest. Sensing it directly always starts milling, including for
        # someone who had shrugged off a warning.
        if (source != "sensed" and self.awareness == "unaware"
                and random.random() < float(CROWD_NONRESPONSE_FRACTION)):
            self.awareness = "nonresponsive"
            return
        self.act_after = self._draw_premovement()
        if source == "robot":
            # An authoritative, specific instruction is the strongest cue
            # there is, and it is the one channel the policy controls.
            self.act_after *= float(MILLING_ROBOT_SPEEDUP)
        self.awareness = "milling"

    def update_awareness(self, neighbors, robot_near: bool) -> None:
        """Advance unaware -> milling -> acting (or nonresponsive) one step.

        Everyone who has been warned keeps perceiving: a pedestrian who walks
        into the hazard, milling or not, learns where it is.
        """
        if self.type == 3 or self.dead:
            return
        if getattr(self.model, "danger_zone", None) is None:
            return
        acting = self._social_cue(neighbors)
        saturation = float(AWARENESS_SOCIAL_SATURATION)

        if self.awareness == "unaware":
            if robot_near:
                self.receive_cue("robot")
            elif self._sense_hazard():
                self.receive_cue("sensed")
            elif acting:
                p = AWARENESS_P_SOCIAL * min(1.0, acting / saturation)
                if random.random() < p:
                    self.receive_cue("social")
            return

        sensed = self._sense_hazard()
        if self.awareness == "nonresponsive":
            if sensed:
                self.receive_cue("sensed")
            return
        if self.awareness == "milling":
            self.act_after -= 1.0 + float(MILLING_SOCIAL_SPEEDUP) * min(
                1.0, acting / saturation)
            if self.act_after <= 0.0:
                self.awareness = "acting"
                self.ever_acted = True

    def _flow_heading(self, neighbors):
        """The heading of the acting people in sight whose direction has a
        basis (M4): they sensed the hazard or follow a robot. None when there
        are none or they disagree: the mean of their unit headings is shorter
        than CROWD_FLOW_FOLLOW_MIN_ALIGNMENT."""
        hs = self._acting_headings(neighbors, grounded_only=True)
        if not hs:
            return None
        mx = sum(h[0] for h in hs) / len(hs)
        my = sum(h[1] for h in hs) / len(hs)
        n = math.hypot(mx, my)
        if n < float(CROWD_FLOW_FOLLOW_MIN_ALIGNMENT):
            return None
        return mx / n, my / n

    def _flow_goal(self, heading):
        """A walkable point the way the flow is going: a vision radius straight
        ahead when that line is clear, otherwise the centre of the triangle
        within two steps of the navmesh whose direction from here best
        matches `heading` (a point projected into a building costs a wall
        detour every step)."""
        m = self.model
        x, y = float(self.xy[0]), float(self.xy[1])
        # Straight ahead when the way is clear: people keep their own line
        # rather than all converging on one point.
        step = max(self.vision_radius, 4.0)
        px = min(max(x + heading[0] * step, 1.0), m.width - 1.0)
        py = min(max(y + heading[1] * step, 1.0), m.height - 1.0)
        if m.is_free_segment(x, y, px, py, padding=0.0):
            return [px, py]
        now = self._here_mesh()
        if now is None:
            return None
        ring = set(m.adjacent_mesh.get(now, ()))
        for t in list(ring):
            ring.update(m.adjacent_mesh.get(t, ()))
        ring.discard(now)
        best, best_score = None, 0.0
        for t in ring:
            cx = (t[0][0] + t[1][0] + t[2][0]) / 3.0
            cy = (t[0][1] + t[1][1] + t[2][1]) / 3.0
            d = math.hypot(cx - x, cy - y)
            if d < 2.0:
                continue
            score = ((cx - x) * heading[0] + (cy - y) * heading[1]) / d
            if score > best_score:
                best, best_score = [cx, cy], score
        return best

    def which_goal_agent_want(self, neighbors, find_another: bool = False) -> None:
        """Where this pedestrian is going now (docs/behavior_model_design.md).

        In order: the robot it chose to follow (M5, M6); acting and having
        sensed the hazard, the belief-route rule (M3); acting without having
        sensed it, the heading of the acting people in sight (M4); otherwise
        its ordinary trip, leaving out destinations that would take it back
        into ground it believes dangerous.
        """
        # A fixed goal set from outside, used by the validation scenarios in
        # validation/ to drive a corridor or a bottleneck. Never set during
        # training; it is here so the verification suite can exercise the real
        # movement code rather than a copy of it.
        scripted = getattr(self, "scripted_goal", None)
        if scripted is not None:
            self.now_goal = [float(scripted[0]), float(scripted[1])]
            self._dwelling = False
            return

        # The same hook, but routed through the navmesh instead of straight at
        # a point. A bottleneck measured with a straight-line goal is not
        # measuring what the model does in a level: there the pedestrian
        # follows waypoints through the triangulation, and the wall-aware
        # heading slides it along a facade rather than into it, which is very
        # different from aiming it at a doorway from across a room.
        target = getattr(self, "scripted_mesh", None)
        if target is not None:
            self.type = 1
            self._dwelling = False
            self.now_pointing_mesh = target
            self.now_goal = self._explore_randomly(self._here_mesh())
            return

        self.responding = False
        if self.type == 0:
            rb = self._robot_by_id(self.following_robot_id)
            if (rb is not None and self.hazard_memory
                    and self._believes_dangerous(float(rb.xy[0]), float(rb.xy[1]))
                    and not self._believes_dangerous(*self.xy)):
                # The robot has gone into ground this person believes
                # dangerous; it does not follow it back in (6.2).
                rb = None
            if rb is not None:
                mode = self.robot_lead_mode or getattr(rb, "mode", "guide")
                self.now_goal = self._robot_led_goal(rb, mode)
                if mode != "direct":
                    # Within ROBOT_FOLLOW_STANDOFF_M a follower keeps pace
                    # with the robot instead of closing on the point it
                    # stands on: every follower aiming at the same spot
                    # packed them several to the square metre around it.
                    dx = float(rb.xy[0]) - float(self.xy[0])
                    dy = float(rb.xy[1]) - float(self.xy[1])
                    if math.hypot(dx, dy) < float(ROBOT_FOLLOW_STANDOFF_M):
                        vx, vy = getattr(rb, "vel", (0.0, 0.0))
                        self.now_goal = [float(self.xy[0]) + 2.0 * float(vx),
                                         float(self.xy[1]) + 2.0 * float(vy)]
                self._dwelling = False
                self.responding = True
                return
            self._stop_following()

        if self.awareness == "acting":
            if self.hazard_memory:
                self._belief_route_goal()
                return
            heading = self._flow_heading(neighbors)
            if heading is not None:
                goal = self._flow_goal(heading)
                if goal is not None:
                    self._dwelling = False
                    self.now_goal = goal
                    self.responding = True
                    return

        self._trip_goal()

    def _belief_route_goal(self) -> None:
        """M3 for an acting pedestrian that has sensed the hazard.

        Inside the ground it believes dangerous: the nearest way out. Outside:
        once, leave the crop (CROWD_DEPART_PROB) or resume trips, and in
        either case only by routes that do not re-enter believed danger; if
        there is none, wait and look again (see _trip_goal).
        """
        if self._believes_dangerous(*self.xy):
            self.responding = True
            if (self._plan_version != self._memory_version
                    or not self._escaping or self.now_pointing_mesh is None):
                self.now_pointing_mesh = self._escape_mesh()
                self._plan_version = self._memory_version
                self._escaping = True
            self._dwelling = False
            self._dwell_until = 0
            if self.now_pointing_mesh is None:
                self.now_goal = [self.xy[0], self.xy[1]]
                return
            now = self._here_mesh()
            self.now_goal = self._explore_randomly(now)
            return

        if self._escaping:
            # Out of what it believes dangerous: the way out has done its job.
            self._escaping = False
            self.now_pointing_mesh = None
        if self.post_safe_intent is None:
            self.post_safe_intent = ("depart"
                                     if random.random() < float(CROWD_DEPART_PROB)
                                     else "continue")
            self.now_pointing_mesh = None
        if self._plan_version != self._memory_version:
            # It knows more than when it chose where to go. A destination
            # whose route now re-enters believed danger is dropped, and a
            # wait is cut short so it can look again.
            self._plan_version = self._memory_version
            if (self.now_pointing_mesh is not None
                    and self._safe_route(self.now_pointing_mesh) is None):
                self.now_pointing_mesh = None
            self._dwell_until = min(self._dwell_until,
                                    int(getattr(self.model, "step_count", 0)))
        self._trip_goal()
        self.responding = (self.post_safe_intent == "depart"
                           and not self._dwelling)

    def _next_destination(self, now_mesh):
        """The next destination triangle, or None when there is nowhere it can
        go without re-entering ground it believes dangerous.

        Leaving (acting, CROWD_DEPART_PROB drawn): the street mouth with the
        shortest such route. Otherwise an ordinary trip destination (od.py),
        redrawn while its route would re-enter.
        """
        import sim.od as od
        remembered = bool(self.hazard_memory)
        if (self.awareness == "acting" and remembered
                and self.post_safe_intent == "depart"
                and self.model.allow_crowd_departure()):
            best = None
            for gate in od.gates(self.model):
                mesh = min(gate.meshes, key=lambda t: math.hypot(
                    self.xy[0] - (t[0][0] + t[1][0] + t[2][0]) / 3.0,
                    self.xy[1] - (t[0][1] + t[1][1] + t[2][1]) / 3.0))
                end = od.gate_edge_goal(self.model, mesh, self.xy)
                pts = self._safe_route(mesh, end)
                if pts is None:
                    continue
                length = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                             for a, b in zip(pts, pts[1:]))
                if best is None or length < best[0]:
                    best = (length, mesh)
            return best[1] if best else None
        if not remembered:
            return od.choose_destination(self.model, self.xy)
        for _ in range(12):
            mesh = od.choose_destination(self.model, self.xy)
            if mesh is None:
                return None
            if self._safe_route(mesh) is not None:
                return mesh
        # The trip model kept drawing destinations behind what it believes
        # dangerous. Look through the street mouths and a sample of interior
        # destinations before concluding there is nowhere to go: waiting
        # should mean there is no safe way on, not that twelve draws missed.
        candidates = [min(g.meshes, key=lambda t: math.hypot(
            self.xy[0] - (t[0][0] + t[1][0] + t[2][0]) / 3.0,
            self.xy[1] - (t[0][1] + t[1][1] + t[2][1]) / 3.0))
            for g in od.gates(self.model)]
        inner = od.interior_meshes(self.model) or list(self.model.pure_mesh)
        if inner:
            candidates += random.sample(inner, min(24, len(inner)))
        random.shuffle(candidates)
        for mesh in candidates:
            if self._safe_route(mesh) is not None:
                return mesh
        return None

    def _trip_goal(self) -> None:
        """Travel to the current destination; arrive, dwell, choose the next.

        Not wandering: on a trip. The destination is a street mouth or an
        interior errand rather than a uniformly random triangle; see od.py for
        why that distinction matters to the task. When no destination is
        possible without re-entering believed danger, it waits where it is for
        CROWD_SHELTER_REPLAN_STEPS and then looks again.
        """
        import sim.od as od
        now_mesh = self._here_mesh()
        step = int(getattr(self.model, "step_count", 0))

        if self.now_pointing_mesh is not None:
            cx = sum(p[0] for p in self.now_pointing_mesh) / 3.0
            cy = sum(p[1] for p in self.now_pointing_mesh) / 3.0
            gate = od.is_gate_mesh(self.model, self.now_pointing_mesh)
            if gate and self.model.allow_crowd_departure():
                gx, gy = od.gate_edge_goal(
                    self.model, self.now_pointing_mesh, self.xy)
                edge_gap = min(self.xy[0], self.xy[1],
                               self.model.width - self.xy[0],
                               self.model.height - self.xy[1])
                reached = (edge_gap <= CROWD_OUTFLOW_MARGIN_M
                           and math.hypot(self.xy[0] - gx,
                                          self.xy[1] - gy) <= 2.5)
            else:
                reached = math.hypot(self.xy[0] - cx,
                                     self.xy[1] - cy) < 2.0
            if reached:
                arrived = self.now_pointing_mesh
                self.now_pointing_mesh = None
                if self.model.arrive_at_destination(self, arrived):
                    return          # walked out of the crop
                lo, hi = CROWD_DWELL_STEPS
                self._dwell_until = step + random.randint(int(lo), int(hi))

        if step < self._dwell_until:
            # Standing at a destination or waiting for a safe way on. Exempt
            # from the wedge release, which exists for pedestrians that cannot
            # move rather than for ones that are not trying to.
            self._dwelling = True
            self.now_goal = [self.xy[0], self.xy[1]]
            return
        self._dwelling = False

        if self.now_pointing_mesh is None:
            self.now_pointing_mesh = self._next_destination(now_mesh)
            if self.now_pointing_mesh is None:
                self._dwell_until = step + int(CROWD_SHELTER_REPLAN_STEPS)
                self._dwelling = True
                self.now_goal = [self.xy[0], self.xy[1]]
                return

        self.now_goal = self._explore_randomly(now_mesh)

    def _robot_led_goal(self, lead, mode: str):
        """Where a pedestrian influenced by `lead` is trying to get to.

        "guide" is the robot's own position: the robot leads and its path is
        the instruction. "direct" is a heading, so the goal sits ahead of the
        pedestrian along the signalled direction rather than on the robot,
        which is what lets one robot turn a flow without standing in it.

        The directed goal is clamped inside the crop for the same reason the
        hazard-avoidance goal is: a goal outside the world is a permanent push
        into the boundary, and a pedestrian given one hugs the wall for the
        rest of the episode.
        """
        if mode != "direct":
            return [lead.xy[0], lead.xy[1]]
        sx, sy = getattr(lead, "signal_dir", (0.0, 0.0))
        norm = math.hypot(sx, sy)
        if norm < 1e-6:
            # Signalling a direction of nothing. Treat it as no instruction
            # rather than as a goal at the pedestrian's feet, which would
            # leave it with no drive force at all.
            return [lead.xy[0], lead.xy[1]]
        ux, uy = sx / norm, sy / norm
        step = float(ROBOT_DIRECT_GOAL_M)
        m = 1.0
        return [
            min(max(self.xy[0] + ux * step, m), self.model.width - m),
            min(max(self.xy[1] + uy * step, m), self.model.height - m),
        ]

    @staticmethod
    def _portal_waypoint(now_mesh, nxt):
        """Through the edge two adjacent triangles share, just past it.

        A centroid-to-centroid segment can cut through a building corner even
        for adjacent triangles; the shared edge is inside both.
        """
        shared = [p for p in now_mesh if p in nxt]
        cx = sum(p[0] for p in nxt) / 3.0
        cy = sum(p[1] for p in nxt) / 3.0
        if len(shared) >= 2:
            mx = (shared[0][0] + shared[1][0]) / 2.0
            my = (shared[0][1] + shared[1][1]) / 2.0
            return [mx + 0.6 * (cx - mx), my + 0.6 * (cy - my)]
        return [cx, cy]

    def _routed_goal(self, goal):
        """`goal` when it is in sight; otherwise the next navmesh waypoint
        toward the walkable ground nearest it."""
        m = self.model
        x, y = float(self.xy[0]), float(self.xy[1])
        gx = min(max(float(goal[0]), 0.01), m.width - 0.01)
        gy = min(max(float(goal[1]), 0.01), m.height - 0.01)
        if math.hypot(gx - x, gy - y) < 1e-6:
            return [gx, gy]
        if m.is_free_segment(x, y, gx, gy, padding=0.0):
            return [gx, gy]
        now = self._here_mesh()
        target = m.find_mesh((gx, gy)) or self.choice_safe_mesh((gx, gy))
        if now is None or target is None or now == target:
            return [gx, gy]
        nxt = m.next_mesh_from_to(now, target)
        if nxt is None:
            return [gx, gy]
        return self._portal_waypoint(now, nxt)

    def _explore_randomly(self, now_mesh):
        """The next waypoint toward whichever triangle this agent is exploring.

        Retries with a different target when the current one is unreachable,
        rather than returning the agent's own position.

        Returning its own position was a permanent stall: the drive force is
        proportional to the offset from the goal, so a goal at the agent's
        feet produces no force and the goal is not reconsidered while the
        agent has not arrived anywhere. Measured on a 140 m grid with sixty
        pedestrians, five of them stood still for the last two hundred steps
        of a six hundred step run, one to four metres clear of any wall with
        their goal reading exactly their own coordinates.

        Unreachable targets are normal, not exceptional. A destination can
        be separated from the current triangle by disconnected walkable
        pockets, so any given draw may be unroutable.
        """
        # The rounded 1 m grid can name a triangle across a wall or portal.
        # Route from the triangle containing the pedestrian whenever possible.
        now_mesh = self._here_mesh() or now_mesh
        # A pocket the street network does not reach: a sliver along the crop
        # edge or a courtyard that shares only a vertex with the streets. No
        # destination is routable from it, so walk back onto the network.
        if (now_mesh is not None and getattr(self.model, "pure_mesh", None)
                and now_mesh not in self.model.main_walkable_component()):
            rejoin = self._nearest_main_mesh_centre()
            if rejoin is not None:
                return rejoin
        for _ in range(8):
            goal_mesh = self.now_pointing_mesh
            if goal_mesh is not None:
                import sim.od as od
                if (goal_mesh == now_mesh
                        and self.model.allow_crowd_departure()
                        and od.is_gate_mesh(self.model, goal_mesh)):
                    return list(od.gate_edge_goal(self.model, goal_mesh,
                                                  self.xy))
                if goal_mesh != now_mesh:
                    nxt = self.model.next_mesh_from_to(now_mesh, goal_mesh)
                    if nxt is not None:
                        # A centroid-to-centroid segment can cut through a
                        # building corner even for adjacent triangles.
                        return self._portal_waypoint(now_mesh, nxt)
                else:
                    # Stay on this trip until the arrival check says the
                    # destination was reached. A large triangle is not its
                    # centroid or gate.
                    return [sum(p[0] for p in goal_mesh) / 3.0,
                            sum(p[1] for p in goal_mesh) / 3.0]
            # Unreachable: pick another destination from the same trip model,
            # not a uniform triangle, and by the same belief-route rule.
            if self.model.pure_mesh:
                self.now_pointing_mesh = self._next_destination(now_mesh)
                if self.now_pointing_mesh is None:
                    break
            else:
                break

        # Nothing routable anywhere. Step toward a neighbouring triangle so
        # the agent still moves and can be pushed out of wherever it is,
        # instead of freezing where it stands.
        neighbours = self.model.adjacent_mesh.get(now_mesh) if now_mesh else None
        if neighbours:
            nb = random.choice(neighbours)
            return [(nb[0][0]+nb[1][0]+nb[2][0])/3.0,
                    (nb[0][1]+nb[1][1]+nb[2][1])/3.0]
        # A triangle with no neighbours: a sliver along the crop edge or a
        # courtyard that shares only a vertex with the streets. Returning the
        # agent's own position here gave a goal that followed it around, so
        # it stood still against the edge for the rest of the episode
        # (measured on OSM crops: 3 of the 8 remaining wedged pedestrians).
        # Walk back onto the connected street network instead.
        rejoin = self._nearest_main_mesh_centre()
        if rejoin is not None:
            return rejoin
        return [self.xy[0], self.xy[1]]

    def _nearest_main_mesh_centre(self):
        m = self.model
        try:
            main = m.main_walkable_component()
        except Exception:
            return None
        if not main:
            return None
        x, y = float(self.xy[0]), float(self.xy[1])
        best, best_d = None, math.inf
        for t in main:
            cx = (t[0][0] + t[1][0] + t[2][0]) / 3.0
            cy = (t[0][1] + t[1][1] + t[2][1]) / 3.0
            d = (cx - x) ** 2 + (cy - y) ** 2
            if d < best_d:
                best, best_d = [cx, cy], d
        return best


  
class RobotAgent(CrowdAgent):
    

    def __init__(self, unique_id, model, pos, type1, robot_index: int = 0):
        super().__init__(unique_id, model, pos, type1)
        # Position in the team. The joint observation, the joint action and
        # the centralised critic are all indexed by it, so it has to be stable
        # for the whole episode and unique within the team.
        self.robot_index = int(robot_index)
        self.action = [0, 0, "GUIDE"]
        self.past_xy = deque(maxlen=20)
        self.collision_check = 0
        # Why the last move calls for a new decision ("arrived", "blocked")
        # or None; see _decision_event. Cleared by each new action.
        self.decision_event = None
        self.detect_abnormal_order = 0
        self.is_game_finished = 0

        self.robot_waypoint = [0, 0]
        self.now_exploration = 0

        self.acc = [0, 0]
        self.vel = [0, 0]
        self.body_radius = ROBOT_BODY_RADIUS

        # What this robot is signalling, and in which direction when the
        # signal is a heading. Set by the policy through `set_signal`; "off"
        # until then, so a robot that has not acted yet influences nobody.
        self.mode = "off"
        self.signal_dir = (0.0, 0.0)
        self.compliance_form_factor = ROBOT_COMPLIANCE_FORM_FACTOR
        self.vision_radius = ROBOT_VISION

        #self.model.space.add(self.unique_id, self.xy, self.radius, ref=self, vel=(0,0,0,0))

        self.desired_speed_a = 2
        self.target_agent = None
    
    # ------------------------------------------------------------
    # 외부에서 호출되는 단일 정책 함수
    # ------------------------------------------------------------

    # def robot_policy_go_and_back(self):
    #     if (self.target_agent == None):
    #         max_d = -1 
    #         max_d_ag = None
    #         for ag in self.model.crowds:
    #             if not ag.dead:
    #                 d = self.point_to_point_distance(self.xy, ag.xy)
    #                 if d > max_d:
    #                     max_d = d
    #                     max_d_ag = ag
    #         if max_d_ag is not None:
    #             self.target_agent = max_d_ag

    #     if (self.target_agent == None):
    #         return
        
    #     if (self.target_agent.dead):
    #         self.target_agent = None
    #         return

    #     goal = [0, 0]
    #     if (self.point_to_point_distance(self.xy, self.target_agent.xy) < 5):
    #         goal = self.model.exit_point[0]
    #     else :
    #         goal = self.target_agent.xy

    #     goal_mesh = self.model.match_grid_to_mesh[int(round(goal[0])), int(round(goal[1]))]
    #     now_mesh = self.model.match_grid_to_mesh[int(round(self.xy[0])), int(round(self.xy[1]))]
    #     next_mesh = self.model.next_vertex_matrix[now_mesh][goal_mesh]
    #     if(now_mesh == next_mesh):
    #         goal_x = goal[0] - self.xy[0]
    #         goal_y = goal[1] - self.xy[1]
            
    #     else:
    #         next_mesh_middle = ((next_mesh[0][0]+next_mesh[1][0]+next_mesh[2][0])/3, (next_mesh[0][1]+next_mesh[1][1]+next_mesh[2][1])/3)
    #         goal_x = next_mesh_middle[0] - self.xy[0]
    #         goal_y = next_mesh_middle[1] - self.xy[1]

    #     goal_x = ROBOT_SPEED_MAX * goal_x / math.sqrt(pow(goal_x, 2) + pow(goal_y, 2))
    #     goal_y = ROBOT_SPEED_MAX * goal_y / math.sqrt(pow(goal_x, 2) + pow(goal_y, 2))
    #     self.receive_action([goal_x, goal_y])


    def robot_policy_going_exit(self):
        ed_idx, q, d = self.model.nearest_exit(self.xy)
        goal = q
        if self.point_to_point_distance(self.xy, goal) < 2:
            self.receive_action([0, 0])  # stop
        
        else :
            goal_mesh = self.model.match_grid_to_mesh[int(round(goal[0])), int(round(goal[1]))]
            now_mesh = self.model.match_grid_to_mesh[int(round(self.xy[0])), int(round(self.xy[1]))]
            next_mesh = self.model.next_vertex_matrix[now_mesh][goal_mesh]
            if(now_mesh == next_mesh):
                goal_x = goal[0] - self.xy[0]
                goal_y = goal[1] - self.xy[1]
                
            else:
                next_mesh_middle = ((next_mesh[0][0]+next_mesh[1][0]+next_mesh[2][0])/3, (next_mesh[0][1]+next_mesh[1][1]+next_mesh[2][1])/3)
                goal_x = next_mesh_middle[0] - self.xy[0]
                goal_y = next_mesh_middle[1] - self.xy[1]

            goal_x = 1* goal_x / math.sqrt(pow(goal_x, 2) + pow(goal_y, 2))
            goal_y = 1* goal_y / math.sqrt(pow(goal_x, 2) + pow(goal_y, 2))
            self.receive_action([goal_x, goal_y])
    

    def set_signal(self, mode, sx: float = 0.0, sy: float = 0.0) -> None:
        """Set what this robot is signalling.

        Separate from `receive_action`, which is about where the robot moves.
        The two halves of the action are independent: a robot can reposition
        silently, lead while moving, or stand still and point.
        """
        mode = str(mode)
        if mode not in ROBOT_MODES:
            raise ValueError(f"unknown robot mode {mode!r}; "
                             f"expected one of {ROBOT_MODES}")
        self.mode = mode
        norm = math.hypot(float(sx), float(sy))
        if mode == "direct" and norm > 1e-6:
            self.signal_dir = (float(sx) / norm, float(sy) / norm)
        else:
            self.signal_dir = (0.0, 0.0)

    def receive_action(self, action, speed_fraction: float = 1.0):
                
        
        direction_probs = action[0]
        

        self.action[0] = action[0]
        self.action[1] = action[1]
        # A new decision: under ROBOT_ACTION_MODE = "waypoint" the target is
        # set from it, relative to where the robot stands now, on its next
        # move.
        self._waypoint = None
        self.decision_event = None
        # Under "waypoint", the share of ROBOT_SPEED_MAX to walk there at.
        self.speed_fraction = min(1.0, max(0.0, float(speed_fraction)))

        
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
    
    def _move_robot_with_walls(self, vx, vy, dt):
        """Move a robot without an impulse away from a nearby wall.

        Commands have bounded speed. A blocked normal component stops at the
        wall while the tangent component can continue; this is a kinematic
        no-penetration rule, not a wall force or a crowd unwedge teleport.
        """
        x, y = float(self.xy[0]), float(self.xy[1])
        self.collision_check = 0
        distance = max(abs(vx * dt), abs(vy * dt))
        if distance <= 1e-12:
            return [x, y]
        steps = max(1, int(math.ceil(distance / 0.1)))
        sx, sy = vx * dt / steps, vy * dt / steps

        def clear(x0, y0, x1, y1):
            if hasattr(self.model, "is_free_segment"):
                if self.model.is_free_segment(
                        x0, y0, x1, y1, padding=self.body_radius):
                    return True
                # Exact tangency is reported as blocked by <= radius. Permit
                # an outward move only if the whole segment immediately
                # after contact is body-clear; never allow a wall crossing.
                ex = x0 + (x1 - x0) * 1e-5
                ey = y0 + (y1 - y0) * 1e-5
                return (self.model.is_free_point(
                    ex, ey, padding=self.body_radius)
                    and self.model.is_free_segment(
                        ex, ey, x1, y1, padding=self.body_radius))
            return self.model.is_free_point(
                x1, y1, padding=self.body_radius)

        def separating(x0, y0, x1, y1):
            # A body already overlapping a wall (spawned or pushed there) is
            # otherwise blocked in every direction: every segment starts
            # closer than its radius. Let it move while each step takes it
            # further from the wall. Steps are 0.1 m, so this cannot carry it
            # through a building.
            if not hasattr(self.model, "obstacle_clearance"):
                return False
            c0 = self.model.obstacle_clearance(x0, y0)
            if c0 >= self.body_radius:
                return False
            return self.model.obstacle_clearance(x1, y1) > c0 + 1e-6

        for _ in range(steps):
            tx, ty = x + sx, y + sy
            if clear(x, y, tx, ty) or separating(x, y, tx, ty):
                x, y = tx, ty
                continue
            self.collision_check = 1
            # Preserve any available tangential motion. A diagonal command
            # toward a façade should slide, not stop or reflect backwards.
            if sx and clear(x, y, x + sx, y):
                x += sx
            if sy and clear(x, y, x, y + sy):
                y += sy
        return [x, y]

    def heading_target(self):
        """Where this robot is going under its current command: its waypoint
        under ROBOT_ACTION_MODE = "waypoint", otherwise the point the
        commanded velocity reaches over the longest decision interval, walls
        ignored. Its own position when it has no command yet."""
        x, y = float(self.xy[0]), float(self.xy[1])
        if ROBOT_ACTION_MODE == "waypoint":
            wp = getattr(self, "_waypoint", None)
            return (float(wp[0][0]), float(wp[0][1])) if wp else (x, y)
        ax, ay = float(self.action[0]), float(self.action[1])
        n = math.hypot(ax, ay)
        if not math.isfinite(n) or n < 1e-9:
            return (x, y)
        if n > 1.0:
            ax, ay = ax / n, ay / n
        reach = (float(ROBOT_SPEED_MAX) * float(ROBOT_TIME_STEP)
                 * _decision_max_steps(_config))
        return (x + ax * reach, y + ay * reach)

    # Arrival tolerance for a waypoint, in metres.
    WAYPOINT_ARRIVAL_M = 0.3
    # A move against a wall that covers less than this share of its command
    # is blocked. Sliding along a wall at 45 degrees still covers 0.71.
    BLOCKED_PROGRESS = 0.5

    def _decision_event(self, x0, y0, vx, vy, dt):
        """Why this robot needs a new decision after the move from (x0, y0)
        at commanded velocity (vx, vy), or None: "arrived" at its waypoint,
        or "blocked" by a wall. Whether the rollout acts on it is
        ROBOT_DECISION_ON_EVENTS_<MODE>."""
        x, y = float(self.xy[0]), float(self.xy[1])
        wp = getattr(self, "_waypoint", None)
        if ROBOT_ACTION_MODE == "waypoint" and wp is not None:
            (gx, gy), _ = wp
            if math.hypot(gx - x, gy - y) < self.WAYPOINT_ARRIVAL_M:
                return "arrived"
        want = math.hypot(vx, vy) * dt
        if (self.collision_check and want > 1e-6
                and math.hypot(x - x0, y - y0) < self.BLOCKED_PROGRESS * want):
            return "blocked"
        return None

    def _waypoint_velocity(self, dt):
        """Velocity toward the current waypoint along the navmesh.

        The action's two movement numbers, each in [-2, 2], are an offset of
        up to ROBOT_WAYPOINT_RANGE_M per axis from where the robot stood when
        it received them. Straight there when the body-clear line is free;
        otherwise through the next portal's safest point, at the action's
        speed (speed_fraction of ROBOT_SPEED_MAX). Slows to land on the
        target instead of oscillating across it.
        """
        m = self.model
        r = self.body_radius
        x, y = float(self.xy[0]), float(self.xy[1])
        if getattr(self, "_waypoint", None) is None:
            ax = max(-2.0, min(2.0, float(self.action[0]))) / 2.0
            ay = max(-2.0, min(2.0, float(self.action[1]))) / 2.0
            rng = float(ROBOT_WAYPOINT_RANGE_M)
            self._waypoint = m.nearest_main_ground(x + ax * rng, y + ay * rng,
                                                   r)
        (gx, gy), goal_tri = self._waypoint
        d_goal = math.hypot(gx - x, gy - y)
        if d_goal < self.WAYPOINT_ARRIVAL_M:
            return 0.0, 0.0
        aim = (gx, gy)
        if not m.is_free_segment(x, y, gx, gy, padding=r):
            now = m.find_mesh(self.xy)
            if now is not None and goal_tri is not None and now != goal_tri:
                nxt = m.next_mesh_from_to(now, goal_tri)
                if nxt is not None:
                    aim = m.portal_point(now, nxt, (x, y), (gx, gy), r)
                    # Standing on the portal the containment test can still
                    # name the triangle being left, and the portal would then
                    # be a target at the robot's feet. Aim into the next one.
                    if math.hypot(aim[0] - x, aim[1] - y) < 0.3:
                        aim = ((nxt[0][0] + nxt[1][0] + nxt[2][0]) / 3.0,
                               (nxt[0][1] + nxt[1][1] + nxt[2][1]) / 3.0)
        dx, dy = aim[0] - x, aim[1] - y
        n = math.hypot(dx, dy)
        if n < 1e-9:
            return 0.0, 0.0
        dx, dy = dx / n, dy / n
        # Touching a wall, every straight line that leans into it is refused
        # by the mover and the robot stands still against it (1 route in 300
        # on the training crops). Drop the part of the heading that goes into
        # the wall and lean slightly away from it instead.
        face = self._nearest_wall_normal(x, y)
        if face is not None:
            ux, uy = face
            into = dx * ux + dy * uy
            if into < 0.0:
                dx, dy = dx - into * ux + 0.3 * ux, dy - into * uy + 0.3 * uy
                n = math.hypot(dx, dy) or 1e-9
                dx, dy = dx / n, dy / n
        speed = min(float(ROBOT_SPEED_MAX)
                    * float(getattr(self, "speed_fraction", 1.0)),
                    d_goal / max(dt, 1e-9))
        return speed * dx, speed * dy

    def planned_path(self, max_points: int = 64):
        """The route _waypoint_velocity will follow from here to the current
        waypoint, as [(x, y), ...] starting at the robot: straight to the
        target once the body-clear line is free, otherwise through the next
        portal's crossing point, the same rule the robot steers by. Empty
        without a waypoint. For drawing; the robot does not read it."""
        wp = getattr(self, "_waypoint", None)
        if wp is None:
            return []
        m = self.model
        r = self.body_radius
        (gx, gy), goal_tri = wp
        cur = (float(self.xy[0]), float(self.xy[1]))
        pts = [cur]
        tri = m.find_mesh(cur)
        for _ in range(max_points):
            if (tri is None or goal_tri is None or tri == goal_tri
                    or m.is_free_segment(cur[0], cur[1], gx, gy, padding=r)):
                break
            nxt = m.next_mesh_from_to(tri, goal_tri)
            if nxt is None:
                break
            cur = tuple(m.portal_point(tri, nxt, cur, (gx, gy), r))
            pts.append(cur)
            tri = nxt
        pts.append((float(gx), float(gy)))
        return pts

    def robot_policy_Q(self):
        if (math.hypot(self.xy[0] - self.robot_waypoint[0],
                       self.xy[1] - self.robot_waypoint[1]) < 2):
            self.now_exploration = 0
            self.robot_waypoint = [0, 0]

        self.previous_danger = getattr(self, "danger", 1e9)
        self.danger = self.model.escape_distance(self.xy)
        if self.model.should_finish():
            self.is_game_finished = 1

        if self.robot_initialized == 0:
            self.robot_initialized = 1
            return tuple(self.xy)
        self.past_xy.append(self.xy)

        ax, ay = float(self.action[0]), float(self.action[1])
        command_norm = math.hypot(ax, ay)
        if not math.isfinite(command_norm):
            ax = ay = 0.0
        elif command_norm > 1.0:
            ax, ay = ax / command_norm, ay / command_norm

        start_x, start_y = self.xy
        dt = float(ROBOT_TIME_STEP)
        if ROBOT_ACTION_MODE == "waypoint":
            vx, vy = self._waypoint_velocity(dt)
        else:
            vx, vy = ROBOT_SPEED_MAX * ax, ROBOT_SPEED_MAX * ay
        self.xy = self._move_robot_with_walls(vx, vy, dt)
        self.vel = [(self.xy[0] - start_x) / dt,
                    (self.xy[1] - start_y) / dt]
        self.model.space.clamp(self.xy)
        self.decision_event = self._decision_event(start_x, start_y,
                                                   vx, vy, dt)
        return tuple(self.xy)
