import json
import random
from typing import List, Tuple, Dict, Any, Optional

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

# ---------------------------
# Utilities
# ---------------------------

def _poly_to_points(poly: Polygon) -> List[List[int]]:
    """Shapely Polygon -> [[x,y], ...] (no duplicated last point)"""
    coords = list(poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [[int(round(x)), int(round(y))] for x, y in coords]

def _rect_to_points(x0: int, y0: int, x1: int, y1: int) -> List[List[int]]:
    """Axis-aligned rectangle -> 4 points (no closure)"""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

def _valid_polygon(poly: Polygon, min_area: float = 10.0) -> bool:
    return (poly.is_valid and (not poly.is_empty) and poly.area >= min_area)

def _too_close_or_intersect(new_poly: Polygon, polys: List[Polygon], min_gap: float) -> bool:
    """True if intersects OR distance < min_gap with any existing polygon."""
    for p in polys:
        if new_poly.intersects(p):
            return True
        if new_poly.distance(p) < min_gap:
            return True
    return False

# ---------------------------
# Exit generation
# ---------------------------

def _random_exit_on_side(
    W: int, H: int,
    side: str,
    size_min: int = 5,
    size_max: int = 10
) -> Polygon:
    ew = random.randint(size_min, size_max)
    eh = random.randint(size_min, size_max)

    if side == "left":
        x0 = 0
        x1 = ew
        y0 = random.randint(0, H - eh)
        y1 = y0 + eh
    elif side == "right":
        x1 = W
        x0 = W - ew
        y0 = random.randint(0, H - eh)
        y1 = y0 + eh
    elif side == "bottom":
        y0 = 0
        y1 = eh
        x0 = random.randint(0, W - ew)
        x1 = x0 + ew
    elif side == "top":
        y1 = H
        y0 = H - eh
        x0 = random.randint(0, W - ew)
        x1 = x0 + ew
    else:
        raise ValueError("side must be one of left/right/top/bottom")

    return box(x0, y0, x1, y1)

def generate_two_exits(
    W: int, H: int,
    size_min: int = 5,
    size_max: int = 10,
    max_tries: int = 500
) -> List[Polygon]:
    """
    - exits are rectangles
    - each is flush to a wall
    - two exits cannot be on the same wall
    """
    sides = ["left", "right", "bottom", "top"]

    for _ in range(max_tries):
        side1, side2 = random.sample(sides, 2)  # different walls
        e1 = _random_exit_on_side(W, H, side1, size_min, size_max)
        e2 = _random_exit_on_side(W, H, side2, size_min, size_max)

        # they might still overlap at corners if sizes are huge, but walls differ so usually not.
        if not e1.intersects(e2):
            return [e1, e2]

    raise RuntimeError("Failed to generate two valid exits within max_tries")

# ---------------------------
# Obstacle generation
# ---------------------------

def _random_rect_obstacle(W: int, H: int, margin: int = 1) -> Polygon:
    """
    Random axis-aligned rectangle inside the map.
    margin: keep it away from absolute border slightly (optional)
    """
    w = random.randint(4, 30)
    h = random.randint(4, 30)
    x0 = random.randint(margin, max(margin, W - margin - w))
    y0 = random.randint(margin, max(margin, H - margin - h))
    return box(x0, y0, x0 + w, y0 + h)

def _random_convex_obstacle(W: int, H: int, margin: int = 1) -> Polygon:
    """
    Random convex polygon by sampling points and taking convex hull.
    """
    n = random.randint(4, 10)
    pts = []
    for _ in range(n):
        x = random.randint(margin, W - margin)
        y = random.randint(margin, H - margin)
        pts.append((x, y))
    hull = Polygon(pts).convex_hull
    return hull

def generate_obstacles(
    W: int, H: int,
    exits: List[Polygon],
    num_obstacles: int = 7,
    min_gap: float = 5.0,
    max_tries: int = 5000,
    keep_gap_from_exits: float = 0.0,  # 원하면 5.0 등으로 설정 가능
) -> List[Polygon]:
    obstacles: List[Polygon] = []

    for _ in range(max_tries):
        if len(obstacles) >= num_obstacles:
            break

        # shape choice
        if random.random() < 0.6:
            cand = _random_rect_obstacle(W, H, margin=1)
        else:
            cand = _random_convex_obstacle(W, H, margin=1)

        if not _valid_polygon(cand, min_area=20.0):
            continue

        # not overlap exits, optionally keep distance
        bad = False
        for e in exits:
            if cand.intersects(e):
                bad = True
                break
            if keep_gap_from_exits > 0 and cand.distance(e) < keep_gap_from_exits:
                bad = True
                break
        if bad:
            continue

        # not overlap and not too close to other obstacles
        if _too_close_or_intersect(cand, obstacles, min_gap=min_gap):
            continue

        # accept
        obstacles.append(cand)

    if len(obstacles) < num_obstacles:
        raise RuntimeError(f"Could only place {len(obstacles)}/{num_obstacles} obstacles. "
                           f"Try reducing num_obstacles or min_gap.")

    return obstacles

# ---------------------------
# Main map generator
# ---------------------------

def generate_random_map(
    width: int = 100,
    height: int = 100,
    num_obstacles: int = 7,
    min_obstacle_gap: float = 5.0,
    exit_size_min: int = 5,
    exit_size_max: int = 10,
    keep_gap_from_exits: float = 0.0,  # 필요하면 5로
    seed: Optional[int] = None
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)

    exits = generate_two_exits(width, height, exit_size_min, exit_size_max)

    obstacles = generate_obstacles(
        width, height,
        exits=exits,
        num_obstacles=num_obstacles,
        min_gap=min_obstacle_gap,
        keep_gap_from_exits=keep_gap_from_exits,
    )

    # to json-like structure
    exits_pts = [_poly_to_points(e) for e in exits]
    obstacles_pts = [_poly_to_points(o) for o in obstacles]

    return {
        "width": width,
        "height": height,
        "obstacles": obstacles_pts,
        "exits": exits_pts
    }

# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    m = generate_random_map(
        width=100,
        height=100,
        num_obstacles=7,
        min_obstacle_gap=5.0,
        exit_size_min=5,
        exit_size_max=10,
        keep_gap_from_exits=0.0,
        seed=42
    )
    print(json.dumps(m, indent=2))
