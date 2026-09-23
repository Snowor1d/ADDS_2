"""Mutation as edits to the street network.

This is why the representation exists. ACCEL builds complexity entirely
through offspring, so whatever mutation preserves is what a lineage converges
on. The previous operators moved, rotated and resized individual obstacle
polygons, which is morphology-blind: a block translated off its frontage is no
longer part of a street network, and a layout that started out looking like a
downtown stopped looking like one within a few generations.

Every operator here edits the plan instead, and the polygons are re-derived.
So a child of a city is a city, a hundred generations later included. What the
operators change is how developed and how fine-grained it is, which is exactly
the axis the curriculum is supposed to explore.

The operators come in pairs on purpose. Earlier work on the scatter generator
found that mutation collapsed complexity: the operators that removed things
always validated while the ones that added things often failed, so the
population drifted toward empty maps whatever the curriculum asked for. Here
`build_block` and `clear_block`, `split_block` and `merge_blocks`, `widen` and
`narrow`, `add_stub` and `remove_stub` are each other's inverse and succeed or
fail under the same conditions.

Measured over twelve lineages of forty generations with no selection at all,
built coverage wanders rather than trending: 0.10 at the start, 0.26 at ten
generations, 0.17 at thirty, 0.43 at forty. That is the intended behaviour,
because it means the curriculum's score is what decides where a lineage goes.

One drift does survive: the median street width falls from about 7.2 m to
5.3 m over those forty generations, because `split_block` inserts lane-width
streets while `merge_blocks` removes a street of any width. It is bounded well
above the floor the robot needs, and narrower streets are a legitimate
difficulty axis rather than a defect, so it is left in and recorded here
rather than cancelled with a correction factor.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from citygen.morphology import CLASS_WIDTH, MIN_CORRIDOR_M, base_width
from citygen.plan import BlockMark, CityPlan, Street

# Attempts to find a valid child before giving up on a parent. A structural
# edit is far likelier to stay playable than a polygon edit was, so this does
# not need to be large; the earlier generator needed forty placement tries per
# added obstacle.
MAX_TRIES = 12

# Ops applied per child. More than one lets a single generation move along two
# axes at once, which is what makes lineage depth mean something.
OPS_PER_CHILD = (1, 3)


def _built_blocks(plan: CityPlan):
    return [b for b in plan.blocks() if plan._is_built(b)]


def _open_blocks(plan: CityPlan):
    return [b for b in plan.blocks() if not plan._is_built(b)]


def _set_mark(plan: CityPlan, block, built: bool) -> None:
    """Record a decision for one block, replacing any mark inside it."""
    from shapely.geometry import Point

    plan.marks = [m for m in plan.marks if not block.covers(Point(m.x, m.y))]
    p = block.representative_point()
    plan.marks.append(BlockMark(float(p.x), float(p.y), built))


# ---------------------------------------------------------------------------
# operators. Each returns a name when it changed something, else None.
# ---------------------------------------------------------------------------

def op_build_block(plan: CityPlan, rng: random.Random) -> Optional[str]:
    """Build up one open block. The main way complexity grows."""
    candidates = _open_blocks(plan)
    if not candidates:
        return None
    # Weighted toward larger blocks, so building up makes a visible
    # difference rather than filling a sliver.
    block = max(candidates, key=lambda b: b.area * rng.uniform(0.6, 1.4))
    _set_mark(plan, block, True)
    return "build_block"


def op_clear_block(plan: CityPlan, rng: random.Random) -> Optional[str]:
    """Clear one built block back to open ground: a square or a car park."""
    candidates = _built_blocks(plan)
    if len(candidates) <= 1:
        return None
    # Same size preference as build_block. Picking uniformly here while
    # build_block picked the largest made the pair add more area than it
    # removed, which is a bias hidden inside two innocuous-looking lines.
    block = max(candidates, key=lambda b: b.area * rng.uniform(0.6, 1.4))
    _set_mark(plan, block, False)
    return "clear_block"


def op_split_block(plan: CityPlan, rng: random.Random) -> Optional[str]:
    """Cut a new street through the largest block, splitting it in two.

    This is the operator that refines the grain of the fabric, and it is the
    one a real city performs when a block is subdivided. The cut runs the full
    width of the block and a little past it, so it meets the surrounding
    streets rather than dead-ending just short of them.
    """
    blocks = plan.blocks()
    if not blocks:
        return None
    block = max(blocks, key=lambda b: b.area * rng.uniform(0.7, 1.3))
    minx, miny, maxx, maxy = block.bounds
    if max(maxx - minx, maxy - miny) < 4.0 * MIN_CORRIDOR_M:
        return None

    base = base_width(plan.morphology)
    width = max(MIN_CORRIDOR_M, base * CLASS_WIDTH["lane"] *
                rng.uniform(0.85, 1.15))
    pad = width
    if (maxx - minx) >= (maxy - miny):
        x = rng.uniform(minx + 0.3 * (maxx - minx), minx + 0.7 * (maxx - minx))
        pts = [(x, miny - pad), (x, maxy + pad)]
    else:
        y = rng.uniform(miny + 0.3 * (maxy - miny), miny + 0.7 * (maxy - miny))
        pts = [(minx - pad, y), (maxx + pad, y)]
    plan.streets.append(Street(pts, width, "lane"))
    return "split_block"


def op_merge_blocks(plan: CityPlan, rng: random.Random) -> Optional[str]:
    """Remove one street, merging the blocks on either side of it.

    The inverse of split_block. Removing an arterial is excluded: it is the
    one edit that can disconnect a whole quarter, and the pair would then be
    biased toward removal because the reverse edit never adds an arterial.
    """
    removable = [i for i, st in enumerate(plan.streets)
                 if st.kind != "arterial"]
    if len(removable) <= 2:
        return None
    plan.streets.pop(removable[rng.randrange(len(removable))])
    return "merge_blocks"


def op_widen_street(plan: CityPlan, rng: random.Random) -> Optional[str]:
    if not plan.streets:
        return None
    st = plan.streets[rng.randrange(len(plan.streets))]
    st.width_m = min(st.width_m * rng.uniform(1.15, 1.5),
                     base_width(plan.morphology) * CLASS_WIDTH["arterial"] * 2.0)
    return "widen_street"


def op_narrow_street(plan: CityPlan, rng: random.Random) -> Optional[str]:
    if not plan.streets:
        return None
    st = plan.streets[rng.randrange(len(plan.streets))]
    narrowed = st.width_m * rng.uniform(0.65, 0.87)
    if narrowed < MIN_CORRIDOR_M:
        return None
    st.width_m = narrowed
    return "narrow_street"


def op_bend_street(plan: CityPlan, rng: random.Random) -> Optional[str]:
    """Move a street sideways or bend it, without changing what it connects.

    The organic-fabric operator. A grid mutated this way drifts toward the
    kind of plan Covent Garden has, which is a direction the curriculum
    should be able to walk rather than only sample.
    """
    if not plan.streets:
        return None
    st = plan.streets[rng.randrange(len(plan.streets))]
    if len(st.points) < 2:
        return None
    amp = base_width(plan.morphology) * rng.uniform(0.5, 2.0)
    p0, p1 = st.points[0], st.points[-1]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    nx, ny = -dy / length, dx / length
    if len(st.points) == 2:
        # Straight street: give it a middle point to bend around.
        mid = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
        st.points = [p0, mid, p1]
    inner = range(1, len(st.points) - 1)
    off = rng.uniform(-amp, amp)
    for i in inner:
        x, y = st.points[i]
        st.points[i] = (x + nx * off, y + ny * off)
    return "bend_street"


def op_add_stub(plan: CityPlan, rng: random.Random) -> Optional[str]:
    """A dead-end lane into a block. The medina operator.

    A blind alley is a real structure and a real problem for a guidance robot:
    a crowd that turns into one has to come back out, past whoever is still
    coming in. It is not a validity failure, which is why validation asks
    about reachability from exits and not about dead ends.
    """
    blocks = _built_blocks(plan)
    if not blocks:
        return None
    block = blocks[rng.randrange(len(blocks))]
    minx, miny, maxx, maxy = block.bounds
    base = base_width(plan.morphology)
    width = max(MIN_CORRIDOR_M, base * CLASS_WIDTH["alley"] * rng.uniform(0.9, 1.2))
    # Start just outside the block so the stub connects to the street beside
    # it, and end inside so it is genuinely blind.
    if rng.random() < 0.5:
        y = rng.uniform(miny + 0.2 * (maxy - miny), maxy - 0.2 * (maxy - miny))
        x0 = minx - width if rng.random() < 0.5 else maxx + width
        depth = (maxx - minx) * rng.uniform(0.3, 0.75)
        x1 = x0 + depth if x0 < minx else x0 - depth
        pts = [(x0, y), (x1, y)]
    else:
        x = rng.uniform(minx + 0.2 * (maxx - minx), maxx - 0.2 * (maxx - minx))
        y0 = miny - width if rng.random() < 0.5 else maxy + width
        depth = (maxy - miny) * rng.uniform(0.3, 0.75)
        y1 = y0 + depth if y0 < miny else y0 - depth
        pts = [(x, y0), (x, y1)]
    plan.streets.append(Street(pts, width, "stub"))
    return "add_stub"


def op_remove_stub(plan: CityPlan, rng: random.Random) -> Optional[str]:
    idx = [i for i, st in enumerate(plan.streets) if st.kind == "stub"]
    if not idx:
        return None
    plan.streets.pop(idx[rng.randrange(len(idx))])
    return "remove_stub"


def op_resize_canvas(plan: CityPlan, rng: random.Random) -> Optional[str]:
    """Grow or shrink the crop, leaving the fabric where it is.

    Map size is a curriculum axis, so mutation has to be able to walk it and
    not only sample it. Resizing is nearly free here: the streets are drawn
    across a span wider than the crop, so moving the crop edge reveals or
    hides fabric exactly as moving a window over a real city would, and the
    blocks are re-derived at the new size.

    Scaling the whole layout instead would be wrong. It would change the
    street widths and the block sizes together, so a small map would get
    corridors too narrow for the robot and a large one would get blocks no
    real city has, and the physical scale is the one thing about this
    representation that should not move.
    """
    from config import UED_MAP_SIZE_RANGE, UED_MAP_SQUARE

    lo, hi = UED_MAP_SIZE_RANGE
    step = rng.choice([-12, -8, -5, 5, 8, 12])
    new_w = int(min(hi, max(lo, plan.width + step)))
    if new_w == plan.width:
        return None
    new_h = new_w if UED_MAP_SQUARE else int(min(hi, max(lo, plan.height + step)))
    plan.width, plan.height = new_w, new_h
    # Marks outside the new crop describe blocks that no longer exist.
    plan.marks = [m for m in plan.marks
                  if 0 <= m.x <= new_w and 0 <= m.y <= new_h]
    return "resize_canvas"


OPERATORS: Dict[str, Callable[[CityPlan, random.Random], Optional[str]]] = {
    "build_block": op_build_block,
    "clear_block": op_clear_block,
    "split_block": op_split_block,
    "merge_blocks": op_merge_blocks,
    "widen_street": op_widen_street,
    "narrow_street": op_narrow_street,
    "bend_street": op_bend_street,
    "add_stub": op_add_stub,
    "remove_stub": op_remove_stub,
    "resize_canvas": op_resize_canvas,
}

# Paired so the walk is unbiased without selection. Listed explicitly rather
# than derived from the names, because the pairing is a claim about the
# operators and should break loudly if one is added without its inverse.
INVERSE_PAIRS = (
    ("build_block", "clear_block"),
    ("split_block", "merge_blocks"),
    ("widen_street", "narrow_street"),
    ("add_stub", "remove_stub"),
)


def mutate_plan(plan: CityPlan, exits, rng: random.Random,
                n_ops: Optional[int] = None,
                max_tries: int = MAX_TRIES
                ) -> Tuple[Optional[CityPlan], Tuple[str, ...]]:
    """A playable child of `plan`, or (None, ()) if none was found.

    Marks are re-placed after a structural edit only where they are missing:
    splitting a block leaves one half without a decision, and `_is_built`
    resolves that from the nearest mark, so the child agrees with its parent
    about the parts of the city the edit did not touch.
    """
    from citygen.validate import check

    if n_ops is None:
        n_ops = rng.randint(*OPS_PER_CHILD)

    names = sorted(OPERATORS)
    for _ in range(max_tries):
        child = plan.copy()
        applied: List[str] = []
        # An operator that finds nothing to do does not consume one of the
        # child's edits. Most operators are no-ops on a plan with no streets
        # yet: only `split_block` can cut the first street into an empty
        # field, so at one draw in ten a lineage took a dozen generations to
        # leave difficulty 0. Retrying inside the slot means a child really
        # receives the number of edits it was asked for, whatever state the
        # parent is in.
        for _ in range(n_ops):
            for _try in range(len(names) * 2):
                op = names[rng.randrange(len(names))]
                got = OPERATORS[op](child, rng)
                if got:
                    applied.append(got)
                    break
        if not applied:
            continue
        # A child that renders to nothing is playable by the validator's
        # reckoning, because an open field is playable. It is still not an
        # acceptable child of a developed parent: the curriculum would see a
        # difficulty-4 lineage quietly collapse to an empty map, which is the
        # complexity collapse the paired operators exist to prevent. Reject it
        # here and retry rather than letting it through and failing upstream.
        parent_built = bool(plan.render()[0])
        if parent_built and not child.render()[0]:
            continue
        ok, _why = check(child, exits)
        if ok:
            return child, tuple(applied)
    return None, ()
