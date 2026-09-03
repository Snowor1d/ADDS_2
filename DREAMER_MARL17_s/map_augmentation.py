"""Geometric data augmentation for polygon-based simulation maps."""

from typing import Iterable, List, Sequence, Tuple


Point = Tuple[float, float]
Polygon = List[Point]

TRANSFORMS = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "reflect",
    "reflect_rotate_90",
    "reflect_rotate_180",
    "reflect_rotate_270",
)


def transformed_size(width: float, height: float, transform: str) -> Tuple[float, float]:
    """Return the map dimensions after ``transform`` is applied."""
    _validate_transform(transform)
    rotation = _rotation_quarters(transform)
    return (height, width) if rotation % 2 else (width, height)


def transform_point(
    point: Sequence[float],
    width: float,
    height: float,
    transform: str,
) -> Point:
    """Transform a point in the closed map domain [0, width] x [0, height]."""
    _validate_transform(transform)
    x, y = point

    # Reflection is left-right about the vertical centre line.  Combining it
    # with the four rotations yields every symmetry in the dihedral group D4.
    if transform.startswith("reflect"):
        x = width - x

    rotation = _rotation_quarters(transform)
    if rotation == 0:
        return (x, y)
    if rotation == 1:
        return (y, width - x)
    if rotation == 2:
        return (width - x, height - y)
    return (height - y, x)


def transform_polygons(
    polygons: Iterable[Iterable[Sequence[float]]],
    width: float,
    height: float,
    transform: str,
) -> List[Polygon]:
    """Apply one symmetry to every vertex of every polygon."""
    return [
        [transform_point(point, width, height, transform) for point in polygon]
        for polygon in polygons
    ]


def transform_map_geometry(
    obstacles: Iterable[Iterable[Sequence[float]]],
    exits: Iterable[Iterable[Sequence[float]]],
    width: float,
    height: float,
    transform: str,
) -> Tuple[List[Polygon], List[Polygon], float, float]:
    """Transform all static map geometry and return its new dimensions."""
    new_obstacles = transform_polygons(obstacles, width, height, transform)
    new_exits = transform_polygons(exits, width, height, transform)
    new_width, new_height = transformed_size(width, height, transform)
    return new_obstacles, new_exits, new_width, new_height


def validate_transforms(transforms: Iterable[str]) -> Tuple[str, ...]:
    """Validate a configured transform collection and return it as a tuple."""
    result = tuple(transforms)
    if not result:
        raise ValueError("MAP_AUGMENTATION_TRANSFORMS must not be empty")
    for transform in result:
        _validate_transform(transform)
    return result


def _rotation_quarters(transform: str) -> int:
    if transform.endswith("rotate_90"):
        return 1
    if transform.endswith("rotate_180"):
        return 2
    if transform.endswith("rotate_270"):
        return 3
    return 0


def _validate_transform(transform: str) -> None:
    if transform not in TRANSFORMS:
        raise ValueError(
            f"Unknown map augmentation transform {transform!r}; "
            f"expected one of {TRANSFORMS}"
        )
