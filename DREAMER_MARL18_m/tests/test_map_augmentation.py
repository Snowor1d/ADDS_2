import unittest

from map_augmentation import (
    TRANSFORMS,
    transform_map_geometry,
    transform_point,
    transformed_size,
    validate_transforms,
)


class MapAugmentationTest(unittest.TestCase):
    def test_all_transforms_keep_points_inside_transformed_map(self):
        width, height = 10, 20
        points = ((0, 0), (width, 0), (width, height), (0, height), (3, 7))

        for transform in TRANSFORMS:
            new_width, new_height = transformed_size(width, height, transform)
            for point in points:
                x, y = transform_point(point, width, height, transform)
                self.assertGreaterEqual(x, 0, transform)
                self.assertLessEqual(x, new_width, transform)
                self.assertGreaterEqual(y, 0, transform)
                self.assertLessEqual(y, new_height, transform)

    def test_quarter_rotation_swaps_rectangular_dimensions(self):
        self.assertEqual(transformed_size(10, 20, "rotate_90"), (20, 10))
        self.assertEqual(transform_point((2, 3), 10, 20, "rotate_90"), (3, 8))
        self.assertEqual(transformed_size(10, 20, "rotate_270"), (20, 10))
        self.assertEqual(transform_point((2, 3), 10, 20, "rotate_270"), (17, 2))

    def test_map_geometry_transforms_obstacles_and_exits_together(self):
        obstacles = [[(1, 2), (4, 2), (4, 6), (1, 6)]]
        exits = [[(0, 10), (2, 10), (2, 14), (0, 14)]]

        new_obstacles, new_exits, width, height = transform_map_geometry(
            obstacles, exits, 10, 20, "reflect_rotate_90"
        )

        self.assertEqual((width, height), (20, 10))
        self.assertEqual(new_obstacles[0][0], (2, 1))
        self.assertEqual(new_exits[0][0], (10, 0))

    def test_transform_configuration_must_be_valid_and_nonempty(self):
        self.assertEqual(validate_transforms(["rotate_90"]), ("rotate_90",))
        with self.assertRaises(ValueError):
            validate_transforms([])
        with self.assertRaises(ValueError):
            validate_transforms(["diagonal"])


if __name__ == "__main__":
    unittest.main()
