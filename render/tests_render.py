"""Unit tests for the renderer module."""
import unittest
import numpy as np
from PIL import Image

from render.renderer import SceneRenderer, InstanceMetadata, LayoutError
from render.validate import (
    validate_relations, check_no_overlap, compute_iou, check_min_distance
)
from dsl.parser import SceneParser
from dsl.canonicalize import to_canonical


class TestRenderer(unittest.TestCase):
    """Test the SceneRenderer class."""

    def setUp(self):
        self.renderer = SceneRenderer(seed=42)
        self.parser = SceneParser()

    def test_render_simple_count(self):
        """Test rendering a simple COUNT_SENT scene."""
        scene_graph = {
            "canvas": {"W": 64, "H": 64, "bg": "white"},
            "objects": [
                {"id": "o1", "shape": "ball", "color": "red", "count": 1}
            ],
            "relations": [],
            "constraints": {"no_overlap": True, "min_dist_px": 4}
        }

        image, meta = self.renderer.render(scene_graph)

        # Check image properties
        self.assertEqual(image.size, (64, 64))
        self.assertEqual(image.mode, 'RGB')

        # Check metadata
        self.assertEqual(len(meta.instances), 1)
        self.assertTrue(meta.relations_valid)

        # Check instance properties
        inst = meta.instances[0]
        self.assertEqual(inst.object_id, "o1")
        self.assertEqual(inst.shape, "ball")
        self.assertEqual(inst.color, "red")
        self.assertEqual(inst.mask.shape, (64, 64))
        self.assertGreater(inst.mask.sum(), 0)  # Mask should have some pixels

    def test_render_multiple_instances(self):
        """Test rendering multiple instances of same object."""
        scene_graph = {
            "canvas": {"W": 64, "H": 64, "bg": "white"},
            "objects": [
                {"id": "o1", "shape": "cube", "color": "blue", "count": 3}
            ],
            "relations": [],
            "constraints": {"no_overlap": True, "min_dist_px": 4}
        }

        image, meta = self.renderer.render(scene_graph)

        # Should have 3 instances
        self.assertEqual(len(meta.instances), 3)

        # All should be blue cubes
        for inst in meta.instances:
            self.assertEqual(inst.shape, "cube")
            self.assertEqual(inst.color, "blue")

    def test_render_left_of_relation(self):
        """Test rendering with 'left of' relation."""
        scene_graph = {
            "canvas": {"W": 64, "H": 64, "bg": "white"},
            "objects": [
                {"id": "o1", "shape": "ball", "color": "red", "count": 1},
                {"id": "o2", "shape": "cube", "color": "blue", "count": 1}
            ],
            "relations": [
                {"type": "left_of", "src": "o1", "dst": "o2"}
            ],
            "constraints": {"no_overlap": True, "min_dist_px": 4}
        }

        image, meta = self.renderer.render(scene_graph)

        self.assertEqual(len(meta.instances), 2)
        self.assertTrue(meta.relations_valid)

        # Find instances
        red_ball = [i for i in meta.instances if i.color == 'red'][0]
        blue_cube = [i for i in meta.instances if i.color == 'blue'][0]

        # Red ball should be to the left of blue cube
        self.assertLess(red_ball.center[0], blue_cube.center[0])

    def test_render_on_relation(self):
        """Test rendering with 'on' relation."""
        scene_graph = {
            "canvas": {"W": 64, "H": 64, "bg": "white"},
            "objects": [
                {"id": "o1", "shape": "ball", "color": "red", "count": 1},
                {"id": "o2", "shape": "block", "color": "green", "count": 1}
            ],
            "relations": [
                {"type": "on", "src": "o1", "dst": "o2"}
            ],
            "constraints": {"no_overlap": True, "min_dist_px": 4}
        }

        image, meta = self.renderer.render(scene_graph)

        self.assertEqual(len(meta.instances), 2)
        self.assertTrue(meta.relations_valid)

        # Find instances
        red_ball = [i for i in meta.instances if i.color == 'red'][0]
        green_block = [i for i in meta.instances if i.color == 'green'][0]

        # Red ball should be above (lower y) green block
        self.assertLess(red_ball.center[1], green_block.center[1])

    def test_shape_drawing(self):
        """Test that different shapes are drawn correctly."""
        shapes = ['ball', 'cube', 'block']

        for shape in shapes:
            scene_graph = {
                "canvas": {"W": 64, "H": 64, "bg": "white"},
                "objects": [
                    {"id": "o1", "shape": shape, "color": "red", "count": 1}
                ],
                "relations": [],
                "constraints": {"no_overlap": True, "min_dist_px": 4}
            }

            image, meta = self.renderer.render(scene_graph)

            self.assertEqual(len(meta.instances), 1)
            inst = meta.instances[0]
            self.assertEqual(inst.shape, shape)

            # Check mask is reasonable size
            mask_area = inst.mask.sum()
            self.assertGreater(mask_area, 50)  # At least 50 pixels
            self.assertLess(mask_area, 1000)  # Not too large


class TestValidation(unittest.TestCase):
    """Test the validation functions."""

    def test_compute_iou_no_overlap(self):
        """Test IoU computation with no overlap."""
        mask1 = np.zeros((64, 64), dtype=np.uint8)
        mask1[10:20, 10:20] = 1

        mask2 = np.zeros((64, 64), dtype=np.uint8)
        mask2[30:40, 30:40] = 1

        iou = compute_iou(mask1, mask2)
        self.assertEqual(iou, 0.0)

    def test_compute_iou_full_overlap(self):
        """Test IoU computation with full overlap."""
        mask1 = np.zeros((64, 64), dtype=np.uint8)
        mask1[10:20, 10:20] = 1

        mask2 = np.zeros((64, 64), dtype=np.uint8)
        mask2[10:20, 10:20] = 1

        iou = compute_iou(mask1, mask2)
        self.assertAlmostEqual(iou, 1.0)

    def test_compute_iou_partial_overlap(self):
        """Test IoU computation with partial overlap."""
        mask1 = np.zeros((64, 64), dtype=np.uint8)
        mask1[10:20, 10:20] = 1

        mask2 = np.zeros((64, 64), dtype=np.uint8)
        mask2[15:25, 15:25] = 1

        iou = compute_iou(mask1, mask2)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_check_no_overlap(self):
        """Test no-overlap checking."""
        # Create non-overlapping instances
        inst1 = InstanceMetadata(
            object_id="o1", shape="ball", color="red",
            center=(20, 20), bbox=(10, 10, 30, 30),
            mask=np.zeros((64, 64), dtype=np.uint8)
        )
        inst1.mask[10:30, 10:30] = 1

        inst2 = InstanceMetadata(
            object_id="o2", shape="cube", color="blue",
            center=(45, 45), bbox=(35, 35, 55, 55),
            mask=np.zeros((64, 64), dtype=np.uint8)
        )
        inst2.mask[35:55, 35:55] = 1

        self.assertTrue(check_no_overlap([inst1, inst2]))

    def test_check_min_distance(self):
        """Test minimum distance checking."""
        inst1 = InstanceMetadata(
            object_id="o1", shape="ball", color="red",
            center=(20, 20), bbox=(10, 10, 30, 30),
            mask=np.zeros((64, 64), dtype=np.uint8)
        )

        inst2 = InstanceMetadata(
            object_id="o2", shape="cube", color="blue",
            center=(30, 20), bbox=(20, 10, 40, 30),
            mask=np.zeros((64, 64), dtype=np.uint8)
        )

        # Distance is 10, should pass min_dist_px=4
        self.assertTrue(check_min_distance([inst1, inst2], min_dist_px=4))

        # Should fail min_dist_px=15
        self.assertFalse(check_min_distance([inst1, inst2], min_dist_px=15))

    def test_validate_left_of(self):
        """Test left_of relation validation."""
        inst1 = InstanceMetadata(
            object_id="o1", shape="ball", color="red",
            center=(20, 30), bbox=(10, 20, 30, 40),
            mask=np.zeros((64, 64), dtype=np.uint8)
        )

        inst2 = InstanceMetadata(
            object_id="o2", shape="cube", color="blue",
            center=(45, 30), bbox=(35, 20, 55, 40),
            mask=np.zeros((64, 64), dtype=np.uint8)
        )

        relations = [{"type": "left_of", "src": "o1", "dst": "o2"}]

        # Should validate successfully
        validate_relations([inst1, inst2], relations)

        # Reverse should fail
        relations_rev = [{"type": "left_of", "src": "o2", "dst": "o1"}]
        with self.assertRaises(ValueError):
            validate_relations([inst1, inst2], relations_rev)

    def test_validate_on(self):
        """Test 'on' relation validation."""
        # Create stacked instances
        inst1 = InstanceMetadata(
            object_id="o1", shape="ball", color="red",
            center=(30, 20), bbox=(20, 10, 40, 30),
            mask=np.zeros((64, 64), dtype=np.uint8)
        )

        inst2 = InstanceMetadata(
            object_id="o2", shape="block", color="blue",
            center=(30, 45), bbox=(20, 35, 40, 55),
            mask=np.zeros((64, 64), dtype=np.uint8)
        )

        relations = [{"type": "on", "src": "o1", "dst": "o2"}]

        # Should validate (inst1 is above inst2)
        validate_relations([inst1, inst2], relations)


class TestIntegrationRender(unittest.TestCase):
    """Integration tests for parsing and rendering."""

    def setUp(self):
        self.parser = SceneParser()
        self.renderer = SceneRenderer(seed=42)

    def test_end_to_end_count(self):
        """Test end-to-end: text -> parse -> render."""
        text = "There are two red balls."
        scene_graph = self.parser.parse(text)

        image, meta = self.renderer.render(scene_graph)

        self.assertEqual(image.size, (64, 64))
        self.assertEqual(len(meta.instances), 2)

        # All should be red balls
        for inst in meta.instances:
            self.assertEqual(inst.color, "red")
            self.assertEqual(inst.shape, "ball")

    def test_end_to_end_relation(self):
        """Test end-to-end: text -> parse -> render with relation."""
        text = "The red ball is left of the blue cube."
        scene_graph = self.parser.parse(text)

        image, meta = self.renderer.render(scene_graph)

        self.assertEqual(image.size, (64, 64))
        self.assertEqual(len(meta.instances), 2)
        self.assertTrue(meta.relations_valid)

    def test_render_all_colors(self):
        """Test rendering all colors."""
        colors = ["red", "green", "blue", "yellow"]

        for color in colors:
            text = f"There is one {color} ball."
            scene_graph = self.parser.parse(text)

            image, meta = self.renderer.render(scene_graph)

            self.assertEqual(len(meta.instances), 1)
            self.assertEqual(meta.instances[0].color, color)

    def test_render_all_shapes(self):
        """Test rendering all shapes."""
        shapes = ["ball", "cube", "block"]

        for shape in shapes:
            text = f"There is one red {shape}."
            scene_graph = self.parser.parse(text)

            image, meta = self.renderer.render(scene_graph)

            self.assertEqual(len(meta.instances), 1)
            self.assertEqual(meta.instances[0].shape, shape)

    def test_no_overlap_constraint(self):
        """Test that rendered instances don't overlap significantly."""
        text = "There are five blue cubes."
        scene_graph = self.parser.parse(text)

        image, meta = self.renderer.render(scene_graph)

        self.assertEqual(len(meta.instances), 5)

        # Check no significant overlap (threshold 0.1 to match validation)
        self.assertTrue(check_no_overlap(meta.instances, threshold=0.1))


if __name__ == '__main__':
    unittest.main()
