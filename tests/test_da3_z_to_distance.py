"""Unit tests for the DA3 axial-z -> ray-distance conversion fix.

Verifies that:
1. cos_theta_map() returns valid per-pixel cos(theta) in (0, 1].
2. Center pixels have cos(theta) ~ 1, edge pixels < 1.
3. depth_distance = depth_z / cos(theta) places the sample so that its
   camera-frame z-component equals the original axial depth (geometric
   round-trip).
4. cos_theta_map() uses the same focal/principal-point convention as
   gen_rays(): with identity pose, rays[..., 5] (world dz) == cos(theta).
"""
import unittest

import torch

from models.common.util import util
from models.bts.model.ray_sampler import PatchRaySampler, ImageRaySampler


def _make_inputs(n, v, h, w, focal=0.5):
    images = torch.rand(n, v, 3, h, w)
    poses = torch.eye(4).repeat(n, v, 1, 1)
    projs = torch.eye(4).repeat(n, v, 1, 1)
    projs[:, :, 0, 0] = focal
    projs[:, :, 1, 1] = focal
    projs[:, :, 0, 2] = 0.0
    projs[:, :, 1, 2] = 0.0
    da3_depths = torch.full((n, v, 1, h, w), 10.0)
    da3_confs = torch.full((n, v, 1, h, w), 1.0)
    return images, poses, projs, da3_depths, da3_confs


class TestCosThetaMap(unittest.TestCase):
    def setUp(self):
        self.w, self.h = 64, 48
        self.fx, self.fy = 0.5, 0.5  # wide FOV so edge angles are large
        self.cx, self.cy = 0.0, 0.0  # normalized coords: 0 == image center
        self.focal = torch.tensor([[self.fx, self.fy]])
        self.center = torch.tensor([[self.cx, self.cy]])

    def cos(self):
        return util.cos_theta_map(
            self.w, self.h, self.focal, c=self.center, device="cpu"
        )[0]  # (h, w)

    def test_shape(self):
        c = self.cos()
        self.assertEqual(tuple(c.shape), (self.h, self.w))

    def test_range(self):
        c = self.cos()
        self.assertTrue((c > 0).all(), "cos(theta) must be positive")
        self.assertTrue(
            (c <= 1.0 + 1e-6).all(), "cos(theta) must be <= 1"
        )

    def test_center_near_one(self):
        c = self.cos()
        # with even size, linspace(-1,1) has no exact 0; center pixel is slightly off
        self.assertAlmostEqual(float(c[self.h // 2, self.w // 2]), 1.0, places=2)
        # center must have larger cos(theta) than any edge/corner
        self.assertGreater(float(c[self.h // 2, self.w // 2]), float(c[0, 0]))

    def test_edge_less_than_one(self):
        c = self.cos()
        self.assertLess(float(c[0, 0]), 0.999)
        self.assertLess(float(c[self.h // 2, 0]), 0.999)

    def test_z_to_distance_roundtrip(self):
        """depth_distance = z / cos(theta) places the point so its camera-frame
        z-component equals the original axial depth z."""
        c = self.cos()
        unproj_norm = util.unproj_map(
            self.w, self.h, self.focal, c=self.center, device="cpu", norm_dir=True
        )[0]  # (h, w, 3) unit camera ray directions

        z_axial = torch.tensor(20.0)
        for (y, x) in [
            (self.h // 2, 0),      # left edge
            (0, 0),                # corner
            (self.h // 2, self.w - 1),  # right edge
            (self.h - 1, self.w // 2),  # bottom
        ]:
            cos_theta = c[y, x]
            distance = z_axial / cos_theta
            point = unproj_norm[y, x] * distance
            self.assertAlmostEqual(
                float(point[2]), float(z_axial), places=4,
                msg=f"pixel ({y},{x}) z={point[2]:.4f} != {z_axial}",
            )
            # distance along ray must be >= axial depth
            self.assertGreaterEqual(float(distance), float(z_axial) - 1e-4)

    def test_consistency_with_gen_rays(self):
        """With identity pose, gen_rays' world dz must equal cos_theta_map."""
        poses = torch.eye(4).unsqueeze(0)  # (1, 4, 4)
        rays = util.gen_rays(
            poses, self.w, self.h, 1.0, 80.0,
            focal=self.focal, c=self.center, norm_dir=True,
        )  # (1, h, w, 8)
        c = self.cos()
        self.assertTrue(
            torch.allclose(rays[0, :, :, 5], c, atol=1e-5),
            "cos_theta_map must match gen_rays world dz under identity pose",
        )


class TestSamplerIntegration(unittest.TestCase):
    """Smoke tests: the fix is wired into the samplers and only fires when
    DA3 depths are present; axial depth is converted (>= input, since
    cos(theta) <= 1), and the no-DA3 path is unchanged."""

    def test_patch_sampler_converts_da3(self):
        n, v, h, w = 1, 2, 48, 64
        images, poses, projs, da3_depths, da3_confs = _make_inputs(n, v, h, w)
        sampler = PatchRaySampler(ray_batch_size=256, z_near=1.0, z_far=80.0, patch_size=8)
        _, _, depths, confs = sampler.sample(
            images, poses, projs, da3_depths=da3_depths, da3_confs=da3_confs
        )
        self.assertEqual(tuple(depths.shape), (n, 256))
        self.assertTrue((depths >= 10.0 - 1e-5).all())
        self.assertTrue((depths <= 30.0 + 1e-4).all())  # max ~3x at corners
        self.assertTrue((depths > 10.0 + 1e-4).any())  # off-center rays grow
        self.assertEqual(tuple(confs.shape), (n, 256))

    def test_patch_sampler_no_da3_unchanged(self):
        n, v, h, w = 1, 2, 48, 64
        images, poses, projs, _, _ = _make_inputs(n, v, h, w)
        sampler = PatchRaySampler(ray_batch_size=256, z_near=1.0, z_far=80.0, patch_size=8)
        _, _, depths, confs = sampler.sample(images, poses, projs)
        self.assertIsNone(depths)
        self.assertIsNone(confs)

    def test_image_sampler_converts_da3(self):
        n, v, h, w = 1, 2, 48, 64
        images, poses, projs, da3_depths, da3_confs = _make_inputs(n, v, h, w)
        sampler = ImageRaySampler(z_near=1.0, z_far=80.0, height=h, width=w)
        _, _, depths, confs = sampler.sample(
            images, poses, projs, da3_depths=da3_depths, da3_confs=da3_confs
        )
        self.assertEqual(tuple(depths.shape), (n, v * h * w))
        self.assertTrue((depths >= 10.0 - 1e-5).all())
        self.assertTrue((depths <= 30.0 + 1e-4).all())
        self.assertTrue((depths > 10.0 + 1e-4).any())
        self.assertEqual(tuple(confs.shape), (n, v * h * w))

    def test_image_sampler_no_da3_unchanged(self):
        n, v, h, w = 1, 2, 48, 64
        images, poses, projs, _, _ = _make_inputs(n, v, h, w)
        sampler = ImageRaySampler(z_near=1.0, z_far=80.0, height=h, width=w)
        _, _, depths, confs = sampler.sample(images, poses, projs)
        self.assertIsNone(depths)
        self.assertIsNone(confs)

    def test_patch_sampler_disable_conversion(self):
        """da3_z_to_distance=False must reproduce the original (buggy) impl:
        raw axial z is passed through unchanged, no division by cos(theta)."""
        n, v, h, w = 1, 2, 48, 64
        images, poses, projs, da3_depths, da3_confs = _make_inputs(n, v, h, w)
        sampler = PatchRaySampler(ray_batch_size=256, z_near=1.0, z_far=80.0, patch_size=8)
        sampler.da3_z_to_distance = False
        _, _, depths, confs = sampler.sample(
            images, poses, projs, da3_depths=da3_depths, da3_confs=da3_confs
        )
        self.assertEqual(tuple(depths.shape), (n, 256))
        self.assertTrue(torch.allclose(depths, torch.full_like(depths, 10.0), atol=1e-5))

    def test_image_sampler_disable_conversion(self):
        """Same as above, for the ImageRaySampler (validation path)."""
        n, v, h, w = 1, 2, 48, 64
        images, poses, projs, da3_depths, da3_confs = _make_inputs(n, v, h, w)
        sampler = ImageRaySampler(z_near=1.0, z_far=80.0, height=h, width=w)
        sampler.da3_z_to_distance = False
        _, _, depths, confs = sampler.sample(
            images, poses, projs, da3_depths=da3_depths, da3_confs=da3_confs
        )
        self.assertEqual(tuple(depths.shape), (n, v * h * w))
        self.assertTrue(torch.allclose(depths, torch.full_like(depths, 10.0), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
