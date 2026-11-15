"""Independent arithmetic/Sobel and autograd checks of the group objective."""
import unittest

import numpy as np
import torch

from src.methods.sage_finetune.losses import FusionLoss


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_intensity_and_anchor_hand_computed(self):
        f=torch.full((1,1,3,4),.25,requires_grad=True)
        v=torch.zeros_like(f); i=torch.ones_like(f); ref=torch.full_like(f,.5)
        parts=FusionLoss()(f,v,i,ref)
        self.assertAlmostEqual(parts["intensity"].item(),(.25**2+.75**2)/2)
        self.assertAlmostEqual(parts["anchor"].item(),.25**2)
        expected=parts["intensity"]+.2*parts["gradient"]+parts["anchor"]
        self.assertTrue(torch.equal(parts["total"],expected))

    def test_sobel_against_independent_numpy_loops(self):
        value=np.arange(20,dtype=np.float32).reshape(4,5)/20
        padded=np.pad(value,1)
        kx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        ky=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        result=np.zeros_like(value)
        for r in range(4):
            for c in range(5):
                patch=padded[r:r+3,c:c+3]
                result[r,c]=abs((patch*kx).sum())+abs((patch*ky).sum())
        actual=FusionLoss().sobel(torch.from_numpy(value)[None,None])[0,0].numpy()
        np.testing.assert_allclose(actual,result,atol=3e-7)

    def test_source_max_gradient_target(self):
        f=torch.zeros(1,1,4,5)
        v=torch.arange(20,dtype=torch.float32).reshape(1,1,4,5)/20
        i=v.flip(-1)
        loss=FusionLoss()
        result=loss(f,v,i,f)["gradient"]
        target=torch.maximum(loss.sobel(v),loss.sobel(i)).mean()
        self.assertAlmostEqual(result.item(),target.item())

    def test_each_component_has_finite_nonzero_output_gradient(self):
        torch.manual_seed(4)
        for component,weights in [
            ("intensity",dict(intensity=1,gradient=0,anchor=0)),
            ("gradient",dict(intensity=0,gradient=1,anchor=0)),
            ("anchor",dict(intensity=0,gradient=0,anchor=1)),
        ]:
            f=torch.rand(1,1,4,5,requires_grad=True)
            ref=torch.rand_like(f,requires_grad=True)
            p=FusionLoss(**weights)(f,torch.rand_like(f),torch.rand_like(f),ref)
            p["total"].backward()
            self.assertTrue(torch.isfinite(f.grad).all(),component)
            self.assertGreater(f.grad.abs().sum().item(),0,component)
            self.assertIsNone(ref.grad)

    def test_constant_interior_border_and_zero_case(self):
        loss=FusionLoss()
        constant=torch.ones(1,1,5,5)
        g=loss.sobel(constant)
        self.assertEqual(g[:,:,1:-1,1:-1].sum().item(),0)
        self.assertGreater(g.sum().item(),0)
        z=torch.zeros_like(constant,requires_grad=True)
        self.assertEqual(loss(z,z,z,z)["total"].item(),0)

    def test_invalid_weights_nan_and_shape_fail(self):
        for weights in [dict(intensity=-1),dict(anchor=float("nan")),
                        dict(intensity=0,gradient=0,anchor=0)]:
            with self.assertRaises(ValueError):
                FusionLoss(**weights)
        z=torch.zeros(1,1,4,5)
        with self.assertRaises(FloatingPointError):
            FusionLoss()(z+float("nan"),z,z,z)
        with self.assertRaises(ValueError):
            FusionLoss()(z,z[:,:,:3],z,z)


if __name__ == "__main__":
    unittest.main()
