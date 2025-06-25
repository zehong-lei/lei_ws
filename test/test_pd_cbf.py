# test/test_pd_cbf.py

import pytest
import torch
from G1.controllers.pd_cbf import PDControllerCBF
from G1.envs.env           import G1Env

def test_pd_dimensions():
    env = G1Env()
    ctrl = PDControllerCBF(env)
    tau = ctrl.get_action()
    assert isinstance(tau, torch.Tensor)
    assert tau.shape[1] == env.num_dof
    