from isaacgym import gymapi, gymtorch   # 如果 main.py 里也 import gymapi/gymtorch
import torch                            # 必须在 gym 之后
import time

from G1.envs.env           import G1Env
from G1.controllers.pd_cbf import PDControllerCBF
from G1.config             import SIM

def main():
    env = G1Env(headless=False)
    env.reset

    controller = PDControllerCBF(env)

    for step in range(10000):

        action = controller.get_action()

        env.step(action)

        env.render()

        time.sleep(SIM['dt'])
    
    env.close


if __name__ == '__main__':
    main()