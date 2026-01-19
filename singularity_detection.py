from isaacsim.examples.interactive.base_sample import BaseSample

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.experimental.prims import Articulation, RigidPrim


class SingularityDetect(BaseSample):
    def __init__(self):
        super().__init__()
        self._robot_path = "/World/robot"
        self.ur10 = None
        self.joint_positions = None
        self.ee_link = None
        self.jacobian = None

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        assets_root = get_assets_root_path()
        ur10_usd = assets_root + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

        stage_utils.add_reference_to_stage(
            usd_path=ur10_usd,
            path=self._robot_path,
            variants=[("Gripper", "Short_Suction")],
        )

    async def setup_post_load(self):
        self.ur10 = Articulation(self._robot_path)

        # Correctly bound suction end effector
        self.ee_link = RigidPrim(f"{self._robot_path}/ee_link")

        self.get_world().add_physics_callback(
            "joint_feedback", self._feedback_callback
        )

    def _feedback_callback(self, step_size):
        if self.ur10 is None:
            return

        self.joint_positions = self.ur10.get_dof_positions().numpy()
        #print("Joint positions:", self.joint_positions)
    
   
