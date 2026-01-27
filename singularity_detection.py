from isaacsim.examples.interactive.base_sample import BaseSample

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.experimental.prims import Articulation, RigidPrim
import numpy as np
import time 
import json
import omni.isaac.core.utils.prims as prim_utils
from pxr import Gf, UsdGeom
from omni.isaac.core.objects import VisualSphere
#import roboticstoolbox as rtb

class SingularityDetect(BaseSample):
    def __init__(self):
        super().__init__()
        self._robot_path = "/World/robot"
        self.ur10e = None
        self.joint_positions = None
        self.ee_link = None
        self.jacobian = None
        self.saved_data = []
        self.ellipsoid = False
        self.velocity_ellipsoid = None


    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        assets_root = get_assets_root_path()
        ur10e_usd = assets_root + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
        
        stage_utils.add_reference_to_stage(
            usd_path=ur10e_usd,
            path=self._robot_path,
            #variants=[("Gripper", "Short_Suction")],
        )
        
        
        

    async def setup_post_load(self):
        self.ur10e = Articulation(self._robot_path)

        # Correctly bound suction end effector
        self.ee_link = RigidPrim(f"{self._robot_path}/wrist_3_link")
        self.get_world().add_physics_callback(
            "joint_feedback", self._feedback_callback
        )
        #self.ellipsoid = False

    def _feedback_callback(self, step_size):
        if self.ur10e is None:
            return

        self.joint_positions = self.ur10e.get_dof_positions().numpy()
        ee_pos, ee_quat = self.ee_link.get_world_poses()
        self.eepos = ee_pos
        self.eequat = ee_quat
        #print(f"EE Position: {ee_pos}, EE Orientation (quat): {ee_quat}")
        J = self.ur10e.get_jacobian_matrices().numpy()  # returns 6xN Jacobian for end-effector
        self.jacobian = J
        ee_idx = self.ur10e.get_link_indices("wrist_3_link").list()[0]
        J2 = J[:, ee_idx - 1, :, :6]
        
        if J2.ndim == 3:
            J2 = J2[0]  # remove batch
        det = np.linalg.det(np.dot(J2, J2.T))
        manipulability = np.sqrt(max(det, 0))  # prevent sqrt of negative
        #manipulability = np.sqrt(np.linalg.det(np.dot(J2, J2.T)))
        
        self.manipulability_index = manipulability
        #print(f"Manipulability index: {manipulability:.6f}")
        #if hasattr(self, "manip_label"):
         #   self.manip_label.text = f"Manipulability: {manipulability:.6f}"
        #print("Joint positions:", self.joint_positions)
    
    def get_manipulability_index(self):
        return self.manipulability_index


    def _on_save_data_event(self, log_path):
        if not hasattr(self, "eepos") or not hasattr(self, "eequat"):
            print("EE pose not available yet")
            return
        ee_pos = np.asarray(self.eepos).reshape(-1, 3)[0]
        ee_quat = np.asarray(self.eequat).reshape(-1, 4)[0]

        entry = {
            "timestamp": str(time.time()),

            "joint_positions": (
                [str(v) for v in self.joint_positions]
                if self.joint_positions is not None
                else []
            ),

            "ee_position": [
                str(ee_pos[0]),
                str(ee_pos[1]),
                str(ee_pos[2]),
            ],

            "ee_orientation": [
                str(ee_quat[0]),
                str(ee_quat[1]),
                str(ee_quat[2]),
                str(ee_quat[3]),
            ],
        }


        self.saved_data.append(entry)

        with open(log_path, "w") as f:
            json.dump(self.saved_data, f, indent=2)

        print(f"Saved EE pose. Total samples: {len(self.saved_data)}")

    def _manip_to_color(self, m):
        if m == 0:
            return Gf.Vec3f(1.0, 0.0, 0.0)   # red
        elif m < 0.2:
            return Gf.Vec3f(1.0, 1.0, 0.0)   # yellow
        else:
            return Gf.Vec3f(0.0, 1.0, 0.0)   # green

    def _on_generate_manip_map_event(self, input_directory):
        #file_path = self.task_ui_elements["Input Directory"].get_value_as_string()

        # Load data
        with open(input_directory, "r") as f:
            data = json.load(f)

        # Create a parent prim to keep things clean
        base_path = "/World/ManipulabilityMap"
        prim_utils.create_prim(base_path, "Xform")

        for i, entry in enumerate(data):
            position = entry["position"]
            mean_manip = entry["mean_manipulability"]

            color = self._manip_to_color(mean_manip)

            sphere_path = f"{base_path}/sphere_{i}"

            # Create sphere
            prim = prim_utils.create_prim(
                sphere_path,
                prim_type="Sphere",
                translation=position
            )

            # Set radius
            UsdGeom.Sphere(prim).GetRadiusAttr().Set(0.02)

            # Set color PROPERLY
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr([color])
    """
    def set_ellipsoid_visibility(self):
        self.ellipsoid = True

        if hasattr(self, "velocity_ellipsoid") and self.velocity_ellipsoid.is_valid():
            return
        if self.ur10e is None:
            return
        ee_pos, ee_quat = self.ee_link.get_world_poses()
        eepos = ee_pos
        eequat = ee_quat

        self.velocity_ellipsoid = VisualSphere(
            prim_path="/World/VelocityEllipsoid",
            radius=0.5,
            position=np.array(eepos),
            orientation=np.array(eequat),
            color=np.array([0.2, 0.6, 1.0]),
        )
    """

    def update_velocity_ellipsoid(self):
        if self.ur10e is None:
            return
        J_all = self.ur10e.get_jacobian_matrices().numpy()

        # ALWAYS use last link = end effector
        J = J_all[0, -1, :, :]      # (6, dof)

        # Translational Jacobian
        J_v = J[:3, :]              # (3, dof)

        JJt = J_v @ J_v.T           # (3, 3)

        eigvals, eigvecs = np.linalg.eigh(JJt)
        eigvals = np.maximum(eigvals, 1e-6)

        axes = np.sqrt(eigvals)

        idx = np.argsort(axes)[::-1]
        axes = axes[idx]
        eigvecs = eigvecs[:, idx]

        ee_pos, ee_quat = self.ee_link.get_world_poses()
        eepos = ee_pos
        eequat = ee_quat

        self.velocity_ellipsoid.set_world_pose(
            position=eepos,
            orientation=eequat
        )
        self.velocity_ellipsoid.set_local_scale(axes)


    def _rotation_matrix_to_quat(self, R):
        import omni.isaac.core.utils.rotations as rot_utils
        return rot_utils.matrix_to_quat(R)



        
   

