# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import omni.ext
from isaacsim.examples.browser import get_instance as get_browser_instance
from isaacsim.examples.interactive.base_sample import BaseSampleUITemplate
from isaacsim.examples.interactive.user_examples import SingularityDetect
from isaacsim.gui.components.ui_utils import btn_builder, state_btn_builder, str_builder
import omni.ui as ui
import numpy as np
import omni.kit.app
from omni.isaac.core.objects import VisualSphere



class DetectionExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self.example_name = "Awesome Example"
        self.category = "MyExamples"
        

        ui_kwargs = {
            "ext_id": ext_id,
            "file_path": os.path.abspath(__file__),
            "title": "My Awesome Example",
            "doc_link": "https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_hello_world.html",
            "overview": "This Example introduces the user on how to do cool stuff with Isaac Sim through scripting in asynchronous mode.",
            "sample": SingularityDetect(),
        }

        ui_handle = DetectionUI(**ui_kwargs)

        # register the example with examples browser
        get_browser_instance().register_example(
            name=self.example_name,
            execute_entrypoint=ui_handle.build_window,
            ui_hook=ui_handle.build_ui,
            category=self.category,
        )

        return

    def on_shutdown(self):
        get_browser_instance().deregister_example(name=self.example_name, category=self.category)

        return


class DetectionUI(BaseSampleUITemplate):

    def build_ui(self):
        super().build_ui()  # IMPORTANT: keeps default buttons
        self.task_ui_elements = {}
        #self.ellipsoid = False
        from omni.ui import Button

        with self.get_extra_frames_handle():
            with ui.CollapsableFrame(title="Commands", collapsed=False):
                with ui.VStack(spacing=6):
                    Button(
                        "Get Joint Angles",
                        clicked_fn=self.on_get_joint_angles
                    )

                    Button(
                        "Get Jacobian",
                        clicked_fn=self.on_get_jacobian
                    )

                    Button(
                        "Show Ellipsoid",
                        clicked_fn=self.on_show_ellipsoid
                    )
        with ui.CollapsableFrame(title="Robot Metrics", collapsed=False):
            with ui.VStack(spacing=5):
                ui.Label("Manipulability Index:", width=150)
                self.manipulability_label = ui.Label("0.0", width=100)
        self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_update)
        with ui.CollapsableFrame(
                title="Data Logging",
                width=ui.Fraction(0.33),
                height=0,
                visible=True,
                collapsed=False,
                # style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            ):

                self.build_data_logging_ui()
        with ui.CollapsableFrame(
            title = "Traverse Manipiulability Map",
            width=ui.Fraction(0.33),
            height=0,
            visible=True,
            collapsed=False,
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        ):
            self.build_manip_map_ui()
        

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    def post_load_button_event(self):
        self._build_joint_sliders()
        self.task_ui_elements["Save Data"].enabled = True
        self.task_ui_elements["Generate Manipulability Map"].enabled = True

    def _build_joint_sliders(self):
        with self.get_extra_frames_handle():
            with ui.CollapsableFrame(title="Joint Control & Feedback", collapsed=False):
                with ui.VStack(spacing=6):
                    self._joint_labels = []

                    for i, name in enumerate(self.JOINT_NAMES):
                        with ui.HStack():
                            ui.Label(name, width=160)

                            slider = ui.FloatSlider(min=-3.14, max=3.14)
                            slider.model.add_value_changed_fn(
                                lambda m, idx=i: self._on_joint_slider(idx, m.get_value_as_float())
                            )

                            label = ui.Label("0.000 rad", width=100)
                            self._joint_labels.append(label)

    def _on_joint_slider(self, joint_index, value_rad):
        if getattr(self.sample, "ur10e", None) is None:
            return

        self.sample.ur10e.set_dof_position_targets(
            value_rad, dof_indices=[joint_index]
        )

    def _on_update(self, event):
        print("Updating UI")
        if not hasattr(self, "manipulability_label"):
            return                                  

        if hasattr(self.sample, "manipulability_index"):
            self.manipulability_label.text = (
                f"Manipulability: {self.sample.manipulability_index:.6f}"
            )  
        #J = self.sample.ur10e.get_jacobian_matrices().numpy()
        if getattr(self.sample, "ellipsoid", True):
            self.sample.update_velocity_ellipsoid()



        # Button callbacks
    def on_get_joint_angles(self):
        if getattr(self.sample, "ur10e", None) is None:
            print("Robot not ready")
            return
        q = self.sample.ur10e.get_dof_positions().numpy()
        print("Joint angles:", q)

    def on_get_jacobian(self):
        if getattr(self.sample, "ur10e", None) is None:
            print("Robot not ready")
            return
        jac = self.sample.ur10e.get_jacobian_matrices().numpy()
        ee_idx = self.sample.ur10e.get_link_indices("wrist_3_link").list()[0]
        jac_ee = jac[:, ee_idx - 1, :, :6]
        print("Jacobian:\n", jac_ee)

    def _on_save_data_button_event(self):
        self.sample._on_save_data_event(self.task_ui_elements["Output Directory"].get_value_as_string())
        return

    def on_show_ellipsoid(self):
        self.sample.ellipsoid = True
        if self.sample.velocity_ellipsoid is None:
            ee_pos, ee_quat = self.sample.ee_link.get_world_poses()
            eepos = ee_pos
            eequat = ee_quat
            self.sample.velocity_ellipsoid = VisualSphere(
                prim_path="/World/VelocityEllipsoid",
                radius=0.5,
                position=np.array(eepos),
                orientation=np.array(eequat),
                color=np.array([0.2, 0.6, 1.0]),
            )
    def _on_generate_manip_map_button_event(self):
        self.sample._on_generate_manip_map_event(self.task_ui_elements["Input Directory"].get_value_as_string())
        return
    
    def build_data_logging_ui(self):
        with ui.VStack(spacing=5):
            dict = {
                "label": "Output Directory",
                "type": "stringfield",
                "default_val": os.path.join(os.getcwd(), "output_positions.json"),
                "tooltip": "Output Directory",
                "on_clicked_fn": None,
                "use_folder_picker": False,
                "read_only": False,
            }
            self.task_ui_elements["Output Directory"] = str_builder(**dict)

            dict = {
                "label": "Save Data",
                "type": "button",
                "text": "Save Data",
                "tooltip": "Save Data",
                "on_clicked_fn": self._on_save_data_button_event,
            }

            self.task_ui_elements["Save Data"] = btn_builder(**dict)
            self.task_ui_elements["Save Data"].enabled = False
        return
    
    def build_manip_map_ui(self):
        with ui.VStack(spacing=5):
            dict = {
                "label": "Input Directory",
                "type": "stringfield",
                "default_val": os.path.join(os.getcwd(), "Isaac_Sim_File.json"),
                "tooltip": "Input Directory",
                "on_clicked_fn": None,
                "use_folder_picker": False,
                "read_only": False,
            }

            self.task_ui_elements["Input Directory"] = str_builder(**dict)
            dict = {
                "label": "Generate Manipulability Map",
                "type": "button",
                "text": "Generate Manipulability Map",
                "tooltip": "Generate Manipulability Map",
                "on_clicked_fn": self._on_generate_manip_map_button_event,
            }

            self.task_ui_elements["Generate Manipulability Map"] = btn_builder(**dict)
            self.task_ui_elements["Generate Manipulability Map"].enabled = False
        return
    
    def on_shutdown(self):
        if hasattr(self, "_update_sub"):
            self._update_sub = None
        self.sample.ellipsoid = False
        self.sample.velocity_ellipsoid = None






