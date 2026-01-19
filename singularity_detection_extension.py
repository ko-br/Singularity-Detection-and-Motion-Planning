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

import omni.ui as ui
import numpy as np

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

        from omni.ui import Button, Label

        with self.get_extra_frames_handle():
            Label("UR10 Commands")

            Button(
                "Get Joint Angles",
                clicked_fn=self.on_get_joint_angles
            )

            Button(
                "Get Jacobian",
                clicked_fn=self.on_get_jacobian
            )

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
        self.build_ui()

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
        if getattr(self.sample, "ur10", None) is None:
            return

        self.sample.ur10.set_dof_position_targets(
            value_rad, dof_indices=[joint_index]
        )

    def update(self):
        if not hasattr(self, "_joint_labels"):
            return

        if getattr(self.sample, "joint_positions", None) is None:
            return

        for i, label in enumerate(self._joint_labels):
            label.text = f"{self.sample.joint_positions[i]:.3f} rad"

        # Button callbacks
    def on_get_joint_angles(self):
        if getattr(self.sample, "ur10", None) is None:
            print("Robot not ready")
            return
        q = self.sample.ur10.get_dof_positions().numpy()
        print("Joint angles:", q)

    def on_get_jacobian(self):
        if getattr(self.sample, "ur10", None) is None:
            print("Robot not ready")
            return
        jac = self.sample.ur10.get_jacobian_matrices().numpy()
        ee_idx = self.sample.ur10.get_link_indices("ee_link").list()[0]
        jac_ee = jac[:, ee_idx - 1, :, :6]
        print("Jacobian:\n", jac_ee)




