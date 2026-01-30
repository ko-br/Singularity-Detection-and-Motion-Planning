import os
from ur_ikfast import ur_kinematics
import json
import numpy as np
import roboticstoolbox as rtb

robot = rtb.models.DH.UR10()
ur10e_arm = ur_kinematics.URKinematics('ur10e')
joint_limits = [
    (-np.pi, np.pi),  # Joint 1
    (-np.pi, np.pi),  # Joint 2
    (-np.pi, np.pi),  # Joint 3
    (-np.pi, np.pi),  # Joint 4
    (-np.pi/2, np.pi/2),  # Joint 5
    (-np.pi, np.pi),  # Joint 6
]

theta_1 = np.linspace(joint_limits[0][0], joint_limits[0][1], 30)
theta_2 = np.linspace(joint_limits[1][0], joint_limits[1][1], 30)
theta_3 = np.linspace(joint_limits[2][0], joint_limits[2][1], 30)
theta_4 = np.linspace(joint_limits[3][0], joint_limits[3][1], 30)
theta_5 = [-np.pi/2, np.pi/2]
theta_6 = np.linspace(joint_limits[5][0], joint_limits[5][1], 30)

"""
prev_theta_1 = np.linspace(joint_limits[0][0], joint_limits[0][1], 30)
prev_theta_2 = np.linspace(joint_limits[1][0], joint_limits[1][1], 30)
prev_theta_3 = np.linspace(joint_limits[2][0], joint_limits[2][1], 30)
prev_theta_4 = np.linspace(joint_limits[3][0], joint_limits[3][1], 30)
#prev_theta_5 = [-np.pi/2, np.pi/2]
prev_theta_6 = np.linspace(joint_limits[5][0], joint_limits[5][1], 30)
theta_1 = (prev_theta_1[:-1] + prev_theta_1[1:]) / 2  # 19 points between previous
theta_2 = (prev_theta_2[:-1] + prev_theta_2[1:]) / 2
theta_3 = (prev_theta_3[:-1] + prev_theta_3[1:]) / 2
theta_4 = (prev_theta_4[:-1] + prev_theta_4[1:]) / 2
theta_5 = [-np.pi/2, np.pi/2]  
theta_6 = (prev_theta_6[:-1] + prev_theta_6[1:]) / 2
"""
# Z-axis aligned with world Z (either up or down)
z_world = np.array([0, 0, 1])

results = []
for q1 in theta_1:
    for q2 in theta_2:
        for q3 in theta_3:
            for q4 in theta_4:
                for q5 in theta_5:
                    for q6 in theta_6:
                        q_full = np.array([q1, q2, q3, q4, q5, q6])
                        
                        # Compute forward kinematics
                        T_final = ur10e_arm.forward(q_full, 'matrix')
                        
                        # Extract Z-axis of end effector (3rd column of rotation matrix)
                        z_axis = T_final[:3, 2]
                        
                        # Check if Z-axis is aligned with world Z (parallel or anti-parallel)
                        dot_product = np.dot(z_axis, z_world)
                        
                        # Accept if pointing up OR down (abs(dot_product) close to 1)
                        if -1.1<=dot_product and dot_product<=-0.99:  # Threshold for alignment
                            pos = np.array(T_final[:3, 3])
                            T_vec = np.array(T_final[:3, :].flatten())
                            
                            ik_sols = ur10e_arm.inverse(T_vec, all_solutions=True)
                            ik_sols_list = [sol.tolist() for sol in ik_sols]
                            results.append({
                                "original_q": q_full.tolist(),
                                "position": pos.tolist(),
                                "T_vec": T_vec.tolist(),
                                "z_axis": z_axis.tolist(),
                                "dot_product": float(dot_product),
                                "ik_solutions": ik_sols_list,
                                "num_solutions": len(ik_sols)
                            })

print(f"Found {len(results)} configurations with vertically-aligned end effector")

Isaac_sim_full = []
for result in results:
    manip = []
    for sol in result["ik_solutions"]:
        J = robot.jacobe(sol)
        det = np.linalg.det(np.dot(J, J.T))
        manipulability = np.sqrt(max(det, 0))
        manip.append(manipulability)
    
    if len(manip) > 0:
        mean = float(np.mean(manip))
        std = float(np.std(manip))
    else:
        mean = 0.0
        std = 0.0
    
    result["mean_manipulability"] = mean
    result["std_manipulability"] = std
    Isaac_sim_full.append({
        "position": result["position"],
        "mean_manipulability": mean,
        #"orientation": result["orientation"],
    })

with open("ik_test_results_try3.json", "w") as f:
    json.dump(results, f, indent=2)
with open("Isaac_Sim_Full_File_Try3.json", "w") as f:
    json.dump(Isaac_sim_full, f, indent=2)
"""
# Read existing data if file exists
if os.path.exists("Isaac_Sim_Full_File_Try3.json"):
    with open("Isaac_Sim_Full_File_Try3.json", "r") as f:
        existing_results = json.load(f)
else:
    existing_results = []

# Merge new results with existing (assumes list structure)
combined_results = existing_results + Isaac_sim_full

# Write back
with open("Isaac_Sim_Full_File_Try3.json", "w") as f:
    json.dump(combined_results, f, indent=2)
"""