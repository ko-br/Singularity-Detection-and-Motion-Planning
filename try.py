from ur_ikfast import ur_kinematics
import json
import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb

robot = rtb.models.DH.UR10()
ur10e_arm = ur_kinematics.URKinematics('ur10e')
joint_limits = [
    (-np.pi, np.pi),  # Joint 1
    (-np.pi, np.pi),  # Joint 2
    (-np.pi, np.pi),  # Joint 3
    (-np.pi/2, -np.pi/2),  # Joint 4
    (-np.pi/2, np.pi/2),  # Joint 5
    (-np.pi, np.pi),  # Joint 6
]
theta_1 = np.linspace(joint_limits[0][0], joint_limits[0][1], 10)
theta_2 = np.linspace(joint_limits[1][0], joint_limits[1][1], 10)
theta_3 = np.linspace(joint_limits[2][0], joint_limits[2][1], 10)
theta_5 = [-np.pi/2, np.pi/2] 
theta_6 = np.linspace(joint_limits[5][0], joint_limits[5][1], 10)
results = []
for x in theta_1:
    for y in theta_2:
        for z in theta_3:
            for a in theta_5:
                for b in theta_6:
                    q = np.array([x, y, z, -np.pi/2, a, b])
                    T = ur10e_arm.forward(q, 'matrix')  # 4x4 FK matrix
                    pos = np.array(T[:3, 3])                      # extract XYZ
                    T_vec = np.array(T[:3, :].flatten())  
                    ik_sols = ur10e_arm.inverse(T_vec, all_solutions=True)
                    ik_sols_list = [sol.tolist() for sol in ik_sols]
                    results.append({
                        "original_q": q.tolist(),
                        "position": pos.tolist(),
                        "T_vec": T_vec.tolist(),
                        "ik_solutions": ik_sols_list,
                        "num_solutions": len(ik_sols)
                    })
Isaac_sim = []
for result in results:
    manip = []
    mean = 0
    for sol in result["ik_solutions"]:
        J=robot.jacobe(sol)
        det = np.linalg.det(np.dot(J, J.T))
        manipulability = np.sqrt(max(det, 0))
        manip.append(manipulability)
    if len(manip) > 0:
        mean = float(np.mean(manip))
        std  = float(np.std(manip))      # population std (ddof=0)
    else:
        mean = 0.0
        std  = 0.0
        Isaac_sim.append({
            "position": result["position"],
            "mean_manipulability": mean,
            "std_manipulability": std
        })
    result["mean_manipulability"] = mean
    result["std_manipulability"] = std

with open("ik_test_results.json", "w") as f:
    json.dump(results, f, indent=2)
with open("Isaac_Sim_File.json", "w") as f:
    json.dump(Isaac_sim, f, indent=2)
""""
def sample_joint():
    q = []
    for low, high in joint_limits:
        if low == high:
            q.append(low)       # fixed joint
        else:
            q.append(np.linspace(low, high, 10))
    return np.array(q)

reachable_points = []

for _ in range(5):  # number of samples
    q = sample_joint()
    T = ur10e_arm.forward(q, 'matrix')  # 4x4 FK matrix
    pos = T[:3, 3]                      # extract XYZ
    reachable_points.append((pos, q))  
    T_vec = T[:3, :].flatten()  
    ik_sols = ur10e_arm.inverse(T_vec, all_solutions=True)
    print(f"Original q: {q}, Position pos: {pos}, IK solutions found: {ik_sols}")

"""