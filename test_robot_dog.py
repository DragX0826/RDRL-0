import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

robot_id = p.loadURDF(
    "robot_dog.urdf",
    basePosition=[0, 0, 0.5],
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION
)

num_joints = p.getNumJoints(robot_id)
print("Robot joints:", num_joints)
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    print(f"Joint {i}: {info[1].decode()} type={info[2]} parent={info[16]} child={info[12].decode()}")

while True:
    time.sleep(1)
