import pybullet as p
import pybullet_data
import os
from PIL import Image
import numpy as np

def run_and_capture(urdf_path, img_path):
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.5], useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
    num_joints = p.getNumJoints(robot_id)
    assert num_joints >= 8, f"Too few joints: {num_joints}"
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[2, 2, 1],
        cameraTargetPosition=[0, 0, 0.3],
        cameraUpVector=[0, 0, 1]
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=3.1
    )
    width, height, rgb, _, _ = p.getCameraImage(
        width=320, height=320, viewMatrix=view_matrix, projectionMatrix=proj_matrix
    )
    rgb_array = np.reshape(rgb, (height, width, 4))
    img = Image.fromarray(rgb_array[:, :, :3], 'RGB')
    img.save(img_path)
    p.disconnect()

def test_robot_visible():
    urdf_path = "robot_dog.urdf"
    img_path = "robot_dog_test.png"
    run_and_capture(urdf_path, img_path)
    img = Image.open(img_path)
    extrema = img.getextrema()
    for channel in extrema:
        assert channel[1] - channel[0] > 10, "Image channel too flat, robot may be invisible."
    print("[PASS] Robot visible and image captured.")
    os.remove(img_path)

if __name__ == "__main__":
    test_robot_visible()