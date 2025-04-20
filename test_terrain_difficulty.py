import time
print("[DEBUG] Script started")
try:
    from minecraft_env import RobotDogEnv
    print("[DEBUG] Imported RobotDogEnv")
except Exception as e:
    print(f"[ERROR] Failed to import RobotDogEnv: {e}")
    raise

if __name__ == "__main__":
    urdf_path = "robot_dog.urdf"  # Adjust if needed
    num_episodes = 5
    render = False  # 預設用 DIRECT mode，避免 GUI 卡住
    print("[INFO] 預設用 DIRECT mode（無 GUI），如需 PyBullet GUI 請將 render = True，並在 Linux/WSL 下執行。")

    try:
        env = RobotDogEnv(urdf_path=urdf_path, render=render)
        print("[DEBUG] RobotDogEnv instance created (DIRECT mode)")
    except Exception as e:
        print(f"[ERROR] Failed to initialize RobotDogEnv: {e}")
        raise

    for ep in range(num_episodes):
        try:
            obs, info = env.reset()
            print(f"Episode {ep+1} - Difficulty: {env.difficulty:.2f}, Target: {env.target_position}")
        except Exception as e:
            print(f"[ERROR] Reset failed: {e}")
            break
        done = False
        step = 0
        while not done and step < 200:
            try:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1
                if render:
                    time.sleep(1/60)  # Slow down for visualization
            except Exception as e:
                print(f"[ERROR] Step failed: {e}")
                done = True
        print(f"Episode {ep+1} finished after {step} steps.\n")

    print("Test complete. Close the GUI window to exit.")
    if render:
        input("Press Enter to close...")
        # Properly disconnect PyBullet
        import pybullet as p
        p.disconnect(env.physics_client_id)
