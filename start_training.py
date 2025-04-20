import os
import sys
from train import main

if __name__ == "__main__":
    # 確保 urdf_path 指向 robot_dog.urdf
    config_path = "configs/default.yaml"
    # 自動覆寫 urdf_path
    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(config_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.strip().startswith("urdf_path:"):
                f.write("urdf_path: robot_dog.urdf\n")
            else:
                f.write(line)
    # 啟動訓練（預設 render=True）
    sys.argv = [sys.argv[0], "--render"]
    main()
