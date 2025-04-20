# 機器狗深度強化學習訓練系統

本專案實現了一套可於程序生成地形上訓練四足機器狗的深度強化學習（DRL）系統，結合地形生成、物理模擬（PyBullet）與強化學習演算法（Stable Baselines3），讓機器狗能在複雜 3D 環境中自主學習移動。

## 主要特色

- 程序化地形（分形雜訊）自動生成
- 多種生態地貌隨機切換
- PyBullet 物理引擎模擬機器狗運動
- 採用 Stable Baselines3 強化學習框架
- 支援多環境並行訓練
- TensorBoard 訓練過程即時監控

## 相依套件

- Python 3.8+
- numpy<2.0.0
- gymnasium>=0.29.1
- pybullet>=3.2.0
- torch==2.2.1+cpu
- opencv-python-headless>=4.8.0
- scipy>=1.11.0
- protobuf==3.20.3
- tensorboard>=2.14.0
- stable-baselines3

安裝方式：
```bash
pip install -r requirements.txt
```

## 專案結構

- `terrain_generator.py`：地形生成邏輯
- `robot_model.py`：機器狗模型與物理模擬
- `minecraft_env.py`：Gymnasium 環境實作
- `configs/default.yaml`：訓練與環境預設參數
- `run.py`：自動化訓練啟動、TensorBoard 啟動
- `start_training.py`：自動設置 URDF 路徑並啟動訓練
- `train.py`：主訓練腳本（PPO）
- `robot_dog.urdf`：機器狗模型 URDF 檔

## 使用方式

1. 建立並啟動虛擬環境：
```bash
python -m venv .venv
# Linux / macOS / WSL
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

2. 安裝相依套件：
```bash
pip install -r requirements.txt
```

3. 將你的機器狗 URDF 檔命名為 `robot_dog.urdf` 並放在專案根目錄

4. 啟動訓練（預設 GUI）：
```bash
python run.py
```

5. 不開啟 PyBullet GUI（推薦 Windows 用戶）：
```bash
python run.py --no-gui
```

也可用快速腳本：
```bash
python start_training.py
```

TensorBoard 會自動開啟於 http://localhost:6006

---

## 注意事項與常見問題（PyBullet GUI 顯示）

### Windows 下請使用 `--no-gui` 模式訓練

執行：
```powershell
python run.py --no-gui
```
這樣會用 PyBullet 的 DIRECT 模式，不會卡主執行緒，所有初始化流程都能執行到底。

這一切都發生在 `render: false`（也就是 PyBullet DIRECT/headless 模式）

#### 為什麼還是沒看到機器狗畫面？
你現在用的是 `--no-gui`（或 `render: false`），這時 PyBullet 不會顯示 3D 畫面，但訓練與物理模擬都在正確執行。
這是 Windows 下推薦的模式，因為 PyBullet GUI 會阻塞主執行緒，導致初始化流程無法進行。

### 如果你想看到機器狗「動起來」的畫面

- **在 Linux/WSL 下用 GUI 模式執行**
  - 到支援 X11 的 Linux 或 WSL2（安裝 X server）環境
  - 執行 `python run.py`（不要加 `--no-gui`）
  - 這樣 PyBullet GUI 就不會阻塞，可以看到真實 3D 畫面

- **Windows 下只能用 headless 訓練，或用「錄影/截圖」功能**
  - 可以在 `test_robot_dog.py` 或 `test_robot_dog_unit.py` 這類腳本裡，短暫開啟 GUI、存圖或錄影
  - 也可以在環境裡加一個「存圖」的 function，訓練過程中定時存下機器狗狀態

## 訓練參數說明

訓練腳本主要參數如下：
- 並行環境數量：4
- 總訓練步數：1,000,000
- 學習率：3e-4
- 批次大小：64
- Epoch 數：10
- 折扣因子 Gamma：0.99
- GAE Lambda：0.95

## 環境觀測與獎勵

觀測內容：
- 關節位置
- 關節速度
- IMU 姿態
- IMU 角速度
- IMU 線加速度
- 腳端接觸狀態

獎勵組成：
- 與目標距離
- 穩定性（IMU 姿態）
- 能源效率（關節扭力）
- 腳端接觸模式

## 授權

MIT License

## 致謝

本專案靈感來自於：
- 程序化內容生成
- 物理模擬技術
- 深度強化學習
- 機器人運動控制
