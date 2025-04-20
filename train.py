import os
import sys
import os
import sys
import torch as th
import numpy as np
import yaml
import logging
import psutil
import gc
from tqdm import tqdm
try:
    import tensorboard
except ImportError:
    import subprocess
    print("TensorBoard not found, installing...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorboard>=2.14.0'], check=True)
    import tensorboard
from typing import Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from minecraft_env import RobotDogEnv
import argparse

class CurriculumCallback(BaseCallback):
    def __init__(self, vec_env, success_threshold=0.8, eval_freq=10000, verbose=0):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.success_threshold = success_threshold
        self.eval_freq = eval_freq
        self.success_history = []

    def _on_step(self) -> bool:
        # Record success events from info dict
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        for info, done in zip(infos, dones):
            if done and info.get('success') is not None:
                self.success_history.append(info['success'])
        # Increase difficulty at intervals
        if self.num_timesteps % self.eval_freq == 0:
            rate = np.mean(self.success_history[-10:]) if self.success_history else 0.0
            if rate > self.success_threshold:
                self.vec_env.env_method('increase_difficulty')
                difficulties = self.vec_env.get_attr('difficulty')
                print(f"Increased difficulty to {difficulties}")
        return True

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict[str, Any], features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.joint_net = nn.Sequential(
            nn.Linear(observation_space['joints'].shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.imu_net = nn.Sequential(
            nn.Linear(observation_space['imu'].shape[0], 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.target_net = nn.Sequential(
            nn.Linear(observation_space['target'].shape[0], 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.Linear(32 + 16 + 8, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        joint_feat = self.joint_net(obs['joints'])
        imu_feat = self.imu_net(obs['imu'])
        target_feat = self.target_net(obs['target'])
        concat = th.cat([joint_feat, imu_feat, target_feat], dim=1)
        return self.final(concat)

def make_env(urdf_path, render=False):
    def _init():
        env = RobotDogEnv(urdf_path, render=render)
        return env
    return _init

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('RLTraining')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def load_config(config_path: str, overrides: list) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    for override in overrides:
        key, value = override.split('=')
        try: value = eval(value)
        except: pass
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value
    return config

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training")
    def _on_step(self):
        if self.pbar:
            self.pbar.update(self.n_calls - self.pbar.n)
        return True
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()
            self.pbar = None

class MemoryTrackingCallback(BaseCallback):
    def __init__(self, warning_threshold_gb=4.0, critical_threshold_gb=6.0):
        super().__init__()
        self.warning = warning_threshold_gb * (1024**3)
        self.critical = critical_threshold_gb * (1024**3)
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss
            if mem > self.critical:
                self.logger.warn(f"CRITICAL memory usage: {mem/1e9:.2f}GB")
                gc.collect()
                if th.cuda.is_available():
                    th.cuda.empty_cache()
            elif mem > self.warning:
                self.logger.info(f"High memory usage: {mem/1e9:.2f}GB")
            self.logger.record("system/memory_usage", mem/1e9)
        return True

def main():
    parser = argparse.ArgumentParser(description='Train RobotDogEnv')
    parser.add_argument('--config', default='configs/default.yaml', help='配置文件路徑')
    parser.add_argument('--override', nargs='+', default=[], help='覆蓋配置，格式: key=value')
    parser.add_argument('--resume', type=str, help='恢復訓練的檢查點路徑')
    parser.add_argument('--render', dest='render', action='store_true', help='啟用 PyBullet GUI')
    parser.add_argument('--no-render', dest='render', action='store_false', help='關閉 PyBullet GUI')
    parser.set_defaults(render=None)
    args = parser.parse_args()
    config = load_config(args.config, args.override)
    if args.render is not None:
        config['env']['render'] = args.render
    # 否則就用 YAML 裡的 render
    logger = setup_logger('output/logs')
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/models', exist_ok=True)
    # 環境初始化
    print(f"準備建立環境，render={config['env']['render']} urdf={config['env']['urdf_path']}")
    try:
        if config['env']['render']:
            env = DummyVecEnv([make_env(config['env']['urdf_path'], render=True)])
            n_envs = 1
        else:
            env = SubprocVecEnv([make_env(config['env']['urdf_path'], render=False) for _ in range(config['env']['n_envs'])])
            n_envs = config['env']['n_envs']
        print("環境建立完成")
    except Exception as e:
        print(f"建立環境失敗: {e}")
        raise
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        training=True,
        gamma=config['training']['gamma'],
        epsilon=1e-8,
        norm_obs_keys=['joints', 'imu', 'target']
    )
    # 評估環境
    eval_env = DummyVecEnv([make_env(config['env']['urdf_path'], render=False)])  # 評估環境禁止 GUI
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        training=False,
        gamma=config['training']['gamma'],
        epsilon=1e-8,
        norm_obs_keys=['joints', 'imu', 'target']
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=config['callbacks']['checkpoint_freq'],
        save_path='output/models',
        name_prefix='robotdog'
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='output/models/best',
        log_path='output/logs/eval',
        eval_freq=config['callbacks']['eval_freq'],
        deterministic=True,
        render=config['env']['render']
    )
    # 進階課程學習回調（可根據需要替換為AdvancedCurriculumCallback）
    curriculum_callback = CurriculumCallback(env)
    memory_callback = MemoryTrackingCallback()
    tqdm_callback = TqdmCallback(total_timesteps=config['training']['timesteps'])

    # === TensorBoard Image Callback ===
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    class TensorboardImageCallback(BaseCallback):
        def __init__(self, eval_env, log_dir='output/logs', freq=10, verbose=0):
            super().__init__(verbose)
            self.eval_env = eval_env
            self.freq = freq
            self.writer = SummaryWriter(log_dir)
            self.episode = 0
        def _on_step(self) -> bool:
            infos = self.locals.get('infos', [])
            dones = self.locals.get('dones', [])
            for i, done in enumerate(dones):
                if done:
                    self.episode += 1
                    if self.episode % self.freq == 0:
                        # 用 eval_env 取得 render 圖片
                        try:
                            obs = self.eval_env.reset()
                            for _ in range(5):  # step 幾步讓畫面有變化
                                action = self.eval_env.action_space.sample()
                                obs, _, _, _ = self.eval_env.step(action)
                            img = self.eval_env.envs[0].render(mode='rgb_array')
                            if img is not None:
                                img = np.transpose(img, (2, 0, 1))
                                self.writer.add_image('robotdog_episode', img, self.episode)
                        except Exception as e:
                            pass
            return True
    image_callback = TensorboardImageCallback(eval_env)
    # === End TensorBoard Image Callback ===


    class VisualizationCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.n_envs = n_envs
            self.rewards = []
            self.episode_lengths = []
            self.episodes = 0
            self.current_steps = [0] * self.n_envs
            import matplotlib.pyplot as plt
            plt.ion()
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
            self.fig.suptitle('Training Progress')
            self.ax1.set_ylabel('Reward')
            self.ax2.set_ylabel('Episode Length')
            self.ax2.set_xlabel('Episode')
            self.fig_path = 'output/training_progress.png'
        def _on_step(self) -> bool:
            rewards = self.locals.get('rewards')
            dones = self.locals.get('dones')
            for i in range(self.n_envs):
                self.current_steps[i] += 1
                if dones[i]:
                    self.episodes += 1
                    self.rewards.append(rewards[i])
                    self.episode_lengths.append(self.current_steps[i])
                    self.current_steps[i] = 0
            if self.episodes and self.episodes % 10 == 0:
                self._update_plots()
            return True
        def _update_plots(self):
            import matplotlib.pyplot as plt
            self.ax1.clear()
            self.ax2.clear()
            self.ax1.plot(self.rewards, label='Reward')
            self.ax2.plot(self.episode_lengths, label='Episode Length')
            self.ax1.legend()
            self.ax2.legend()
            self.fig.tight_layout()
            self.fig.savefig(self.fig_path)
            plt.pause(0.1)
    vis_callback = VisualizationCallback()
    policy_kwargs = dict(
        features_extractor_class=CustomFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=config['features']['features_dim'],
        )
    )
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log='output/logs',
        learning_rate=config['training']['learning_rate'],
        n_steps=config['training']['n_steps'],
        batch_size=config['training']['batch_size'],
        gamma=config['training']['gamma'],
        ent_coef=config['training']['ent_coef'],
        clip_range=config['training']['clip_range']
    )
    # 恢復訓練
    if args.resume:
        model = PPO.load(args.resume)
        vec_normalize_path = os.path.join(os.path.dirname(args.resume), "vec_normalize.pkl")
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
        logger.info(f"從 {args.resume} 恢復訓練，當前步數: {model.num_timesteps}")
    callbacks = [checkpoint_callback, eval_callback, curriculum_callback, memory_callback, tqdm_callback]
    if config['callbacks']['use_vis']:
        callbacks.append(vis_callback)
    logger.info("開始訓練，配置參數:\n" + yaml.dump(config, allow_unicode=True))
    model.learn(
        total_timesteps=config['training']['timesteps'],
        callback=callbacks,
        log_interval=10,
        reset_num_timesteps=not bool(args.resume)
    )
    model.save("output/final_model")
    if isinstance(env, VecNormalize):
        env.save("output/vec_normalize.pkl")
    logger.info("訓練完成！")

if __name__ == '__main__':
    import subprocess
    import webbrowser
    import time
    # 啟動 TensorBoard 伺服器
    tb_proc = subprocess.Popen([
        sys.executable, '-m', 'tensorboard', '--logdir=output/logs', '--port=6006'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print('\n[INFO] TensorBoard 已自動啟動，請稍後片刻...')
    time.sleep(3)
    print('[INFO] 請在瀏覽器打開 http://localhost:6006 查看訓練曲線')
    try:
        webbrowser.open('http://localhost:6006')
    except Exception:
        pass
    main()
    # 訓練結束後關閉 TensorBoard
    tb_proc.terminate()