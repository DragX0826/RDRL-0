import os
import sys
import subprocess
from pathlib import Path
import argparse
import threading
import time
import webbrowser
import socket
import psutil

def setup_environment():
    """Setup virtual environment and install dependencies"""
    if not Path('.venv').exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', '.venv'], check=True)

    # Get the correct pip and python paths
    if os.name == 'nt':
        pip_path = '.venv/Scripts/pip'
        python_path = '.venv/Scripts/python'
    else:
        pip_path = '.venv/bin/pip'
        python_path = '.venv/bin/python'

    print("Installing dependencies...")
    subprocess.run([pip_path, 'install', '-r', 'requirements.txt'], check=True)
    return python_path

def verify_tensorboard(port=6006, timeout=30):
    """Check if TensorBoard is accepting connections before proceeding."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(('localhost', port))
                print(f"[验证] TensorBoard 在端口 {port} 就绪")
                return True
        except Exception:
            print(f"[验证] 等待 TensorBoard 启动 ({int(time.time()-start)}s)...")
            time.sleep(2)
    return False

def kill_port(port):
    """Terminate processes listening on the given port."""
    try:
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                proc = psutil.Process(conn.pid)
                proc.kill()
                print(f"[清理] 杀掉占用端口 {port} 的进程 PID {conn.pid}")
    except Exception as e:
        print(f"[清理] 无法清理端口 {port}: {e}")

# Enhanced start_tensorboard with explicit new window and readiness check
def start_tensorboard(logdir, port=6006):
    # 清理端口占用
    kill_port(port)
    """
    Launch TensorBoard in a separate window and verify it's ready before opening browser.
    """
    import subprocess
    tensorboard_cmd = f"{sys.executable} -m tensorboard.main --logdir {logdir} --port {port}"
    print(f"[自动启动] TensorBoard command: {tensorboard_cmd}")
    if os.name == 'nt':
        # Launch in new cmd window on Windows
        os.system(f'start cmd /k "{tensorboard_cmd}"')
    else:
        subprocess.Popen(tensorboard_cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    url = f"http://localhost:{port}"
    if verify_tensorboard(port):
        webbrowser.open(url)
        print(f"[自动启动] 已自动开启浏览器 {url}")
    else:
        print(f"[错误] TensorBoard 未就绪，请手动打开 {url}")
    # No process handle needed for main workflow
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gui', action='store_true', help='Disable PyBullet GUI')
    args = parser.parse_args()

    render = not args.no_gui
    python_path = sys.executable

    train_args = [
        '--config', 'configs/default.yaml',
        '--render' if render else '--no-render',
    ]

    # 啟動 TensorBoard
    tb_thread = threading.Thread(target=start_tensorboard, args=('output/logs', 6006), daemon=True)
    tb_thread.start()

    print("\n[自動啟動] 開始訓練...")
    try:
        subprocess.run([python_path, 'train.py'] + train_args, check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()