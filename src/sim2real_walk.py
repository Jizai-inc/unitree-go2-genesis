#!/usr/bin/env python
import time
import sys
import os
import pickle
import torch
import numpy as np

# 学習済みポリシーのロード用モジュール
from rsl_rl.runners import OnPolicyRunner

# unitree_sdk2_python の各種モジュール
import sys
sys.path.append('../unitree_sdk2_python')
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
sys.path.remove('../unitree_sdk2_python')

sys.path.append('../unitree_sdk2_python/example/go2/low_level')
import unitree_legged_const as go2
sys.path.remove('../unitree_sdk2_python/example/go2/low_level')

import genesis as gs

# 設定ファイルのロード（以前のコードと同様に cfgs.pkl を利用）
def load_configs(exp_name):
    cfg_path = os.path.join("../logs", exp_name, "cfgs.pkl")
    with open(cfg_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg

class RealRobotDeployer:
    """
    実機の低レベルセンサデータ（low_state）から obs を構築し，
    学習済みポリシーの推論結果から target_dof_pos を算出、実機へコマンド送信するクラス．
    """
    def __init__(self, policy, env_cfg, obs_cfg):
        self.policy = policy
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        
        # 初期状態用変数
        self.obs = None
        # 前回のアクション（12次元，初期はゼロ）
        self.last_actions = torch.zeros((env_cfg["num_actions"],), dtype=torch.float32)
        
        # スケール（obs_cfg["obs_scales"]は genesis_go2_env.py に準じる）
        self.ang_vel_scale = obs_cfg["obs_scales"]["ang_vel"]
        self.dof_pos_scale = obs_cfg["obs_scales"]["dof_pos"]
        self.dof_vel_scale = obs_cfg["obs_scales"]["dof_vel"]
        # コマンド情報（ここでは実機用に外部コマンドが無い場合ゼロベクトル）
        self.commands = torch.zeros(3, dtype=torch.float32)
        # 実際には genesis_go2_env.py の commands_scale に準じた値を設定（例として各値1.0）
        self.commands_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        
        # IMU情報：実機からは low_state.imu_state で受信（ここでは角速度のみ利用）
        self.base_ang_vel = torch.zeros(3, dtype=torch.float32)
        # 重力情報：実機では直接得られないため，仮に [0,0,-1] とする
        self.projected_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        
        # 関節角度のデフォルト値（env_cfg["default_joint_angles"]）
        self.default_joint_angles = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["dof_names"]],
            dtype=torch.float32
        )
        self.num_actions = env_cfg["num_actions"]
        self.action_scale = env_cfg["action_scale"]
        
        # 低レベルコマンド構造体の初期化
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.init_low_cmd()
        self.crc = CRC()
        
        # Publisher の初期化
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        
        # low_state の初期化と Subscriber の設定
        self.low_state = None
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self.LowStateHandler, 10)
    
    def init_low_cmd(self):
        # go2_stand_example.py の初期化処理に準じる
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # 位置制御モード
            self.low_cmd.motor_cmd[i].q = go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def LowStateHandler(self, msg):
        """
        go2_stand_example.py の LowStateMessageHandler に準じて，low_state を更新．
        また，IMU 情報（例：gyro）を更新する。
        """
        self.low_state = msg
        if hasattr(msg, "imu_state"):
            # ここでは imu_state.gyroscope に角速度（[wx, wy, wz]）が入っていると仮定
            self.base_ang_vel = torch.tensor(msg.imu_state.gyroscope, dtype=torch.float32)
            # projected_gravity は仮設定（必要なら実際の姿勢から算出する）
            self.projected_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    
    def construct_obs(self):
        """
        実機から取得した low_state から，genesis の obs と同様の45次元ベクトルを構築する．
        以下の順序で連結：
          - base_ang_vel (3) * ang_vel_scale
          - projected_gravity (3)
          - commands (3) * commands_scale
          - (dof_pos - default_joint_angles) (12) * dof_pos_scale
          - dof_vel (12) * dof_vel_scale
          - last_actions (12)
        """
        if self.low_state is None:
            return None  # センサデータ待ち
        
        # base_ang_vel
        base_ang_vel = self.base_ang_vel * self.ang_vel_scale
        
        # projected_gravity（そのまま使用）
        projected_gravity = self.projected_gravity
        
        # commands（ここではゼロベクトル）
        commands = self.commands * self.commands_scale
        
        # dof_pos と dof_vel：先頭12モータから取得
        dof_pos = []
        dof_vel = []
        for i in range(12):
            dof_pos.append(self.low_state.motor_state[i].q)
            dof_vel.append(self.low_state.motor_state[i].dq)
        dof_pos = torch.tensor(dof_pos, dtype=torch.float32)
        dof_vel = torch.tensor(dof_vel, dtype=torch.float32)
        
        dof_offset = (dof_pos - self.default_joint_angles) * self.dof_pos_scale
        
        # last_actions（前回のポリシー出力）
        last_actions = self.last_actions
        
        # 45次元ベクトルに連結
        obs = torch.cat([
            base_ang_vel,           # 3
            projected_gravity,      # 3
            commands,               # 3
            dof_offset,             # 12
            dof_vel * self.dof_vel_scale,  # 12
            last_actions            # 12
        ])
        self.obs = obs
        return obs
    
    def control_loop(self):
        """
        RecurrentThread により周期的に呼ばれる制御ループ．
        現在のセンサ値から obs を構築し，ポリシー推論，target_dof_pos の算出，コマンド送信を行う．
        """
        obs = self.construct_obs()
        if obs is None:
            return  # センサ待ち
        
        # バッチ次元を付加してポリシーへ入力
        obs_tensor = obs.unsqueeze(0)
        with torch.no_grad():
            action = self.policy(obs_tensor).squeeze(0)
        
        # 今回の出力を記録（次回の obs の last_actions として利用）
        self.last_actions = action
        
        # target_dof_pos の計算
        target_dof_pos = action * self.action_scale + self.default_joint_angles
        
        # 先頭12モータ（脚関節）に対して目標角度を設定
        for i in range(12):
            self.low_cmd.motor_cmd[i].q = float(target_dof_pos[i])
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = self.env_cfg["kp"]
            self.low_cmd.motor_cmd[i].kd = self.env_cfg["kd"]
            self.low_cmd.motor_cmd[i].tau = 0
        
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)
    
    def start(self, control_dt):
        self.thread = RecurrentThread(interval=control_dt, target=self.control_loop, name="real_robot_control")
        self.thread.Start()
    
    def stop(self):
        if hasattr(self, "thread"):
            self.thread.Stop()

def main():
    # DDS 通信初期化
    # if len(sys.argv) > 1:
    #     ChannelFactoryInitialize(0, sys.argv[1])
    # else:
    #     ChannelFactoryInitialize(0)
    
    exp_name = "go2-walking"
    ckpt = 1000
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_configs(exp_name)
    log_dir = os.path.join("../logs", exp_name)
    resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    
    gs.init()
    
    # 学習済みポリシーのロード
    # ここではダミー環境（Go2Env）を用いて OnPolicyRunner 経由でロード
    from genesis_go2_env import Go2Env
    env = Go2Env(
        num_envs=1, 
        env_cfg=env_cfg, 
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, 
        command_cfg=command_cfg, 
        show_viewer=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cpu")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cpu")
    
    # RealRobotDeployer の初期化
    deployer = RealRobotDeployer(policy, env_cfg, obs_cfg)
    
    print("Waiting for low_state from real robot...")
    while deployer.low_state is None:
        time.sleep(0.1)

    print("low_state received. Starting control loop.")
    print("FR_0 motor state: ", deployer.low_state.motor_state[go2.LegID["FR_0"]])
    print("IMU state: ", deployer.low_state.imu_state)
    print("Battery state: voltage: ", deployer.low_state.power_v, "current: ", deployer.low_state.power_a)
    print("Start !!!")

    # deployer.start(control_dt=0.02)  # 50Hz 制御ループ
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping control loop.")
        deployer.stop()

if __name__ == "__main__":
    main()
