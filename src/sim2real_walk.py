#!/usr/bin/env python
import csv
import threading
import time
import sys
import os
import pickle
import torch
import numpy as np
import math
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

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
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

sys.path.remove('../unitree_sdk2_python')

sys.path.append('../unitree_sdk2_python/example/go2/low_level')
import unitree_legged_const as go2
sys.path.remove('../unitree_sdk2_python/example/go2/low_level')

import genesis as gs

# 関節名称（CSVのヘッダー用）
dof_names = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

# --- ヘルパー関数：クオータニオン計算 ---
def quat_conjugate(q):
    # q: [w, x, y, z]
    return [q[0], -q[1], -q[2], -q[3]]

def quat_multiply(q1, q2):
    # q1, q2: [w, x, y, z]
    w = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    x = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    y = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    z = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    return [w, x, y, z]

def rotate_vector_by_quat(v, q):
    # v: 3次元ベクトル, q: [w, x, y, z]（正規化済みと仮定）
    q_v = [0.0] + v
    q_conj = quat_conjugate(q)
    tmp = quat_multiply(q, q_v)
    rotated = quat_multiply(tmp, q_conj)
    return rotated[1:]  # vector部分

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
    def __init__(self, policy, env_cfg, obs_cfg, env):
        self.startPos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.policy = policy
        # self.env_cfg = env_cfg
        # self.obs_cfg = obs_cfg
        self.env = env
        self.device = "cpu"
        self.num_envs = 1
        self.num_actions = self.env.env_cfg["num_actions"]
        self.action_scale = self.env.env_cfg["action_scale"]
        self.num_commands = self.env.command_cfg["num_commands"]
        self.actions = torch.zeros((1, self.num_actions), device="cpu", dtype=gs.tc_float)
        self.motor_dofs = [7, 11, 15, 6, 10, 14, 9, 13, 17, 8, 12, 16]
        
        self.obs, _ = self.env.reset()
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        # with self.csv_lock:
        #     self.csv_writer.writerow([timestamp] + target_dof_pos.tolist())
        #     self.csv_file.flush()torch.zeros_like(self.actions)

        self._targetPos_1 = [0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                             -0.2, 1.36, -2.65, 0.2, 1.36, -2.65]
        self._targetPos_2 = [0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                             0.0, 0.67, -1.3, 0.0, 0.67, -1.3]
        self._targetPos_3 = [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
                             -0.5, 1.36, -2.65, 0.5, 1.36, -2.65]
        self.duration_1 = 50 #500
        self.duration_2 = 50 #500
        self.duration_3 = 100 #1000
        self.duration_4 = 90 #900
        self.percent_1 = 0
        self.percent_2 = 0
        self.percent_3 = 0
        self.percent_4 = 0
        self.firstRun = True

        # スケール
        self.ang_vel_scale = obs_cfg["obs_scales"]["ang_vel"]
        self.dof_pos_scale = obs_cfg["obs_scales"]["dof_pos"]
        self.dof_vel_scale = obs_cfg["obs_scales"]["dof_vel"]

        # コマンド情報：ここでは実機用に、commands を [0.5, 0.0, 0.0] に設定
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands[0, 0] = 0.5

        # 実際には genesis_go2_env.py の commands_scale に準じた値を設定
        self.commands_scale = torch.tensor([2.0, 2.0, 0.25], dtype=torch.float32)
        
        # IMU情報：実機からは low_state.imu_state で受信（ここでは角速度のみ利用）
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        
        # 重力情報
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        
        # 関節角度のデフォルト値
        self.default_dof_pos = torch.tensor(
            [self.env.env_cfg["default_joint_angles"][name] for name in self.env.env_cfg["dof_names"]],
            device="cpu",
            dtype=gs.tc_float,
        )

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

        # Low-level 制御用にモード変更
        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # --- CSVログ出力の初期化 ---
        self.csv_lock = threading.Lock()
        self.csv_file = open("target_dof_pos_log.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        header = ["timestamp"] + dof_names
        self.csv_writer.writerow(header)
        self.csv_file.flush()
    
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
        また，IMU 情報（例：gyro）を更新し，IMUの quaternion から projected_gravity を算出する．
        """
        self.low_state = msg

    
    def construct_obs(self):
        """
        実機から取得した low_state から，genesis の obs と同様の45次元ベクトルを構築する．
        以下の順序で連結：
          - base_ang_vel (3) * ang_vel_scale
          - projected_gravity (3)
          - commands (3) * commands_scale
          - (dof_pos - default_dof_pos) (12) * dof_pos_scale
          - dof_vel (12) * dof_vel_scale
          - last_actions (12)
        """

        # ====================================================================================================
        # (WIP) Observation from Real Go2
        #self.dof_pos = torch.tensor([self.low_state.motor_state[i].q  for i in range(12)]).view(self.num_envs, 12)
        #self.dof_vel = torch.tensor([self.low_state.motor_state[i].dq for i in range(12)]).view(self.num_envs, 12)

        #if hasattr(self.low_state, "imu_state"):
        #    # imu_state.gyroscope に角速度（[wx, wy, wz]）が入っていると仮定
        #    #self.base_ang_vel = torch.tensor(self.low_state.imu_state.gyroscope, dtype=torch.float32)
        #    self.base_ang_vel = torch.tensor((self.low_state.imu_state.gyroscope, 3), device=self.device, dtype=gs.tc_float)

        #    # 世界座標系での重力は (x, y, z)=[0, 0, -1]
        #    self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
        #        self.num_envs, 1
        #    )
        #    # imu_state.quaternion は [x, y, z, w]
        #    q_xyzw = torch.tensor(self.low_state.imu_state.quaternion, device=self.device, dtype=gs.tc_float)

        #    # projected_gravity = rotate( global_gravity, inverse(q) )
        #    #proj_grav = rotate_vector_by_quat(self.global_gravity, quat_conjugate(q))
        #    #self.projected_gravity = torch.tensor(proj_grav, dtype=torch.float32)
        #    self.projected_gravity = transform_by_quat(self.global_gravity, inv_quat(q_xyzw))
        # ====================================================================================================

        # ====================================================================================================
        # Observation from Genesis sim
        self.dof_pos[:] = self.env.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.env.robot.get_dofs_velocity(self.motor_dofs)

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device="cpu", dtype=gs.tc_float).repeat(1, 1)
        self.base_quat = torch.zeros((1, 4), device="cpu", dtype=gs.tc_float)
        self.base_quat[:] = self.env.robot.get_quat()
        inv_base_quat = inv_quat(self.base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.env.robot.get_ang(), inv_base_quat)
        # ====================================================================================================

        self.obs = torch.cat(
            [
                self.base_ang_vel * self.ang_vel_scale,  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.dof_pos_scale,  # 12
                self.dof_vel * self.dof_vel_scale,  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

    
    def control_loop(self):
        """
        RecurrentThread により周期的に呼ばれる制御ループ．
        現在のセンサ値から obs を構築し，ポリシー推論，target_dof_pos の算出，コマンド送信を行う．
        """

        # --- CSVへログ出力 ---
        # 現在時刻（秒）と target_dof_pos（リストへ変換）を1行として記録
        # timestamp = time.time()
        # with self.csv_lock:
        #     self.csv_writer.writerow([timestamp] + target_dof_pos.tolist[0]())
        #     self.csv_file.flush()
        
        if self.firstRun:
            for i in range(12):
                self.startPos[i] = self.low_state.motor_state[i].q
                self.firstRun = False

        self.percent_1 += 1.0 / self.duration_1
        self.percent_1 = min(self.percent_1, 1)
        
        if self.percent_1 < 1:
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = (1 - self.percent_1) * self.startPos[i] + self.percent_1 * self._targetPos_1[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = 60.0 # self.env.env_cfg["kp"] *3
                self.low_cmd.motor_cmd[i].kd = 5.0 #self.env.env_cfg["kd"] *10
                self.low_cmd.motor_cmd[i].tau = 0

        elif (self.percent_1 == 1) and (self.percent_2 < 1):
            self.percent_2 += 1.0 / self.duration_2
            self.percent_2 = min(self.percent_2, 1)

            for i in range(12):
                self.low_cmd.motor_cmd[i].q =  (1 - self.percent_2) * self._targetPos_1[i] + self.percent_2 * self._targetPos_2[i]
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = 60.0 # self.env.env_cfg["kp"] *3
                self.low_cmd.motor_cmd[i].kd = 5.0 #self.env.env_cfg["kd"] *10
                self.low_cmd.motor_cmd[i].tau = 0

        elif (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 < 1):
            self.percent_3 += 1.0 / self.duration_3
            self.percent_3 = min(self.percent_3, 1)
            
            for i in range(12):
                self.low_cmd.motor_cmd[i].q =  self._targetPos_2[i] 
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = 60.0 # self.env.env_cfg["kp"] *3
                self.low_cmd.motor_cmd[i].kd = 5.0 #self.env.env_cfg["kd"] *10
                self.low_cmd.motor_cmd[i].tau = 0

        elif (self.percent_1 == 1) and (self.percent_2 == 1) and (self.percent_3 == 1) and (self.percent_4 <= 1):

            print("WALKING")
            with torch.no_grad():
                self.actions = self.policy(self.obs)
                print("UPDATE")

            target_dof_pos = self.actions * self.action_scale + self.default_dof_pos
            for i in range(12):
                self.low_cmd.motor_cmd[i].q = float(target_dof_pos[0][i])
                self.low_cmd.motor_cmd[i].dq = 0
                self.low_cmd.motor_cmd[i].kp = 60.0 #self.env.env_cfg["kp"] *3
                self.low_cmd.motor_cmd[i].kd = 5.0 #self.env.env_cfg["kd"] *10
                self.low_cmd.motor_cmd[i].tau = 0
            
            self.env.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
            self.env.scene.step()

        #timestamp = time.time()
        # with self.csv_lock:
        #     self.csv_writer.writerow([timestamp] + target_dof_pos[0].tolist())
        #     #self.csv_writer.writerow([timestamp] + self.target_dof_pos)
        #     self.csv_file.flush()

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)
    
        self.construct_obs()

    def start(self, control_dt):
        self.thread = RecurrentThread(interval=control_dt, target=self.control_loop, name="real_robot_control")
        self.thread.Start()
    
    def stop(self):
        if hasattr(self, "thread"):
            self.thread.Stop()
        # CSVファイルをクローズ
        with self.csv_lock:
            self.csv_file.close()

def main():
    # DDS 通信初期化
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
    
    exp_name = "go2-walking"
    ckpt = 1000
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = load_configs(exp_name)
    log_dir = os.path.join("../logs", exp_name)
    resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    
    gs.init(backend=gs.cpu)
    
    # 学習済みポリシーのロード
    # ここではダミー環境（Go2Env）を用いて OnPolicyRunner 経由でロード
    from my_genesis_go2_env import Go2Env
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
    deployer = RealRobotDeployer(policy, env_cfg, obs_cfg, env)
    
    print("Waiting for low_state from real robot...")
    while deployer.low_state is None:
        time.sleep(0.1)

    print("low_state received. Starting control loop.")
    print("FR_0 motor state: ", deployer.low_state.motor_state[go2.LegID["FR_0"]])
    print("IMU state: ", deployer.low_state.imu_state)
    print("Battery state: voltage: ", deployer.low_state.power_v, "current: ", deployer.low_state.power_a)
    print("Start !!!")

    deployer.start(control_dt=0.02)  # 50Hz 制御ループ
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping control loop.")
        deployer.stop()

if __name__ == "__main__":
    main()
