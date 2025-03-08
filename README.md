# go2_sim2real

## About

This is experimental code for sim2real with the go2 forward gait obtained from the Genesis locomotion example.

## Notice

これは実験に用いたコードを共有するために公開されており，本リポジトリの内容を用いたことによる一切の損害に対する責任を負いません．自己責任の下でご使用ください．

This repository is made public in order to share the code used in the experiments, and we are not responsible for any damages caused by the use of the contents of this repository. Please use it at your own risk.

## Installation

1. Clone this repository
   ```bash
   cd ~
   git clone <repo-url>
   cd go2_sim2real
   git submodule update --init --recursive
   ```

1. Set venv
   ```bash
   cd ~/go2_sim2real
   python -m venv venv
   source venv/bin/activate
   pip3 install --upgrade pip
   ```

1. Install dependencies
   ```bash
   pip3 -r install requirements.txt
   ```

1. Install unitree_sdk2_python
   ```bash
   cd ~/go2_sim2real/unitree_sdk2_python
   pip3 install -e .
   ```

1. Install rsl_rl
   ```bash
   cd ~/go2_sim2real/rsl_rl
   git checkout v1.0.2 && pip3 install -e .
   ```

## Run

1. Place trained weight
   ```bash
   cd ~/go2_sim2real/logs
   cp <YOUR_TRAINED_WEIGHT_PATH> .
   ```

1. Connect to go2

   Please refer [Unitree Documentation / Go2 Quick Start](https://support.unitree.com/home/en/developer/Quick_start).

1. Run script
   ```bash
   cd ~/go2_sim2real/src
   python sim2real_walk_2.py <YOUR_IF_NAME>
   ```

If go2 crouches down once, stands up, and starts walking, you've succeeded!
**Let's enjoy your sim2real !**