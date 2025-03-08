# unitree-go2-genesis

## About

This is experimental code for sim2real with the go2 forward gait obtained from the Genesis locomotion example.

## Notice

これは実験に用いたコードを共有するために公開されており，本リポジトリの内容を用いたことによる一切の損害に対する責任を負いません．
自己責任の下でご使用ください．リファクタリングなどのメンテナンスは随時行われる予定です．

This repository is made public in order to share the code used in the experiments, and we are not responsible for any damages caused by the use of the contents of this repository. Please use it at your own risk. Refactoring and other maintenance will be carried out from time to time.

## Installation

1. Clone this repository
   ```bash
   cd ~
   git clone https://github.com/Jizai-inc/unitree-go2-genesis
   cd unitree-go2-genesis
   git submodule update --init --recursive
   ```

1. Set venv
   ```bash
   cd ~/unitree-go2-genesis
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
   cd ~/unitree-go2-genesis/unitree_sdk2_python
   pip3 install -e .
   ```

1. Install rsl_rl
   ```bash
   cd ~/unitree-go2-genesis/rsl_rl
   git checkout v1.0.2 && pip3 install -e .
   ```

## Run

1. Connect to go2

   Please refer [Unitree Documentation / Go2 Quick Start](https://support.unitree.com/home/en/developer/Quick_start).

1. Run script
   ```bash
   cd ~/unitree-go2-genesis/src
   python sim2real_walk_2.py <YOUR_IF_NAME>
   ```

If go2 crouches down once, stands up, and starts walking, you've succeeded!

**Let's enjoy your sim2real !**