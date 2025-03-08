# go2_sim2real

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

**Let's enjoy your sim2real !**