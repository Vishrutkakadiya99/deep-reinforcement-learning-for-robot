# Deep Reinforcement Learning for Robotic Manipulation (ROS2)

This repository provides a modular, professional template for a Deep Reinforcement Learning (DRL) project aimed at robotic manipulation. It is built with **ROS2** and **PyTorch**, demonstrating a clean separation of concerns between perception, agent logic, and ROS2 communication.

## Core Technologies
* **ROS2 (Humble/Iron):** The communication backbone for all components.
* **PyTorch:** The deep learning framework for all neural networks (Perception CNN, Actor, Critic).
* **OpenCV:** Used by the perception node for image pre-processing.
* **Modular Design:** A clean separation of concerns:
    * **`perception/`**: The "Eyes" - The vision system.
    * **`rl_agent/`**: The "Brain" - The DRL logic and models.
    * **`ros_nodes/`**: The "Nervous System" - Connects the "Brain" and "Eyes" to the ROS2 world.

---

## Project Architecture

This system runs as two primary ROS2 nodes that "talk" to each other and the robot simulation (e.g., Gazebo) using ROS2 topics.



### 1. `perception_node`
* **Subscribes** to `/camera/image_raw` (from your Gazebo simulation).
* **Processes** the raw image:
    1.  Converts it to an OpenCV image using `cv_bridge`.
    2.  Pre-processes it (e.g., resize to 84x84).
    3.  Feeds it into the `PerceptionCNN` (from `perception/cnn_model.py`).
* **Publishes** the resulting compact state vector (e.g., a 256-element array) to the `/perception/state` topic.

### 2. `agent_node`
* **Subscribes** to:
    1.  `/perception/state` (from the `perception_node`).
    2.  `/joint_states` (from your robot/Gazebo).
    3.  `/env/reward` and `/env/done` (from a reward/environment node you will create).
* **Combines** the perception state and joint state into one complete **`state`** vector.
* **Decides** on an action:
    1.  Passes the complete `state` to the `DRLAgent` (from `rl_agent/agent.py`).
    2.  The `DRLAgent` uses its `Actor` network (from `rl_agent/models.py`) to select the best `action`.
* **Publishes** the chosen `action` (e.g., joint velocities) to `/robot_controller/commands`.
* **Learns** (in the background):
    1.  Stores `(state, action, reward, next_state, done)` transitions in its Replay Buffer.
    2.  Periodically calls its `train()` method to update the Actor and Critic networks.

---

## Setup and Installation

This project is structured as a ROS2 `ament_python` package.

### 1. Prerequisites
* A working ROS2 environment (e.g., Humble).
* A Python environment with the following packages:
    ```bash
    pip install torch numpy opencv-python
    ```
* A ROS2 workspace (e.g., `~/ros2_ws`).

### 2. Build Instructions
1.  Clone this repository into your ROS2 workspace's `src` folder:
    ```bash
    cd ~/ros2_ws/src
    git clone [https://github.com/YOUR_USERNAME/robotic_manipulation_project.git](https://github.com/YOUR_USERNAME/robotic_manipulation_project.git)
    ```
2.  Navigate to the root of your workspace:
    ```bash
    cd ~/ros2_ws
    ```
3.  Install dependencies (if any are listed in `package.xml`):
    ```bash
    rosdep install -i --from-path src --rosdistro $ROS_DISTRO -y
    ```
4.  Build the package:
    ```bash
    colcon build --packages-select robotic_manipulation
    ```
5.  Source your workspace:
    ```bash
    source install/setup.bash
    ```

---

## How to Run

The included launch file will start the DRL agent and perception nodes.

**Note:** For this to work, you must *also* be running a simulation (like Gazebo) that provides the `/camera/image_raw` and `/joint_states` topics.

```bash
# Source your workspace
source install/setup.bash

# Run the launch file
ros2 launch robotic_manipulation start_simulation.launch.py
