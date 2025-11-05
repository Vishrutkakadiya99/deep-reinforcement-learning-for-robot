# In file: src/ros_nodes/agent_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64, Bool
from sensor_msgs.msg import JointState
import numpy as np

# Import the DRL agent class
from rl_agent.agent import DRLAgent

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        self.get_logger().info('DRL Agent Node Started.')
        
        # --- Parameters ---
        # These must match your environment and agent
        self.cnn_state_dim = 256     # From PerceptionNode
        self.joint_state_dim = 7     # Example: 7-DOF arm
        self.action_dim = 7          # Example: 7 joint velocities
        self.max_action = 1.0        # Max velocity (rad/s)
        
        # Combined state dimension
        total_state_dim = self.cnn_state_dim + self.joint_state_dim
        
        # --- Agent ---
        self.agent = DRLAgent(total_state_dim, self.action_dim, self.max_action)
        self.current_cnn_state = None
        self.current_joint_state = None
        self.last_state = None
        self.last_action = None
        
        # --- ROS2 Publishers & Subscribers ---
        
        # State inputs
        self.cnn_state_sub = self.create_subscription(
            Float32MultiArray,
            '/perception/state', # From PerceptionNode
            self.cnn_state_callback,
            10)
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',     # From Gazebo/robot
            self.joint_state_callback,
            10)
            
        # Action output
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            '/robot_controller/commands', # To robot controller
            10)
            
        # --- Inputs for Training (from a separate reward/env node) ---
        self.reward_sub = self.create_subscription(Float64, '/env/reward', self.reward_callback, 10)
        self.done_sub = self.create_subscription(Bool, '/env/done', self.done_callback, 10)
        
        # Main control loop timer
        self.control_timer = self.create_timer(0.1, self.control_loop) # 10 Hz
        # Training loop timer
        self.train_timer = self.create_timer(1.0, self.train_loop) # 1 Hz
        
    def cnn_state_callback(self, msg):
        self.current_cnn_state = np.array(msg.data, dtype=np.float32)
        
    def joint_state_callback(self, msg):
        # Assuming joint order is correct.
        # In a real robot, you'd map names.
        self.current_joint_state = np.array(msg.position, dtype=np.float32)[:self.joint_state_dim]

    def control_loop(self):
        # Only act if we have received both parts of the state
        if self.current_cnn_state is None or self.current_joint_state is None:
            self.get_logger().warn('Waiting for full state (CNN and Joints)...', throttle_duration_sec=5.0)
            return
            
        # 1. Combine states
        combined_state = np.concatenate([self.current_cnn_state, self.current_joint_state])
        
        # 2. Select Action
        action = self.agent.select_action(combined_state)
        
        # 3. Publish Action
        action_msg = Float3S2MultiArray()
        action_msg.data = action.tolist()
        self.action_pub.publish(action_msg)
        
        # 4. Store for training
        self.last_state = combined_state
        self.last_action = action
        
    def reward_callback(self, msg):
        reward = msg.data
        
        # We need a *full* transition (S, A, R, S')
        if self.last_state is None or self.last_action is None or \
           self.current_cnn_state is None or self.current_joint_state is None:
            return
            
        # Create the next_state
        next_state = np.concatenate([self.current_cnn_state, self.current_joint_state])
        
        # Add to replay buffer
        self.agent.add_experience_to_buffer(
            self.last_state, self.last_action, next_state, reward, False
        )

    def done_callback(self, msg):
        if msg.data: # Episode is done
            self.get_logger().info('Episode Done. Storing final transition.')
            # Handle final transition
            if self.last_state is not None and self.last_action is not None:
                # 'next_state' doesn't matter as much as 'done=True'
                next_state = np.concatenate([self.current_cnn_state, self.current_joint_state])
                self.agent.add_experience_to_buffer(
                    self.last_state, self.last_action, next_state, 0.0, True # Assuming 0 reward on done
                )
            
            # Reset for next episode
            self.last_state = None
            self.last_action = None
            # A real env would reset the robot here
            
    def train_loop(self):
        self.agent.train() # Call the agent's training placeholder
        
def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
