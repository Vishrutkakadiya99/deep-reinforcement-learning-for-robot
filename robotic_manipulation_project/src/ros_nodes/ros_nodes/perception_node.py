# In file: src/ros_nodes/perception_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image       # Input
from std_msgs.msg import Float32MultiArray  # Output
import cv_bridge
import torch
import cv2
import numpy as np

# Import the CNN model from our perception package
from perception.cnn_model import PerceptionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.get_logger().info('Perception Node Started.')

        # --- Parameters ---
        self.image_size = 84   # Must match the CNN input
        self.state_dim = 256   # Must match the CNN output
        
        # --- Models ---
        self.bridge = cv_bridge.CvBridge()
        self.cnn_model = PerceptionCNN(output_dim=self.state_dim).to(device)
        self.cnn_model.eval() # Set to evaluation mode (no gradients)
        
        # In a real project, you'd load trained weights:
        # self.cnn_model.load_state_dict(torch.load('path/to/cnn_weights.pth'))
        
        # --- ROS2 Publishers & Subscribers ---
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',  # Topic from Gazebo camera
            self.image_callback,
            10) # 10 is the queue size
            
        self.state_pub = self.create_publisher(
            Float32MultiArray,
            '/perception/state', # Topic for the DRL agent
            10)

    def image_callback(self, msg):
        try:
            # 1. Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # 2. Pre-process the image
            processed_image = self.preprocess(cv_image)
            
            # 3. Convert to PyTorch Tensor
            # [H, W, C] -> [C, H, W] -> [B, C, H, W]
            img_tensor = torch.FloatTensor(processed_image).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 4. Run the CNN model
            with torch.no_grad():
                state_vector = self.cnn_model(img_tensor)
            
            # 5. Publish the state as a Float32MultiArray
            state_msg = Float32MultiArray()
            # Convert tensor to a standard Python list
            state_msg.data = state_vector.cpu().flatten().tolist()
            self.state_pub.publish(state_msg)

        except cv_bridge.CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
        except Exception as e:
            self.get_logger().error(f'Perception Error: {e}')

    def preprocess(self, img):
        # Resize to the model's expected input size (e.g., 84x84)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return img


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
