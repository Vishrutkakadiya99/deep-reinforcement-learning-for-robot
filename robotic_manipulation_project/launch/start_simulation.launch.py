# In file: launch/start_simulation.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # --- Placeholder for Gazebo ---
    # In a real project, you would launch Gazebo and your robot here
    # using 'IncludeLaunchDescription'
    # Example:
    # gazebo_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         get_package_share_directory('your_robot_gazebo_pkg'),
    #         '/launch/start_world.launch.py'
    #     ])
    # )
    
    # --- Perception Node ---
    perception_node = Node(
        package='robotic_manipulation',     # Your package name from setup.py
        executable='perception_node', # The executable name from setup.py
        name='perception_node',
        output='screen'
    )
    
    # --- Agent Node ---
    agent_node = Node(
        package='robotic_manipulation',
        executable='agent_node',
        name='agent_node',
        output='screen'
    )

    return LaunchDescription([
        # gazebo_launch, # Uncomment when you have a sim
        perception_node,
        agent_node
    ])
