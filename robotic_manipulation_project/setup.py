# In file: setup.py

from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'robotic_manipulation_project'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(where='src'),  
    package_dir={'': 'src'},  
    
    # This section tells ROS2 where to find your launch, config, and urdf files
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Add launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        
        # Add config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        
        # Add urdf files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf.xacro')),
    ],
    
    install_requires=[
        'setuptools',
        'torch',         
        'numpy',
        'opencv-python', 
        'rclpy',         
        'cv_bridge',     
        'sensor_msgs',   
        'std_msgs',      
        'geometry_msgs', 
    ],
    zip_safe=True,
    maintainer='vishrut',
    maintainer_email='vishrutkakadiya99@gmail.com',
    description='Deep Reinforcement Learning for Robotic Manipulation',
    license='Apache License 2.0',
    
    # This section creates the executables for your ROS2 nodes
    entry_points={
        'console_scripts': [
            'perception_node = ros_nodes.perception_node:main',
            'agent_node = ros_nodes.agent_node:main',
        ],
    },
)
