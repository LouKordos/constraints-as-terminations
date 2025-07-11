from setuptools import find_packages, setup

package_name = 'cmd_vel_zmq_translation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='loukas',
    maintainer_email='loukas@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ros2_cmd_vel_zmq_translation = cmd_vel_zmq_translation.ros2_cmd_vel_zmq_translation:main'
        ],
    },
)
