from setuptools import find_packages, setup
import os, glob

package_name = 'cat_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'), glob.glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'rviz'), glob.glob(os.path.join('rviz', '*.rviz')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Loukas K.',
    maintainer_email='loukas.kordos@uni-tuebingen.de',
    description='Main launch orchestrator for CaT Sim2Real',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
