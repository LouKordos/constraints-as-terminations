import glob
import os

from setuptools import find_packages, setup


package_name = "locomposition_state_estimation"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer="Loukas Kordos",
    maintainer_email="loukas.kordos@tum.de",
    description="LiDAR and odometry launch files for LoComposition deployment",
    license="TODO: License declaration",
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
