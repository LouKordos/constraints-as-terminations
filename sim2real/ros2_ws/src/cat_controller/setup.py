import glob
import os

from setuptools import setup


package_name = "cat_controller"

setup(
    name=package_name,
    version="0.0.1",
    packages=[],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob.glob(os.path.join("launch", "*.launch.py")),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Loukas Kordos",
    maintainer_email="loukas.kordos@tum.de",
    description="Compatibility launches forwarding to locomposition_controller",
    license="Apache-2.0",
)
