from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pkg_python_rewrite'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name,'launch'),glob(os.path.join('launch','*.launch.py')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='standard',
    maintainer_email='liguangcheng@standard-robots.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "service_adder_client_rewrite = pkg_python_rewrite.service_adder_client_rewrite:main",
            "service_adder_server_rewrite = pkg_python_rewrite.service_adder_server_rewrite:main",
            "action_move_client_rewrite = pkg_python_rewrite.action_move_client_rewrite:main",
            "action_move_server_rewrite= pkg_python_rewrite.action_move_server_rewrite:main",
            "topic_helloworld_pub = pkg_python_rewrite.topic_helloworld_pub_rewrite:main",
            "topic_helloworld_sub = pkg_python_rewrite.topic_helloworld_sub_rewrite:main",
        ],
    },
)
