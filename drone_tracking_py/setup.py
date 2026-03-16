from setuptools import find_packages, setup
from glob import glob

package_name = 'drone_tracking_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/net_train/weights', glob('net_train/weights/*')),
        ('share/' + package_name + '/scripts', glob('scripts/*.py')),  # scripts soltos, se precisar
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lacea',
    maintainer_email='lacea@todo.todo',
    description='YOLO-based drone tracking',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracking = drone_tracking_py.tracking:main',
            'tracking_subscriber = drone_tracking_py.tracking_subscriber:main',
            'compressed_image_decoder = drone_tracking_py.compressed_image_decoder:main'
        ],
    },
)
