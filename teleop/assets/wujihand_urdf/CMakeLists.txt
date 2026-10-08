cmake_minimum_required(VERSION 3.8)
project(wujihand-urdf)

find_package(ament_cmake REQUIRED)

install(
    DIRECTORY launch meshes rviz urdf
    DESTINATION share/${PROJECT_NAME}
)

ament_package()
