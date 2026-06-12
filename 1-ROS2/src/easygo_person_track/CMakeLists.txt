cmake_minimum_required(VERSION 3.8)
project(easygo_person_track)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(easygo_follow_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(glog REQUIRED)



# 编译
set(EASYGO_PACKAGE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(EASYGO_BYTETRACK_DIR "${EASYGO_PACKAGE_ROOT}/bytetrack")
set(EASYGO_FRONT_DETECT_DIR "${EASYGO_PACKAGE_ROOT}/front_detect")
set(EASYGO_RKNN_INCLUDE_DIR "${EASYGO_PACKAGE_ROOT}/third_party/rknpu2/include")
set(EASYGO_RGA_INCLUDE_DIR "${EASYGO_PACKAGE_ROOT}/third_party/librga/include")
set(EASYGO_RKNN_LIB "${EASYGO_PACKAGE_ROOT}/third_party/rknpu2/lib/librknnrt.so")
set(EASYGO_RGA_LIB "${EASYGO_PACKAGE_ROOT}/third_party/librga/Linux/aarch64/librga.so")

set(EASYGO_PERSON_DETECT_SOURCES
  src/person_track_node.cpp
  ${EASYGO_BYTETRACK_DIR}/BYTETracker.cpp
  ${EASYGO_BYTETRACK_DIR}/kalmanFilter.cpp
  ${EASYGO_BYTETRACK_DIR}/lapjv.cpp
  ${EASYGO_BYTETRACK_DIR}/STrack.cpp
  ${EASYGO_BYTETRACK_DIR}/utils.cpp
  ${EASYGO_FRONT_DETECT_DIR}/yolov8_detector.cpp
)

add_executable(person_track_node ${EASYGO_PERSON_DETECT_SOURCES})
ament_target_dependencies(person_track_node
  rclcpp
  rclcpp_action
  easygo_follow_msgs
  sensor_msgs
  std_msgs
  cv_bridge)

target_include_directories(person_track_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${EASYGO_BYTETRACK_DIR}
  ${EASYGO_FRONT_DETECT_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EASYGO_RKNN_INCLUDE_DIR}
  ${EASYGO_RGA_INCLUDE_DIR}
)

target_link_libraries(person_track_node
  Eigen3::Eigen
  ${OpenCV_LIBS}
  dl
  ${EASYGO_RKNN_LIB}
  ${EASYGO_RGA_LIB}
  glog::glog
)

# 安装
set_target_properties(person_track_node PROPERTIES
  BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknpu2/lib;${CMAKE_CURRENT_SOURCE_DIR}/third_party/librga/Linux/aarch64"
  INSTALL_RPATH "$ORIGIN"
)
install(TARGETS
  person_track_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
  )


install(DIRECTORY
  model
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  ${EASYGO_RKNN_LIB}
  ${EASYGO_RGA_LIB}
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  third_party
  DESTINATION share/${PROJECT_NAME}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
