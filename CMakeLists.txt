cmake_minimum_required(VERSION 3.16)
project(VehiclePedestrianTrafficAnalysisSystem CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn video calib3d highgui)

add_executable(vehicle-and-pedestrian-traffic-analysis-system
    src/main.cpp
    src/Detector.cpp
    src/KalmanFilterXYWH.cpp
    src/STrack.cpp
    src/Matching.cpp
    src/Hungarian.cpp
    src/GMC.cpp
    src/BotSort.cpp
)

target_include_directories(vehicle-and-pedestrian-traffic-analysis-system PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vehicle-and-pedestrian-traffic-analysis-system PRIVATE ${OpenCV_LIBS})
