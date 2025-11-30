cmake_minimum_required(VERSION 4.0)
project(HandMatrix)

set(CMAKE_CXX_STANDARD 20)

# OpenCV Pfad setzen
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")

# OpenCV suchen
find_package(OpenCV REQUIRED)

# Eigen Library einbinden (Wichtig für NeuralNet.h)
include_directories("C:/eigen")

# Executable erstellen (Main + Header)
add_executable(HandMatrix main.cpp NeuralNet.h)

# Libraries linken
target_link_libraries(HandMatrix PRIVATE ${OpenCV_LIBS})