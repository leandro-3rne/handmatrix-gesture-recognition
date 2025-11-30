cmake_minimum_required(VERSION 4.0)
project(HandMatrix_CNN)

set(CMAKE_CXX_STANDARD 20)

# Dein Pfad zu OpenCV
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")

find_package(OpenCV REQUIRED)

add_executable(HandMatrix_CNN main.cpp)

# DNN Modul ist im Hauptpaket von OpenCV enthalten
target_link_libraries(HandMatrix_CNN PRIVATE ${OpenCV_LIBS})