cmake_minimum_required(VERSION 4.0)
project(DataCollector)

set(CMAKE_CXX_STANDARD 20)

# OpenCV Pfad setzen (wie in deiner Vorlage)
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")

# OpenCV suchen
find_package(OpenCV REQUIRED)

# Executable erstellen (Nur die Collector Main)
add_executable(DataCollector main_collector.cpp)

# Libraries linken
target_link_libraries(DataCollector PRIVATE ${OpenCV_LIBS})