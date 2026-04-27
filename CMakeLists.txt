cmake_minimum_required(VERSION 3.28)
project(VideoEnhancer LANGUAGES CXX)

# ייצוא פקודות עבור clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# אכיפת סטנדרט C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 1. מציאת כל הספריות (חובה למקם לפני יצירת ה-Target!)
find_package(OpenCV REQUIRED)
find_package(mdspan CONFIG REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenVINO REQUIRED)

# 2. יצירת ה-Target
add_executable(${PROJECT_NAME} main.cpp)

# 3. הגדרות קומפילציה לפי סוג הבנייה
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE -fsanitize=address,undefined -g -O0)
    target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=address,undefined)
    message(STATUS "Build type is Debug: Sanitizers ENABLED")
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -O3 -march=native)
    message(STATUS "Build type is Release: Sanitizers DISABLED, Optimizations ENABLED")
endif()

# 4. קישור כל הספריות אל ה-Target שכבר נוצר
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} std::mdspan TBB::tbb openvino::runtime)