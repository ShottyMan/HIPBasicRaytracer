cmake_minimum_required(VERSION 3.22)

project(HIPRaytracing VERSION 0.1 DESCRIPTION "HIP implemtation of Raytacing tutorial" LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


#add_compile_options(-fsanitize=address)
#add_link_options(-fsanitize=address)
message(STATUS ${CMAKE_PREFIX_PATH})
find_package(hip REQUIRED)
find_package(rocrand REQUIRED)
find_package(hiprand REQUIRED)

#enable_language(hip)
#set(CMAKE_hip_COMPILER "hipcc")

set(CMAKE_CXX_COMPILER "hipcc")
add_compile_options(-O0 -ggdb)


include_directories(includes)

add_subdirectory(libs)

add_subdirectory(src)

target_link_libraries(HIPRaytracing PRIVATE hip::host hip::device hip::hiprand)

target_link_libraries(HIPRaytracing PRIVATE roc::rocrand)

set_target_properties(HIPRaytracing PROPERTIES HIP_STANDARD 17)
