add_executable(HIPRaytracing main.cxx camera.cxx)
target_link_libraries(HIPRaytracing PRIVATE hip::device)
