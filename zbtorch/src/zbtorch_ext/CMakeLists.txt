add_library(zbtorch_core STATIC core/tensor.cpp
        include/zbtorch_ext/tensor.h)

target_include_directories(zbtorch_core PUBLIC include)

add_executable(zbtorch_standalone test/test.cpp)
target_link_libraries(zbtorch_standalone PRIVATE zbtorch_core)

pybind11_add_module(_C bindings.cpp)
target_link_libraries(_C PRIVATE zbtorch_core)

install(TARGETS _C
    LIBRARY DESTINATION zbtorch
    RUNTIME DESTINATION zbtorch   # Windows .pyd lands here
)