pybind11_add_module(_C test.cpp)

install(TARGETS _C
    LIBRARY DESTINATION zbtorch
    RUNTIME DESTINATION zbtorch   # Windows .pyd lands here
)