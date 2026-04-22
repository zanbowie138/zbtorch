#pragma once
#include <array>
#include <format>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <core/device.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

static Device parseDevice(const py::object& obj) {
    auto s = py::str(obj).cast<std::string>();
    for (const auto& [i, name] : std::views::enumerate(DEVICE_STRINGS))
        if (s == name) return static_cast<Device>(i);
    throw std::invalid_argument(
        std::format("Unknown device: {}. Supported devices are: {}", s,
            DEVICE_STRINGS | std::views::join_with(std::string_view(", ")) | std::ranges::to<std::string>())
    );
}

static std::string_view getDeviceName(const Device d) {
    return DEVICE_STRINGS[static_cast<int>(d)];
}