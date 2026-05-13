#pragma once
#include <array>
#include <string_view>

enum Device {
    CPU, CUDA, COUNT
};
constexpr auto DEVICE_STRINGS = std::to_array<std::string_view>({"cpu", "cuda"});
static std::string_view getDeviceName(const Device d) {
    return DEVICE_STRINGS[static_cast<int>(d)];
}
static_assert(DEVICE_STRINGS.size() == static_cast<size_t>(Device::COUNT),
              "FATAL: DEVICE_STRINGS array size does not match Device enum count!");