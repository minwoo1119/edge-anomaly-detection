#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

void writeFloatNpy(
    const std::string& path,
    const float* values,
    std::size_t count,
    const std::vector<std::size_t>& shape
);

void writeUint64Npy(
    const std::string& path,
    const std::uint64_t* values,
    std::size_t count,
    const std::vector<std::size_t>& shape
);
