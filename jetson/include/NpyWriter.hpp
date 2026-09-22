#pragma once

#include <cstddef>
#include <string>
#include <vector>

void writeFloatNpy(
    const std::string& path,
    const float* values,
    std::size_t count,
    const std::vector<std::size_t>& shape
);
