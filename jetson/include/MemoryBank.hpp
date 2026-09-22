#pragma once

#include <cstddef>
#include <string>
#include <vector>


class MemoryBank {
public:
    static MemoryBank loadNpy(const std::string& path);

    MemoryBank(std::vector<float> values, std::size_t rows, std::size_t dimensions);

    const float* row(std::size_t index) const;
    const std::vector<float>& values() const noexcept;
    std::size_t rows() const noexcept;
    std::size_t dimensions() const noexcept;
    std::size_t sizeBytes() const noexcept;

private:
    std::vector<float> values_;
    std::size_t rows_{0};
    std::size_t dimensions_{0};
};
