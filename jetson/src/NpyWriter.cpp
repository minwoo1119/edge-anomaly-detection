#include "NpyWriter.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace {
void writeNpy(
    const std::string& path,
    const void* values,
    std::size_t count,
    const std::vector<std::size_t>& shape,
    const std::string& descriptor,
    std::size_t itemSize
) {
    if (values == nullptr || shape.empty()) {
        throw std::invalid_argument("NumPy output values and shape must be non-empty.");
    }
    std::size_t expected = 1;
    for (const std::size_t dimension : shape) {
        if (dimension == 0 || expected > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::invalid_argument("Invalid NumPy output shape.");
        }
        expected *= dimension;
    }
    if (expected != count) {
        throw std::invalid_argument("NumPy output shape does not match value count.");
    }

    std::ostringstream shapeText;
    shapeText << '(';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) shapeText << ", ";
        shapeText << shape[index];
    }
    if (shape.size() == 1) shapeText << ',';
    shapeText << ')';

    std::string header = "{'descr': '" + descriptor + "', 'fortran_order': False, 'shape': "
        + shapeText.str() + ", }";
    constexpr std::size_t preambleSize = 10;
    const std::size_t padding = (16 - ((preambleSize + header.size() + 1) % 16)) % 16;
    header.append(padding, ' ');
    header.push_back('\n');
    if (header.size() > std::numeric_limits<std::uint16_t>::max()) {
        throw std::overflow_error("NumPy v1 header is too large.");
    }

    const std::filesystem::path outputPath(path);
    if (outputPath.has_parent_path()) std::filesystem::create_directories(outputPath.parent_path());
    std::ofstream output(outputPath, std::ios::binary);
    if (!output) throw std::runtime_error("Failed to create NumPy output: " + path);
    const unsigned char magic[] = {0x93U, 'N', 'U', 'M', 'P', 'Y', 1U, 0U};
    output.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    const std::uint16_t headerLength = static_cast<std::uint16_t>(header.size());
    const unsigned char length[] = {
        static_cast<unsigned char>(headerLength & 0xFFU),
        static_cast<unsigned char>((headerLength >> 8U) & 0xFFU)
    };
    output.write(reinterpret_cast<const char*>(length), sizeof(length));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / itemSize) {
        throw std::overflow_error("NumPy payload is too large.");
    }
    output.write(
        reinterpret_cast<const char*>(values),
        static_cast<std::streamsize>(count * itemSize)
    );
    if (!output) throw std::runtime_error("Failed to write NumPy output: " + path);
}
}  // namespace

void writeFloatNpy(
    const std::string& path,
    const float* values,
    std::size_t count,
    const std::vector<std::size_t>& shape
) {
    writeNpy(path, values, count, shape, "<f4", sizeof(float));
}

void writeUint64Npy(
    const std::string& path,
    const std::uint64_t* values,
    std::size_t count,
    const std::vector<std::size_t>& shape
) {
    writeNpy(path, values, count, shape, "<u8", sizeof(std::uint64_t));
}
