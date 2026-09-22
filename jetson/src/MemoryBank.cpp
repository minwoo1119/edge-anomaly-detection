#include "MemoryBank.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>


namespace {

std::uint16_t readUint16(std::istream& stream) {
    std::array<unsigned char, 2> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!stream) {
        throw std::runtime_error("Unexpected end of .npy header.");
    }
    return static_cast<std::uint16_t>(bytes[0])
        | (static_cast<std::uint16_t>(bytes[1]) << 8U);
}


std::uint32_t readUint32(std::istream& stream) {
    std::array<unsigned char, 4> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!stream) {
        throw std::runtime_error("Unexpected end of .npy header.");
    }
    return static_cast<std::uint32_t>(bytes[0])
        | (static_cast<std::uint32_t>(bytes[1]) << 8U)
        | (static_cast<std::uint32_t>(bytes[2]) << 16U)
        | (static_cast<std::uint32_t>(bytes[3]) << 24U);
}


float halfToFloat(std::uint16_t half) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(half) & 0x8000U) << 16U;
    std::uint32_t exponent = (static_cast<std::uint32_t>(half) >> 10U) & 0x1FU;
    std::uint32_t fraction = static_cast<std::uint32_t>(half) & 0x03FFU;
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            exponent = 127U - 15U + 1U;
            while ((fraction & 0x0400U) == 0) {
                fraction <<= 1U;
                --exponent;
            }
            fraction &= 0x03FFU;
            bits = sign | (exponent << 23U) | (fraction << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (fraction << 13U);
    } else {
        exponent += 127U - 15U;
        bits = sign | (exponent << 23U) | (fraction << 13U);
    }

    float value = 0.0F;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}


std::size_t checkedElementCount(std::size_t rows, std::size_t dimensions) {
    if (rows == 0 || dimensions == 0) {
        throw std::runtime_error("Memory bank shape must be non-empty.");
    }
    if (rows > std::numeric_limits<std::size_t>::max() / dimensions) {
        throw std::overflow_error("Memory bank element count overflow.");
    }
    return rows * dimensions;
}

}  // namespace


MemoryBank MemoryBank::loadNpy(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open memory bank: " + path);
    }

    std::array<unsigned char, 6> magic{};
    file.read(reinterpret_cast<char*>(magic.data()), magic.size());
    const std::array<unsigned char, 6> expected{
        0x93U, 'N', 'U', 'M', 'P', 'Y'
    };
    if (!file || magic != expected) {
        throw std::runtime_error("Invalid NumPy .npy magic bytes: " + path);
    }

    unsigned char major = 0;
    unsigned char minor = 0;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    if (!file || major < 1 || major > 3) {
        throw std::runtime_error("Unsupported NumPy .npy format version.");
    }
    (void)minor;

    const std::size_t headerLength = major == 1
        ? readUint16(file)
        : readUint32(file);
    std::string header(headerLength, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!file) {
        throw std::runtime_error("Failed to read NumPy .npy header.");
    }

    const std::regex descrPattern("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]");
    const std::regex fortranPattern("['\"]fortran_order['\"]\\s*:\\s*(True|False)");
    const std::regex shapePattern(
        "['\"]shape['\"]\\s*:\\s*\\(\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*,?\\s*\\)"
    );
    std::smatch match;

    if (!std::regex_search(header, match, descrPattern)) {
        throw std::runtime_error("NumPy .npy header has no dtype descriptor.");
    }
    const std::string dtype = match[1].str();

    if (!std::regex_search(header, match, fortranPattern) || match[1] != "False") {
        throw std::runtime_error("Fortran-ordered memory banks are not supported.");
    }

    if (!std::regex_search(header, match, shapePattern)) {
        throw std::runtime_error("Memory bank must be a two-dimensional NumPy array.");
    }
    const std::size_t rows = std::stoull(match[1].str());
    const std::size_t dimensions = std::stoull(match[2].str());
    const std::size_t count = checkedElementCount(rows, dimensions);

    std::vector<float> values(count);
    if (dtype == "<f4" || dtype == "|f4" || dtype == "=f4") {
        file.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(count * sizeof(float))
        );
    } else if (dtype == "<f2" || dtype == "|f2" || dtype == "=f2") {
        std::vector<std::uint16_t> halves(count);
        file.read(
            reinterpret_cast<char*>(halves.data()),
            static_cast<std::streamsize>(count * sizeof(std::uint16_t))
        );
        if (file) {
            for (std::size_t index = 0; index < count; ++index) {
                values[index] = halfToFloat(halves[index]);
            }
        }
    } else {
        throw std::runtime_error(
            "Memory bank dtype must be little-endian float32 or float16, got: "
            + dtype
        );
    }

    if (!file) {
        throw std::runtime_error("Memory bank payload is truncated: " + path);
    }
    return MemoryBank(std::move(values), rows, dimensions);
}


MemoryBank::MemoryBank(
    std::vector<float> values,
    std::size_t rows,
    std::size_t dimensions
)
    : values_(std::move(values)),
      rows_(rows),
      dimensions_(dimensions) {
    if (values_.size() != checkedElementCount(rows_, dimensions_)) {
        throw std::invalid_argument("Memory bank values do not match its shape.");
    }
}


const float* MemoryBank::row(std::size_t index) const {
    if (index >= rows_) {
        throw std::out_of_range("Memory bank row index is out of range.");
    }
    return values_.data() + index * dimensions_;
}


const std::vector<float>& MemoryBank::values() const noexcept {
    return values_;
}


std::size_t MemoryBank::rows() const noexcept {
    return rows_;
}


std::size_t MemoryBank::dimensions() const noexcept {
    return dimensions_;
}


std::size_t MemoryBank::sizeBytes() const noexcept {
    return values_.size() * sizeof(float);
}
