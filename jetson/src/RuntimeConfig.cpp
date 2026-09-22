#include "RuntimeConfig.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c); });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) { return std::isspace(c); }).base();
    return first >= last ? std::string{} : std::string(first, last);
}

std::string unquote(std::string value) {
    if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"')
        || (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

std::unordered_map<std::string, std::string> loadFlatYaml(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open runtime config: " + path);
    }
    std::unordered_map<std::string, std::string> values;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        line = trim(line);
        if (line.empty()) continue;
        const std::size_t separator = line.find(':');
        if (separator == std::string::npos) {
            throw std::runtime_error("Invalid config entry at line " + std::to_string(lineNumber));
        }
        const std::string key = trim(line.substr(0, separator));
        const std::string value = unquote(trim(line.substr(separator + 1)));
        if (key.empty() || value.empty()) {
            throw std::runtime_error("Empty config key or value at line " + std::to_string(lineNumber));
        }
        if (!values.emplace(key, value).second) {
            throw std::runtime_error("Duplicate config key: " + key);
        }
    }
    return values;
}

const std::string& required(const std::unordered_map<std::string, std::string>& values, const std::string& key) {
    const auto iterator = values.find(key);
    if (iterator == values.end()) throw std::runtime_error("Missing required config key: " + key);
    return iterator->second;
}

int parseInt(const std::string& value, const std::string& key) {
    std::size_t parsed = 0;
    const int result = std::stoi(value, &parsed);
    if (parsed != value.size()) throw std::runtime_error("Config key is not an integer: " + key);
    return result;
}

double parseDouble(const std::string& value, const std::string& key) {
    std::size_t parsed = 0;
    const double result = std::stod(value, &parsed);
    if (parsed != value.size()) throw std::runtime_error("Config key is not numeric: " + key);
    return result;
}
}  // namespace

RuntimeConfig RuntimeConfig::load(const std::string& path) {
    const auto values = loadFlatYaml(path);
    RuntimeConfig config;
    config.enginePath = required(values, "engine_path");
    config.memoryBankPath = required(values, "memory_bank_path");
    config.category = required(values, "category");
    config.model = required(values, "model");
    config.precision = required(values, "precision");
    config.bankPrecision = required(values, "bank_precision");
    config.nnBackend = required(values, "nn_backend");
    config.inputWidth = parseInt(required(values, "input_width"), "input_width");
    config.inputHeight = parseInt(required(values, "input_height"), "input_height");
    config.threshold = static_cast<float>(parseDouble(required(values, "threshold"), "threshold"));
    config.coresetRatio = parseDouble(required(values, "coreset_ratio"), "coreset_ratio");
    const int numNeighbors = parseInt(required(values, "num_neighbors"), "num_neighbors");
    config.gaussianSigma = parseDouble(required(values, "gaussian_sigma"), "gaussian_sigma");
    config.warmup = parseInt(required(values, "warmup"), "warmup");
    config.repeats = parseInt(required(values, "repeats"), "repeats");
    if (config.inputWidth <= 0 || config.inputHeight <= 0) {
        throw std::runtime_error("Input dimensions must be positive.");
    }
    if (numNeighbors <= 0 || config.warmup < 0 || config.repeats <= 0) {
        throw std::runtime_error("num_neighbors and repeats must be positive; warmup cannot be negative.");
    }
    if (config.coresetRatio <= 0.0 || config.coresetRatio > 1.0) {
        throw std::runtime_error("coreset_ratio must be in the interval (0, 1].");
    }
    if (config.nnBackend != "cpu") {
        throw std::runtime_error("Only the cpu NN backend is currently available.");
    }
    config.numNeighbors = static_cast<std::size_t>(numNeighbors);
    return config;
}
