#include "NearestNeighborSearch.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>


SearchResult CpuBruteForceSearch::search(
    const float* queries,
    std::size_t queryCount,
    std::size_t dimensions,
    const MemoryBank& memoryBank
) const {
    if (queries == nullptr && queryCount != 0) {
        throw std::invalid_argument("Query pointer is null.");
    }
    if (dimensions == 0 || dimensions != memoryBank.dimensions()) {
        throw std::invalid_argument(
            "Query dimensions do not match the memory bank dimensions."
        );
    }

    SearchResult result;
    result.distances.resize(queryCount);
    result.indices.resize(queryCount);

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t queryIndex = 0;
         queryIndex < static_cast<std::ptrdiff_t>(queryCount);
         ++queryIndex) {
        const float* query = queries
            + static_cast<std::size_t>(queryIndex) * dimensions;
        float bestSquaredDistance = std::numeric_limits<float>::infinity();
        std::size_t bestIndex = 0;

        for (std::size_t bankIndex = 0;
             bankIndex < memoryBank.rows();
             ++bankIndex) {
            const float* bankRow = memoryBank.row(bankIndex);
            float squaredDistance = 0.0F;

            #pragma omp simd reduction(+:squaredDistance)
            for (std::ptrdiff_t dimension = 0;
                 dimension < static_cast<std::ptrdiff_t>(dimensions);
                 ++dimension) {
                const float difference = query[dimension] - bankRow[dimension];
                squaredDistance += difference * difference;
            }

            if (squaredDistance < bestSquaredDistance) {
                bestSquaredDistance = squaredDistance;
                bestIndex = bankIndex;
            }
        }

        result.distances[queryIndex] = std::sqrt(bestSquaredDistance);
        result.indices[queryIndex] = bestIndex;
    }

    return result;
}
