/*
Mega simple rng library
*/

#pragma once

#include <random>
#include <cstdint>

namespace randomt {
    uint64_t seed;
    std::mt19937 rng(seed);

    void set_seed(uint64_t new_seed) {
        seed = new_seed;
        rng.seed(seed);
    }

    uint64_t get_seed() {
        return seed;
    }

    float rand() {
        static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng);
    }

    float randn() {
        static std::normal_distribution<float> dist(0.0f, 1.0f);
        return dist(rng);
    }

    float uniform(float min_val, float max_val) {
        return min_val + (max_val - min_val) * rand();
    }

    float normal(float mean, float stddev) {
        return mean + stddev * randn();
    }

    int randint(int min_val, int max_val) {
        return std::uniform_int_distribution<int>(min_val, max_val)(rng);
    }
}