#include <benchmark/benchmark.h>
#include "../src/tensor/tensor.h"

using namespace blass;

static void BM_ElemwiseAdd(benchmark::State& state) {
    size_t rows = state.range(0);
    size_t cols = state.range(1);
    Tensor<float> a = Tensor<float>::fill_random({rows, cols}, 0.0f, 1.0f);
    Tensor<float> b = {1};

    for (auto _ : state) {
        Tensor<float> c = a + b;
        benchmark::DoNotOptimize(c);
    }
    state.SetItemsProcessed(state.iterations() * rows * cols);
}

static void BM_ElemwiseMul(benchmark::State& state) {
    size_t rows = state.range(0);
    size_t cols = state.range(1);
    Tensor<float> a = Tensor<float>::fill_random({rows, cols}, 0.0f, 1.0f);
    Tensor<float> b = {1};

    for (auto _ : state) {
        Tensor<float> c = a * b;
        benchmark::DoNotOptimize(c);
    }
    state.SetItemsProcessed(state.iterations() * rows * cols);
}

BENCHMARK(BM_ElemwiseAdd)->Args({(int)1e8, (int)1})->Args({(int)1, (int)1e8})->Args({10000, 10000})->Unit(benchmark::kMillisecond)->Iterations(5);
BENCHMARK(BM_ElemwiseMul)->Args({(int)1e8, (int)1})->Args({(int)1, (int)1e8})->Args({10000, 10000})->Unit(benchmark::kMillisecond)->Iterations(5);

BENCHMARK_MAIN();