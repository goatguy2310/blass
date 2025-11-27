#include <benchmark/benchmark.h>
#include "../src/tensor/tensor.h"

using namespace blass;

static void BM_ElemwiseAddScalar(benchmark::State& state) {
    size_t rows = state.range(0);
    size_t cols = state.range(1);
    Tensor<float> a = Tensor<float>::fill_random({rows, cols}, 0.0f, 10.0f);
    Tensor<float> b = {3.0f};

    for (auto _ : state) {
        Tensor<float> c = a + b;
        benchmark::DoNotOptimize(c);
    }
    state.SetItemsProcessed(state.iterations() * rows * cols);
}

static void BM_ElemwiseMulScalar(benchmark::State& state) {
    size_t rows = state.range(0);
    size_t cols = state.range(1);
    Tensor<float> a = Tensor<float>::fill_random({rows, cols}, 0.0f, 10.0f);
    Tensor<float> b = {3.0f};

    for (auto _ : state) {
        Tensor<float> c = a * b;
        benchmark::DoNotOptimize(c);
    }
    state.SetItemsProcessed(state.iterations() * rows * cols);
}

static void BM_ElemwiseAddSimple(benchmark::State& state) {
    size_t dim1 = state.range(0);
    size_t dim2 = state.range(1);
    size_t dim3 = state.range(2);
    Tensor<float> a = Tensor<float>::fill_random({dim1, dim2, dim3}, 0.0f, 10.0f);
    Tensor<float> b = Tensor<float>::fill_random({dim1, dim2, dim3}, 0.0f, 10.0f);

    for (auto _ : state) {
        Tensor<float> c = a + b;
        benchmark::DoNotOptimize(c);
    }
    state.SetItemsProcessed(state.iterations() * dim1 * dim2 * dim3);
}

static void BM_ElemwiseAddBroadcast(benchmark::State& state) {
    size_t dim1 = state.range(0);
    size_t dim2 = state.range(1);
    size_t dim3 = state.range(2);
    Tensor<float> a = Tensor<float>::fill_random({dim1, 1, dim3}, 0.0f, 10.0f);
    Tensor<float> b = Tensor<float>::fill_random({dim1, dim2, 1}, 0.0f, 10.0f);

    for (auto _ : state) {
        Tensor<float> c = a + b;
        benchmark::DoNotOptimize(c);
    }
    state.SetItemsProcessed(state.iterations() * dim1 * dim2 * dim3);
}

static void BM_ElemwiseAddNonContiguous(benchmark::State& state) {
    size_t dim1 = state.range(0);
    size_t dim2 = state.range(1);
    size_t dim3 = state.range(2);
    Tensor<float> a = Tensor<float>::fill_random({dim1, 1, dim3}, 0.0f, 10.0f);
    Tensor<float> b = Tensor<float>::fill_random({dim3, dim2, 1}, 0.0f, 10.0f).transpose();

    for (auto _ : state) {
        Tensor<float> c = a + b;
        benchmark::DoNotOptimize(c);
    }
    state.SetItemsProcessed(state.iterations() * dim1 * dim2 * dim3);
}

BENCHMARK(BM_ElemwiseAddScalar)->Args({16, 16})
                         ->Args({64, 32})
                         ->Args({32, 64})
                         ->Args({(int)1e8, (int)1})
                         ->Args({(int)1, (int)1e8})
                         ->Args({10000, 10000})
                         ->Args({100, 1000000})
                         ->Args({1000000, 100})->Unit(benchmark::kMillisecond)->Iterations(5)->UseRealTime();
BENCHMARK(BM_ElemwiseMulScalar)->Args({16, 16})
                         ->Args({64, 32})
                         ->Args({32, 64})
                         ->Args({(int)1e8, (int)1})
                         ->Args({(int)1, (int)1e8})
                         ->Args({10000, 10000})
                         ->Args({100, 1000000})
                         ->Args({1000000, 100})->Unit(benchmark::kMillisecond)->Iterations(5)->UseRealTime();
BENCHMARK(BM_ElemwiseAddSimple)->Args({8, 8, 8})
                                 ->Args({16, 32, 16})
                                 ->Args({200, 200, 200})
                                 ->Args({500, 500, 500})
                                 ->Args({1000, 1000, 1000})->Unit(benchmark::kMillisecond)->Iterations(5)->UseRealTime();
BENCHMARK(BM_ElemwiseAddBroadcast)->Args({8, 8, 8})
                                 ->Args({16, 32, 16})
                                 ->Args({200, 200, 200})
                                 ->Args({500, 500, 500})
                                 ->Args({1000, 1000, 1000})->Unit(benchmark::kMillisecond)->Iterations(5)->UseRealTime();
BENCHMARK(BM_ElemwiseAddNonContiguous)->Args({8, 8, 8})
                                    ->Args({16, 32, 16})
                                    ->Args({200, 200, 200})
                                     ->Args({500, 500, 500})
                                     ->Args({1000, 1000, 1000})->Unit(benchmark::kMillisecond)->Iterations(5)->UseRealTime();
BENCHMARK_MAIN();