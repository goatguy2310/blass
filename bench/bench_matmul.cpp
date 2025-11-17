#include <benchmark/benchmark.h>
#include "../src/tensor/tensor.h"

using namespace blass;

static void BM_Matmul2D_Square(benchmark::State& state) {
    size_t N = state.range(0);
    Tensor<float> a = Tensor<float>::fill_random({N, N}, 0.0f, 1.0f);
    Tensor<float> b = Tensor<float>::fill_random({N, N}, 0.0f, 1.0f);

    for (auto _ : state) {
        Tensor<float> c = matmul_2d(a, b);
        benchmark::DoNotOptimize(c);
    }
    state.SetItemsProcessed(state.iterations() * N * N * N);
}

static void BM_Matmul2D_Rectangular(benchmark::State& state) {
    size_t M = state.range(0);
    size_t K = state.range(1);
    size_t N = state.range(2);
    Tensor<float> a = Tensor<float>::fill_random({M, K}, 0.0f, 1.0f);
    Tensor<float> b = Tensor<float>::fill_random({K, N}, 0.0f, 1.0f);

    for (auto _ : state) {
        Tensor<float> c = matmul_2d(a, b);
        benchmark::DoNotOptimize(c);
    }
    state.SetItemsProcessed(state.iterations() * M * K * N);
}

static void BM_Matmul_Broadcast(benchmark::State& state) {
    size_t batch_size = state.range(0);
    size_t M = state.range(1);
    size_t K = state.range(2);
    size_t N = state.range(3);
    Tensor<float> a = Tensor<float>::fill_random({batch_size, M, K}, 0.0f, 1.0f);
    Tensor<float> b = Tensor<float>::fill_random({K, N}, 0.0f, 1.0f);

    for (auto _ : state) {
        Tensor<float> c = matmul(a, b);
        benchmark::DoNotOptimize(c);
    }
    state.SetItemsProcessed(state.iterations() * batch_size * M * K * N);
}

BENCHMARK(BM_Matmul2D_Square)->Args({256})
                                 ->Args({512})
                                 ->Args({1024})
                                 ->Args({2048})->Unit(benchmark::kMillisecond)->Iterations(5);
BENCHMARK(BM_Matmul2D_Rectangular)->Args({256, 512, 128})
                                    ->Args({512, 256, 1024})
                                    ->Args({1024, 512, 2048})
                                    ->Args({2048, 1024, 4096})
                                    ->Args({1, 100000000, 1})
                                    ->Args({10000, 1, 10000})->Unit(benchmark::kMillisecond)->Iterations(5);
BENCHMARK(BM_Matmul_Broadcast)->Args({10, 256, 512, 128})
                                ->Args({20, 512, 256, 1024})
                                ->Args({30, 1024, 512, 2048})->Unit(benchmark::kMillisecond)->Iterations(5);
BENCHMARK_MAIN();