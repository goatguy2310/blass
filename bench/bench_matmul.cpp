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

BENCHMARK(BM_Matmul2D_Square)->Args({256})
                                 ->Args({512})
                                 ->Args({1024})
                                 ->Args({2048})->Unit(benchmark::kMillisecond)->Iterations(5);
BENCHMARK_MAIN();