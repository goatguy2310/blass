#include <benchmark/benchmark.h>
#include "../src/tensor/tensor.h"

using namespace blass;

static void BM_Convolve1D_NoPadding(benchmark::State& state) {
    size_t batch_size = state.range(0);
    size_t in_channels = state.range(1);
    size_t input_length = state.range(2);
    size_t kernel_size = state.range(3);
    Tensor<float> input = Tensor<float>::fill_random({batch_size, in_channels, input_length}, 0.0f, 1.0f);
    Tensor<float> kernel = Tensor<float>::fill_random({1, in_channels, kernel_size}, 0.0f, 1.0f);

    for (auto _ : state) {
        Tensor<float> output = convolve1D(input, kernel, 0);
        benchmark::DoNotOptimize(output);
    }
    size_t output_length = input_length - kernel_size + 1;
    state.SetItemsProcessed(state.iterations() * batch_size * output_length * kernel_size * in_channels);
}

static void BM_Convolve1D_WithPadding(benchmark::State& state) {
    size_t batch_size = state.range(0);
    size_t in_channels = state.range(1);
    size_t input_length = state.range(2);
    size_t kernel_size = state.range(3);
    Tensor<float> input = Tensor<float>::fill_random({batch_size, in_channels, input_length}, 0.0f, 1.0f);
    Tensor<float> kernel = Tensor<float>::fill_random({1, in_channels, kernel_size}, 0.0f, 1.0f);

    for (auto _ : state) {
        Tensor<float> output = convolve1D(input, kernel, 1);
        benchmark::DoNotOptimize(output);
    }
    size_t output_length = input_length;
    state.SetItemsProcessed(state.iterations() * batch_size * output_length * kernel_size * in_channels);
}

BENCHMARK(BM_Convolve1D_NoPadding)->Args({8, 1024, 10, 3})
                                    ->Args({16, 2048, 20, 5})
                                    ->Args({32, 4096, 40, 7})
                                    ->Args({64, 8192, 80, 9})
                                    ->Args({32, 256, 50, 3})
                                    ->Args({32, 512, 128, 5})
                                    ->Args({64, 768, 512, 3})->Unit(benchmark::kMillisecond)->Iterations(5)->UseRealTime();
BENCHMARK(BM_Convolve1D_WithPadding)->Args({8, 1024, 10, 3})
                                    ->Args({16, 2048, 20, 5})
                                    ->Args({32, 4096, 40, 7})
                                    ->Args({64, 8192, 80, 9})
                                    ->Args({32, 256, 50, 3})
                                    ->Args({32, 512, 128, 5})
                                    ->Args({64, 768, 512, 3})->Unit(benchmark::kMillisecond)->Iterations(5)->UseRealTime();
BENCHMARK_MAIN();