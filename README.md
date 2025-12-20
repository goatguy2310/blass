# blass
Basic Linear Algebra Subprogram Simplified

\+Basic LLM Inference Engine

## To-do list

### Tensor
- [x] More comprehensive constructors (for `initializer_list` for example)
- [x] Getters
- [x] Add stride and offset (easier to handle when data structure changes)
- [x] Flatten, reshape, slicing, transpose
- [ ] Constructors
    - [x] Fill
    - [ ] Random
- [x] Basic math ops
    - [x] Element-wise
    - [x] Broadcasting
    - [x] Matmul, convolution
- [ ] Optimizations for non-contiguous
    - [x] Remove recursion
    - [ ] Coalesce (flattening)
    - [ ] Permute (find dimension of size 1)
    - [ ] Optimize blocking to make it usable
- [ ] Random module (distribution, seed, etc.)
- [ ] Model loading
    - [x] Parsing GGUF file
    - [ ] Tokenizer library API
    - [ ] Implementing all NN layers
- [ ] Template functions for ops and strides
- [ ] Autograd

### Perf measurement
- [ ] Record useful metrics
- [ ] Visualization

### Graph compiler (to optimize autograd graph, and maybe inference for ONNX)
- [ ] Graph

## Usage

To compile main
```
make
```

To run all tests & benchmarks
```
make tests
make bench
```

To run individual tests & benchmarks
```
make tests_elemwise

make bench_elemwise
make bench_matmul
```
