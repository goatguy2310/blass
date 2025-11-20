# blass
Basic Linear Algebra Subprogram Simplified

## To-do list

### Tensor
- [x] More comprehensive constructors (for `initializer_list` for example)
- [x] Getters
- [x] Add stride and offset (easier to handle when data structure changes)
- [x] Flatten, reshape, slicing, transpose
- [ ] Constructors
    - [x] Fill
    - [ ] Random (+ distributions)
- [ ] Basic math ops
    - [x] Element-wise
    - [x] Broadcasting
    - [ ] Matmul, convolution
- [ ] Optimizations for non-contiguous
    - [ ] Remove recursion
    - [ ] Coalesce (flattening)
    - [ ] Permute (find dimension of size 1)
- [ ] Template functions for ops and strides
- [ ] Autograd

### Graph compiler (to optimize autograd graph)
- [ ] Graph
- [ ] Constant
- [ ] Whatever

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
