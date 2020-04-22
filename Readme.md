Code for testing the repo is in main.cu. Use Makefile to compile.

For detailed documentation navigate to documentation/html/index.html and open the file in a browser.

Current features :-
1. Forward Pass and Backward Pass implementations for Convolution(CuDNN), Fully Connected layers (CuBLAS), Softmax(CuDNN), Relu (CuDNN), Pooling and Flatten.
2. Support for offloading and prefetching buffers.
3. 4 modes of training - Minimum memory, total memory, partial memory usage with prefetching and offloading heuristics(two kinds of prefetching heuristic).
4. DataLoader and Dataset classes for easy accessing and usage of data.
5. Virtual Memory Manager for timely defragmentation of device memory.
6. Mnist dataset class.
7. Clear documentation.


