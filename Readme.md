# Code for testing the repo is in main.cu. Use Makefile to compile.

## For detailed documentation navigate to documentation/html/index.html and open the file in a browser.


# Workflow of main.cu:
1. MNIST Dataset class in initialized and data is parsed.
2. DataLoader object is created and initialized with MNIST Dataset.
3. Network specification is mentioned as a vector of strings.
4. seqNetwork object is created with network specification. (Option to specify memory budget and choose subbatch selection algorithm.)
5. Virtual memory manager is intialized with the memory mentioned takes the chunk of memory from the GPU.
6. Trainer function is called and training is done.

# Current features :-
1. Forward Pass and Backward Pass implementations for Convolution(CuDNN), Fully Connected layers (CuBLAS), Softmax(CuDNN), Relu (CuDNN), Pooling and Flatten.
2. Support for offloading and prefetching buffers.
3. 4 modes of training - Minimum memory, total memory, partial memory usage with prefetching and offloading heuristics(two kinds of prefetching heuristic).
4. DataLoader and Dataset classes for easy accessing and usage of data.
5. Virtual Memory Manager for timely defragmentation of device memory.
6. Mnist dataset class.
7. Clear documentation.

## Setting Batch_size and learning rate :
1. Navigate to layers/layer.h and change the defined values in lines 32,33.

## Changing the trainer algorithm :
1. Check line 81 in main.cu

## Setting memory budget of network: 
1. Check line 71 in main.cu

## Setting memory size of the VMM:
1. Check line 77 in main.cu