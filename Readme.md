Code for testing the repo is in main.cu. Use Makefile to compile.

Current features :-
1. Forward Pass implementations for Convolution(CuDNN), Fully Connected layers (CuBLAS), Softmax(CuDNN) and Flatten.
2. Function to offload activation buffers for any layer to the host from the device has been implemented.
3. Backward pass implementation for Softmax and FCLayer - weights.

Note :- Everything is being tested with numpy for correcteness in parallel.
