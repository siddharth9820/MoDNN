#ifndef FC_LAYER_H_
#define FC_LAYER_H_

#include "layers.h"

#define USE_CUBLAS true //DONT SET TO FALSE
#define TILE_SIZE  32
#define BLOCK_SIZE 8


namespace layers{


/*! \class FCLayer
  \brief Fully Connected Layer

  This class is used to create a Fully Connected layer.
*/  
class FCLayer : public layers::Layer
 {
  public:
    cublasHandle_t handle; /*!< CUBLAS Handle. */

    /**
     * Intializes FC Layer Class.
     * @param cudnn CUBLAS handle
     * @param batch_size Input batch size.
     * @param input_height Input data height. 
     * @param height Output height.
     */
    FCLayer(cublasHandle_t cublas,int batch_size,int input_height,int output_height);

    /**
     * Return the no of bytes occupied by the output and set the shape of the output in the passed array pointer.
     * @param shape Shape of the output is set in this array.
     */
    int get_output_shape_and_bytes(int shape[]);

    /**
     * Forward pass 
     * @param d_input Input buffer pointer.
     * @param d_kernel Params buffer pointer.
     * @param d_output output buffer pointer.
     */
    void forward(float* d_input, float * d_kernel, float * d_output); 

    /**
     * Backward pass 
     * @param alpha alpha is multiplied to the computed values before writing it in the buffers (Set to 1 usually).
     * @param beta_weights beta_weights is multiplied to the params gradient before the gradient is added to the d_diffkernel buffer. (Usually set to 0, set to 1 to accumulate gradients).
     * @param beta_input beta_input is multiplied to the input gradient before the gradient is added to the d_diffinput buffer. (Usually set to 0, set to 1 to accumulate gradients).
     * @param d_input Input buffer pointer.
     * @param d_kernel Params buffer pointer.
     * @param d_diffkernel Kernel gradients buffer pointer.
     * @param d_diffinput Input gradients buffer pointer.
     * @param d_diffoutput Output gradients buffer pointer.
     * @param lr Learning rate.
     */
    void backward(float alpha, float beta_weights, float beta_input, float *d_input, float* d_kernel,float *d_diffkernel,float *d_diffinput, float *d_diffoutput,float lr); //checked with numpy for parameter gradient
    
    /**
     * Allocate memory for params and param gradients. 
     * Returns the size of the params.
     * @param d_kernel Params buffer pointer.
     * @param d_diffkernel Param gradients buffer pointer.
     */
    int allocate_internal_mem(float **d_kernel,float **d_diffkernel);
    
    /**
     * Update params by applying gradients.
     * @param d_kernel Pointer to kernel buffer.
     * @param d_diffkernel Pointer to kernel gradients buffer.
     * @param lr Learning rate.
     * @param compute_stream Cuda stream for computations.
     */
    void update_weights(float* d_kernel, float* d_diffkernel, float lr,cudaStream_t compute_stream);
    
    /**
     * Randomly generate and populate params.
     * @param d_kernel params buffer pointer.
     */
    void populate_filter_params(float *d_kernel);

    /**
     * Return the no of bytes occupied by the input and set the shape of the input.
     * @param shape Shape of the input is set in this array.
     */
    int get_input_shape_and_bytes(int shape[]);

    /**
     * Return the no of bytes occupied by the parameters and set the shape of the parameters.
     * @param shape Shape of the parameters is set in this array.
     */
    int get_params_shape_and_bytes(int shape[]);

    /**
     * Returns the total memory occupied by Convolution Layer.
     */
    int get_total_memory();
 };
}

 #endif
