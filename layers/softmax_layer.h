#ifndef SOFTMAX_LAYER_H_
#define SOFTMAX_LAYER_H_

#include "layers.h"

namespace layers {

 /*! \class Softmax
  \brief Softmax Layer

  This class is used to create a Softmax layer.
 */ 
 class Softmax : public layers::Layer
 {

  public:
    cudnnHandle_t handle; /*!< CUDNN Handle. */
    cudnnTensorDescriptor_t input_descriptor; /*!< cudnnTensor descriptor for input data to this layer. */
    cudnnTensorDescriptor_t output_descriptor; /*!< cudnnTensor descriptor for ouput data from this layer. */
    cudnnTensorDescriptor_t diff_descriptor; /*!< #DEPRICATED# cudnnTensor descriptor for gradients data from this layer. */

    /**
     * Intializes Softmax Layer Class.
     * @param batch_size Input batch size.
     * @param input_height Input data height. 
     */
    Softmax(cudnnHandle_t cudnn,int batch_size,int input_height);

    /**
      * Return the no of bytes occupied by the output and set the shape of the output in the passed array.
      * @param shape Shape of the output is set in this array.
      */
    int get_output_shape_and_bytes(int shape[]);

    /**
      * Return the no of bytes occupied by the intput and set the shape of the input in the passed array.
      * @param shape Shape of the input is set in this array.
      */
    int get_input_shape_and_bytes(int shape[]);

    /**
      * Forward pass of Softmax
      * @param d_input Input buffer pointer to the softmax layer.
      * @param d_output Output buffer pointer from the softmax layer.
      */
    void forward(float* d_input, float * d_output);

    /**
      * Backward pass of Softmax
      * @param output Output buffer pointer from the relu layer.
      * @param diff Output gradients buffer pointer.
      * @param label Ground truth labels.
      */
    void backward(const int *label, float *diff, float * output);

    /**
      * Return the total memory occupied by Flattern Layer.
      */
    int get_total_memory();


 };
}
#endif
