#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layers.h"

namespace layers {

  /*! \class Relu
    \brief Relu Layer

    This class is used to create a Relu layer.
  */ 
  class relu : public Layer
  {

    public:
      cudnnHandle_t handle; /*!< CUDNN Handle. */
      cudnnTensorDescriptor_t input_descriptor; /*!< cudnnTensor descriptor for input data to this layer. */
      cudnnTensorDescriptor_t output_descriptor; /*!< cudnnTensor descriptor for ouput data from this layer. */
      cudnnActivationDescriptor_t activation_descriptor; /*!< cudnn descriptor for activation operation. */

    /**
     * Intializes Relu Layer Class.
     * @param batch_size Input batch size.
     * @param input_channels Input data channels.
     * @param input_height Input data height. 
     * @param input_width Input data width.
     */
      relu(cudnnHandle_t cudnn,int batch_size,int input_channels,int input_height,int input_width);

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
      * Forward pass of Relu
      * @param d_input Input buffer pointer to the relu layer.
      * @param d_output Output buffer pointer from the relu layer.
      */
      void forward(float* d_input, float * d_output);

     /**
      * Backward pass of Relu
      * @param d_input Input buffer pointer to the relu layer.
      * @param d_output Output buffer pointer from the relu layer.
      * @param d_diffinput Input gradients buffer pointer.
      * @param d_diffoutput Output gradients buffer pointer.
      */
      void backward(float * d_input, float *d_output, float *d_diffinput, float *d_diffoutput);

     /**
      * Return the total memory occupied by Flattern Layer.
      */
      int get_total_memory();
  };
}
#endif
