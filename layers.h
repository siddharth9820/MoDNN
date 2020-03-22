#include <cudnn.h>


enum padding_type{
  SAME,
  VALID
};

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


namespace layers
{
  class ConvLayer
  {
    public:
      cudnnHandle_t handle;
      cudnnTensorDescriptor_t input_descriptor;
      cudnnFilterDescriptor_t kernel_descriptor;
      cudnnConvolutionDescriptor_t convolution_descriptor;
      cudnnTensorDescriptor_t output_descriptor;
      cudnnTensorDescriptor_t filter_derivative_descriptor;
      cudnnConvolutionFwdAlgo_t convolution_algorithm;
      cudnnConvolutionBwdFilterAlgo_t filter_algo;
      cudnnConvolutionBwdDataAlgo_t data_algo;
      size_t forward_workspace_bytes, backward_workspace_bytes;
      int obatch_size,ochannels,oheight,owidth;
      int ibatch_size,ichannels,iheight,iwidth,ikernel_width,ikernel_height;



    ConvLayer(cudnnHandle_t cudnn,
                    int batch_size,
                    int input_height,
                    int input_width,
                    int input_channels,
                    int kernel_height,
                    int kernel_width,
                    int output_channels,
                    padding_type pad
             );
    int get_output_shape_and_bytes(int shape[]);
    int get_input_shape_and_bytes(int shape[]);
    size_t get_forward_workspace_bytes();
    size_t get_backward_workspace_bytes();
    size_t get_total_workspace_size();
    void forward(float alpha, float beta, float* d_input, float* d_kernel, void* d_workspace, float * d_output);
    int allocate_internal_mem(float **d_kernel, void **d_workspace);
    ~ConvLayer();

  };

}
