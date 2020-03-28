#include "conv_layer.h"
using namespace layers;

ConvLayer::ConvLayer(cudnnHandle_t cudnn,
    int batch_size,
    int input_height,
    int input_width,
    int input_channels,
    int kernel_height,
    int kernel_width,
    int output_channels,
    padding_type pad
)
{
    handle = cudnn;
    ibatch_size = batch_size;
    ichannels = input_channels;
    iheight = input_height;
    iwidth = input_width;
    ikernel_width = kernel_width;
    ikernel_height = kernel_height;

    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                /*format=*/CUDNN_TENSOR_NCHW,
                                /*dataType=*/CUDNN_DATA_FLOAT,
                                /*batch_size=*/batch_size,
                                /*channels=*/input_channels,
                                /*image_height=*/input_height,
                                /*image_width=*/input_width));
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));

    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                /*dataType=*/CUDNN_DATA_FLOAT,
                                /*format=*/CUDNN_TENSOR_NCHW,
                                /*out_channels=*/output_channels,
                                /*in_channels=*/input_channels,
                                /*kernel_height=*/kernel_height,
                                /*kernel_width=*/kernel_width));

    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));

    if(pad == SAME){
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    /*pad_height=*/kernel_height/2,
                                    /*pad_width=*/kernel_width/2,
                                    /*vertical_stride=*/1,
                                    /*horizontal_stride=*/1,
                                    /*dilation_height=*/1,
                                    /*dilation_width=*/1,
                                    /*mode=*/CUDNN_CROSS_CORRELATION,
                                    /*computeType=*/CUDNN_DATA_FLOAT));
    }
    else if(pad == VALID)
    {
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                    /*pad_height=*/0,
                                    /*pad_width=*/0,
                                    /*vertical_stride=*/1,
                                    /*horizontal_stride=*/1,
                                    /*dilation_height=*/1,
                                    /*dilation_width=*/1,
                                    /*mode=*/CUDNN_CROSS_CORRELATION,
                                    /*computeType=*/CUDNN_DATA_FLOAT));
    }

    obatch_size= 0, ochannels= 0, oheight = 0, owidth = 0;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                        input_descriptor,
                                        kernel_descriptor,
                                        &obatch_size,
                                        &ochannels,
                                        &oheight,
                                        &owidth));

    //std::cerr << "Output Image: " << obatch_size << " x "<< oheight << " x " << owidth << " x " << ochannels
    //           << std::endl;



    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/obatch_size,
                            /*channels=*/ochannels,
                            /*image_height=*/oheight,
                            /*image_width=*/owidth));

    checkCUDNN(
    cudnnGetConvolutionForwardAlgorithm(cudnn,
                                input_descriptor,
                                kernel_descriptor,
                                convolution_descriptor,
                                output_descriptor,
                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                /*memoryLimitInBytes=*/0,
                                &convolution_algorithm));

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                            input_descriptor,
                                            kernel_descriptor,
                                            convolution_descriptor,
                                            output_descriptor,
                                            convolution_algorithm,
                                            &forward_workspace_bytes));


    //std::cerr << "Forward Workspace Size: " << (forward_workspace_bytes / 1048576.0) << "MB"
    //          << std::endl;



    //Filter Backward Algorithm and Workspace Size

    size_t temp;
    backward_workspace_bytes=0;



    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
    handle, input_descriptor, output_descriptor, convolution_descriptor, kernel_descriptor,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &filter_algo));


    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
    handle, input_descriptor, output_descriptor, convolution_descriptor, kernel_descriptor,
    filter_algo, &temp));



    backward_workspace_bytes = std::max(temp,backward_workspace_bytes);


    //Data Backward Algorithm and workspace size

    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
    handle, kernel_descriptor, output_descriptor, convolution_descriptor, input_descriptor,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &data_algo));

    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
    handle, kernel_descriptor, output_descriptor, convolution_descriptor, input_descriptor,
    data_algo, &temp));

    backward_workspace_bytes = std::max(temp,backward_workspace_bytes);
    //std::cerr << "Backward Workspace Size: " << (backward_workspace_bytes / 1048576.0) << "MB"
    //          << std::endl;
}


int ConvLayer::get_output_shape_and_bytes(int shape[])
{
    //Get Output Shape in NHWC format
    shape[0] = obatch_size;
    shape[1] = oheight;
    shape[2] = owidth;
    shape[3] = ochannels;
    return shape[0]*shape[1]*shape[2]*shape[3]*sizeof(float);
}


int ConvLayer::get_input_shape_and_bytes(int shape[])
  {
    //Get Output Shape in NHWC format
    shape[0] = ibatch_size;
    shape[1] = iheight;
    shape[2] = iwidth;
    shape[3] = ichannels;
    return shape[0]*shape[1]*shape[2]*shape[3]*sizeof(float);
  }

size_t ConvLayer::get_forward_workspace_bytes()
  {
    return forward_workspace_bytes;
  }

size_t ConvLayer::get_backward_workspace_bytes()
  {
    return backward_workspace_bytes;
  }

size_t ConvLayer::get_total_workspace_size()
  {
    return std::max(forward_workspace_bytes,backward_workspace_bytes);
  }

void ConvLayer::forward(float alpha, float beta, float* d_input, float* d_kernel, void* d_workspace, float * d_output)
  {
    checkCUDNN(cudnnConvolutionForward(handle,
                                       &alpha,
                                       input_descriptor,
                                       d_input,
                                       kernel_descriptor,
                                       d_kernel,
                                       convolution_descriptor,
                                       convolution_algorithm,
                                       d_workspace,
                                       forward_workspace_bytes,
                                       &beta,
                                       output_descriptor,
                                       d_output));
  }

int ConvLayer::allocate_internal_mem(float **d_kernel, void **d_workspace,float **d_diffkernel)
  {
      int param_size = sizeof(float)*ikernel_width*ikernel_height*ichannels*ochannels;
      int workspace_size = get_total_workspace_size();
      cudaMalloc(d_kernel, param_size);
      cudaMalloc(d_workspace,workspace_size);
      cudaMalloc(d_diffkernel,param_size);

      return param_size+workspace_size;

  }

void ConvLayer::populate_filter_params(float *d_kernel)
{
  float* init_params = (float*) malloc(ochannels*ikernel_height*ikernel_width*ichannels*sizeof(float));
  std::normal_distribution<float> distribution(MU,SIGMA);
  std::default_random_engine generator;

  int dim1 = ikernel_width*ichannels;
  int dim2 = ikernel_height*dim1;
  
  for(int ochannel = 0; ochannel < ochannels; ochannel++)
    for(int row=0;row<ikernel_height;row++)
      for(int col=0;col<ikernel_width;col++)
        for(int ichannel=0;ichannel < ichannels; ichannel++)
          init_params[ochannel*dim2 + row*dim1 + col*ichannels + ichannel] = distribution(generator);


  cudaMemcpy(d_kernel,init_params,sizeof(init_params),cudaMemcpyHostToDevice);

}

ConvLayer::~ConvLayer()
{
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
}