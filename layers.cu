#include "layers.h"

using namespace layers;

std::map<std::string,float*> init_buffer_map()
{
  std::map<std::string,float*> buffer_map;
  buffer_map["input"] = nullptr;
  buffer_map["output"] = nullptr;
  buffer_map["workspace"] = nullptr;
  buffer_map["params"] = nullptr;

  buffer_map["dinput"] = nullptr;
  buffer_map["doutput"] = nullptr;
  buffer_map["dparams"] = nullptr;

  return buffer_map;
}

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}


void Layer::forward()
{

}

Flatten::Flatten(int batch_size,int input_height,int input_width,int input_channels)
{
  ibatch_size = batch_size;
  ichannels = input_channels;
  iheight = input_height;
  iwidth = input_width;
  obatch_size = batch_size;
  oheight = input_channels*input_height*input_width;

}

int Flatten::get_output_shape_and_bytes(int shape[])
{
  shape[0] = obatch_size;
  shape[1] = oheight;
  shape[2] = -1;
  shape[3] = -1;

  return obatch_size*oheight*sizeof(float);
}

int Flatten::get_input_shape_and_bytes(int shape[])
{
  shape[0] = ibatch_size;
  shape[1] = iheight;
  shape[2] = iwidth;
  shape[3] = ichannels;

  return obatch_size*oheight*sizeof(float);
}

FCLayer::FCLayer(cublasHandle_t cublas,int batch_size, int input_height, int output_height)
{
  handle = cublas;
  ibatch_size = batch_size;
  iheight = input_height;
  obatch_size = ibatch_size;
  oheight = output_height;
}

int FCLayer::get_output_shape_and_bytes(int shape[])
{
  shape[0] = obatch_size;
  shape[1] = oheight;
  shape[2] = -1;
  shape[3] = -1;

  return obatch_size*oheight*sizeof(float);
}

int FCLayer::get_input_shape_and_bytes(int shape[])
{
  shape[0] = obatch_size;
  shape[1] = iheight;
  shape[2] = -1;
  shape[3] = -1;

  return obatch_size*oheight*sizeof(float);
}

int FCLayer::get_params_shape_and_bytes(int shape[])
{
  shape[0] = iheight;
  shape[1] = oheight;
  shape[2] = -1;
  shape[3] = -1;

  return iheight*oheight*sizeof(float);
}

int FCLayer::allocate_internal_mem(float **d_kernel,float **d_diffkernel)
{
  int param_size = iheight*oheight*sizeof(float);
  cudaMalloc(d_kernel,param_size);
  cudaMalloc(d_diffkernel,param_size);
  return param_size;
}

void FCLayer::populate_filter_params(float *d_kernel)
{
  float* init_params = (float*) malloc(iheight*oheight*sizeof(float));
  std::normal_distribution<float> distribution(MU,SIGMA);
  std::default_random_engine generator;
  for(int i=0;i<iheight;i++)
    for(int j=0;j<oheight;j++)
      init_params[i*oheight+j] = distribution(generator);

  cudaMemcpy(d_kernel,init_params,iheight*oheight*sizeof(float),cudaMemcpyHostToDevice);
}

void FCLayer::forward(float * d_input, float * d_kernel, float * d_output)
{
  float alpha = 1.0,beta = 0.0;
  //we are multiplying d_input(A) * d_kernel(B), both stored in row major form
  //d_input [obatch_sizexiheight] d_okernel[iheight*oheight]
  //in comments A[MxK] B[KxN]
  checkCUBLAS(cublasSgemm(handle,
              CUBLAS_OP_N, //info for B, use CUBLAS_OP_T if you want to use BT
              CUBLAS_OP_N, //info for A, use CUBLAS_OP_T if you want to use AT
              oheight,/*N*/
              obatch_size,/*M*/
              iheight,/*K*/
              &alpha,
              d_kernel, //B
              oheight, //N
              d_input, //A
              iheight, //K
              &beta,
              d_output,//C
              oheight //K
            ));
}

void FCLayer::backward(float *d_input, float* d_kernel,float *d_diffkernel,float *d_diffinput, float *d_diffoutput)
{
  float alpha = 1.0,beta = 0.0;
  checkCUBLAS(cublasSgemm(handle,
              CUBLAS_OP_N, //info for B, use CUBLAS_OP_T if you want to use BT
              CUBLAS_OP_T, //info for A, use CUBLAS_OP_T if you want to use AT
              oheight,/*N*/
              iheight,/*M*/
              obatch_size,/*K*/
              &alpha,
              d_diffoutput, //B
              oheight, //N
              d_input, //A
              iheight, //K
              &beta,
              d_diffkernel,//C
              oheight //K
            ));

  checkCUBLAS(cublasSgemm(handle,
              CUBLAS_OP_T, //info for B, use CUBLAS_OP_T if you want to use BT
              CUBLAS_OP_N, //info for A, use CUBLAS_OP_T if you want to use AT
              iheight,/*N*/
              obatch_size,/*M*/
              oheight,/*K*/
              &alpha,
              d_kernel, //B
              oheight, //N
              d_diffoutput, //A
              oheight, //K
              &beta,
              d_diffinput,//C
              iheight //K
            ));
}



Softmax::Softmax(cudnnHandle_t cudnn,int batch_size,int input_height)
{
  handle = cudnn;
  ibatch_size = obatch_size = batch_size;
  iheight = oheight = input_height;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnCreateTensorDescriptor(&diff_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptorEx(input_descriptor,CUDNN_DATA_FLOAT,obatch_size,oheight,1,1,oheight,1,1,1));
  checkCUDNN(cudnnSetTensor4dDescriptorEx(output_descriptor,CUDNN_DATA_FLOAT,obatch_size,oheight,1,1,oheight,1,1,1));
  checkCUDNN(cudnnSetTensor4dDescriptorEx(diff_descriptor,CUDNN_DATA_FLOAT,obatch_size,oheight,1,1,oheight,1,1,1));

}

int Softmax::get_output_shape_and_bytes(int shape[])
{
  shape[0] = obatch_size;
  shape[1] = oheight;
  shape[2] = -1;
  shape[3] = -1;

  return obatch_size*oheight*sizeof(float);
}

int Softmax::get_input_shape_and_bytes(int shape[])
{
  shape[0] = obatch_size;
  shape[1] = oheight;
  shape[2] = -1;
  shape[3] = -1;

  return obatch_size*oheight*sizeof(float);
}

void Softmax::forward(float* d_input, float * d_output)
{
  float alpha = 1.0;
  float beta = 0.0;
  checkCUDNN(cudnnSoftmaxForward(handle,
                      CUDNN_SOFTMAX_ACCURATE,
                      CUDNN_SOFTMAX_MODE_CHANNEL,
                      &alpha,
                      input_descriptor,
                      d_input,
                      &beta,
                      output_descriptor,
                      d_output));

}


void Softmax::backward(const int *label, float *diff, float * output)
{
  cudaMemcpy(diff,output,obatch_size*oheight*sizeof(float),cudaMemcpyDeviceToDevice);
  SoftmaxLossBackprop<<<(obatch_size+255)/256, 256>>>(label, oheight, obatch_size, diff);
}

int calc_bytes_from_shape(int shape[])
{
  int bytes = 1;
  for(int i=0;i<4;i++)
  {
    if(shape[i]==-1)break;
    else bytes*=shape[i];
  }
  return bytes;
}
