#include "fc_layer.h"

using namespace layers;

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


