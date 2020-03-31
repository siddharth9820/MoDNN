#ifndef LAYER_H_
#define LAYER_H_

#include <cudnn.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <map>
#include <cublas_v2.h>
#include <cassert>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <random>
#include <cuda.h>
#include <fstream>

enum padding_type{
  SAME,
  VALID
};


#define MU 0
#define SIGMA 0.1
#define LR 0.01
#define TILE_SIZE  32
#define BLOCK_SIZE 8


#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

const char *cublasGetErrorString(cublasStatus_t error);

#define checkCUBLAS(expression)                              \
  {                                                          \
    cublasStatus_t status = (expression);                     \
    if (status != CUBLAS_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cublasGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


std::map<std::string,float*> init_buffer_map();
__global__ void SoftmaxLossBackprop(const int *label, int num_labels, int batch_size, float *diff);
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
                                    int numARows, int numAColumns,
                                    int numBRows, int numBColumns,
                                    int numCRows, int numCColumns);
__global__ void transposeCoalesced(float *odata, const float *idata,int idata_rows,int idata_cols);
__global__ void matrixMultiplyNaive(float * A, float * B, float * C,
                                    int N,int K,int M);
__global__ void transposeNaive(float *odata, const float *idata,int idata_rows,int idata_cols);
__global__ void update(float * weights, float * grad,float lr,int N);

int calc_bytes_from_shape(int shape[]);

namespace layers
{

  class Layer
  {
    public:
      int obatch_size,ochannels,oheight,owidth;
      int ibatch_size,ichannels,iheight,iwidth;
      void forward();

  };

}

namespace network
{
  class seqNetwork
  {
    public:
      int num_layers;
      float lr;
      std::vector<std::vector<std::string > > layer_info;
      std::vector<std::map<std::string,float*> > layer_buffers;
      std::vector<std::map<std::string,float*> > layer_offloaded_buffers;
      std::vector< layers::Layer *> layer_objects;


      cudnnHandle_t handle;
      cublasHandle_t blas_handle;


      seqNetwork(cudnnHandle_t cudnn,cublasHandle_t cublas,std::vector<std::string> &specs, float lr);
      void print_network_info();
      void allocate_memory();
      void get_output_shape(int shape[], int i);
      void randomise_batch(); //randomise input to the neural network
      void enqueue_batch(float * batch);
      void randomise_params();
      void forward();
      void backward();



      float* offload_buffer(int layer_number,std::string type,int shape[]); //type is one of "output","workspace","input"
      void prefetch_buffer(int layer_number,std::string type);
      ~seqNetwork();

  };
}

#endif