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

#ifndef LAYER_H_
#define LAYER_H_

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


std::map<std::string,float*> init_buffer_map();


namespace layers
{

  class Layer
  {
    public:
      int obatch_size,ochannels,oheight,owidth;
      int ibatch_size,ichannels,iheight,iwidth;
      void forward();

  };

  class ConvLayer : public Layer
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
      int ikernel_width,ikernel_height;



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
    int get_input_shape_and_bytes(int shape[]);
    size_t get_forward_workspace_bytes();
    size_t get_backward_workspace_bytes();
    size_t get_total_workspace_size();
    void forward(float alpha, float beta, float* d_input, float* d_kernel, void* d_workspace, float * d_output);
    int allocate_internal_mem(float **d_kernel, void **d_workspace);
    void populate_filter_params(float *d_kernel);
    int get_output_shape_and_bytes(int shape[]);

    ~ConvLayer();

  };

 class FCLayer : public Layer
 {
  public:
    cublasHandle_t handle;
    FCLayer(cublasHandle_t cublas,int batch_size,int input_height,int output_height);
    int get_output_shape_and_bytes(int shape[]);
    void forward(float* d_input, float * d_kernel, float * d_output);
    int allocate_internal_mem(float **d_kernel);
    void populate_filter_params(float *d_kernel);

 };
 class Flatten : public Layer
 {
  public:
   Flatten(int batch_size,int input_height,int input_width,int input_channels);
   int get_output_shape_and_bytes(int shape[]);
 };
 class Softmax : public Layer
 {

  public:
    cudnnHandle_t handle;
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;


    Softmax(cudnnHandle_t cudnn,int batch_size,int input_height);
    int get_output_shape_and_bytes(int shape[]);
    void forward(float* d_input, float * d_output);


 };
}

namespace network
{
  class seqNetwork
  {
    public:
      int num_layers;
      std::vector<std::vector<std::string > > layer_info;
      std::vector<std::map<std::string,float*> > layer_buffers;
      std::vector<std::map<std::string,float*> > layer_offloaded_buffers;

      std::vector< layers::Layer *> layer_objects;
      cudnnHandle_t handle;
      cublasHandle_t blas_handle;


      seqNetwork(cudnnHandle_t cudnn,cublasHandle_t cublas,std::vector<std::string> &specs);
      void print_network_info();
      void allocate_memory();
      void get_output_shape(int shape[], int i);
      void randomise_input();
      void randomise_params();
      void forward();
      void offload_buffer(int layer_number,std::string type); //type is one of "output","workspace","input"
      void prefetch_buffer(int layer_number,std::string type);
  };
}

#endif