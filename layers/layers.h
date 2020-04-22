#ifndef LAYER_H_
#define LAYER_H_

#include "../vmm/vmm.h"
#include <cudnn.h>
#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <sstream>
#include <map>
#include <cublas_v2.h>
#include <cassert>
#include <cstdlib>
#include <random>
#include <cuda.h>
#include <fstream>

/*! \enum padding_type
    \brief Type of padding.
    SAME - for having same input and output dimensions,
    VALID - No padding
*/
enum padding_type{
  SAME,
  VALID
};


#define MU 0
#define SIGMA 0.1
#define LR 0.0001
#define BATCH_SIZE 128
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
      fprintf(stdout,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


std::map<std::string,float*> init_buffer_map();

/**
 * Softmax backpropagation kernel.
 * @param label Ground truth label.
 */
__global__ void SoftmaxLossBackprop(const int *label, int num_labels, int batch_size, float *diff);

/**
 * Matrix Multiplication kenel using shared memory and tiling.
 */
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
                                    int numARows, int numAColumns,
                                    int numBRows, int numBColumns,
                                    int numCRows, int numCColumns);


/**
 * Kernel to find transpose of a matrix using coalesing.
 */                 
__global__ void transposeCoalesced(float *odata, const float *idata,int idata_rows,int idata_cols);

/**
 * Naive Matrix Multiplication kernel.
 */
__global__ void matrixMultiplyNaive(float * A, float * B, float * C,
                                    int N,int K,int M);

                                    
/**
 * Kernel to find transpose of a matrix in a naive way.
 */
__global__ void transposeNaive(float *odata, const float *idata,int idata_rows,int idata_cols);

/**
 * Updates the parameters of neural network.
 * @param weights Parameters to update.
 * @param grad Gradients of parameters.
 * @param lr Learning rate.
 * @param N Num of parameters.
 */
__global__ void update(float * weights, float * grad,float lr,int N);

int calc_bytes_from_shape(int shape[]);

/*! \namespace layers
    \brief Contains all function and classes involving neural network layers.

    This namespace contains classes of Convolution layer, full connected layer, flatten layer, 
    pooling layer, activation layer, softmax layer.
*/
namespace layers
{
  /*! \class Layer
    \brief Abstract class Layer.

    This class is inherited by all the neural network layers.
  */
  class Layer
  {
    public:
      int obatch_size; /*!< Batch size of the output returned by layer. */
      int ochannels; /*!< No of channels in the output returned by layer. */
      int oheight; /*!< Height of the output returned by layer. */
      int owidth; /*!< Width of the output returned by layer. */
      int ibatch_size; /*!< Batch size of the input passed to the layer. */
      int ichannels; /*!< No of channels in the input passed to the layer. */
      int iheight; /*!< Height of the input passed to the layer. */
      int iwidth; /*!< Width of the input passed to the layer. */
      void forward(); /*!< Does Forward pass of layer. */
      virtual int get_total_memory()=0; /*!< Return the total memory occupied by the layer - activation(output) + params + workspace.  */

  };

}

/*! \namespace network
    \brief Contains all function and classes involving initialization training of neural networks.

    This namespace primarily contains the class seqNetwork which is the main class that handles the creating and training of neural network.
*/
namespace network
{
  /*! \class seqNetwork
    \brief Main neural network class.

    This class is used to create and train neural networks.
  */
  class seqNetwork
  {
    public:
      int num_layers; /*!< Number of layers in the neural network. */
      float lr; /*!< Learning rate. */
      int batch_size; /*!< Batch size of the input data. */
      std::vector<std::vector<std::string > > layer_info; /*!< Vector describing the each layer name and specifications. */
      std::vector<std::map<std::string,float*> > layer_buffers; /*!< Vector of maps that point to each layer's device memory pointers. */
      std::vector<std::map<std::string,float*> > layer_offloaded_buffers; /*!< Vector of maps that point to each layer's host memory pointers. (This will be relevant only when layer device memory is offloaded to host). */
      std::vector<std::map<std::string,int> > layer_buffer_bytes; /*!< Vector of maps that point to each layer's memory requirement(in bytes). */
      std::vector<std::map<std::string,int> > layer_buffer_redundant_bytes; /*!< Vector of maps that point to each layer's redundant memory used. */
      std::vector< layers::Layer *> layer_objects; /*!< Vector of Layer object pointers. */
      std::queue<int> locally_allocated_layers;
      std::queue<int> globally_allocated_layers;

      cudnnHandle_t handle; /*!< CUDNN handle. */
      cublasHandle_t blas_handle; /*!< CUBLAS handle */

      /**
       * Intializes seqNetwork Class.
       * @param cudnn CUDNN handle
       * @param cublas CUBLAS handle.
       * @param specs Layer-wise specification of neural network. 
       * @param lr Learning rate.
       * @param max_allowed_bytes Maximum usable GPU memory.
       * @param sub_batch_selection Set true to enable sub_batch_selection algorithm.
       */

      seqNetwork(cudnnHandle_t cudnn,cublasHandle_t cublas,std::vector<std::string> &specs, float lr, unsigned max_allowed_bytes,int sub_batch_selection);
      
      /**
       * Prints the network.
       */
      void print_network_info();

      /**
       * Gets output shape of ith layer.
       * @param shape 4-dim array storing the output shape.
       * @param i Layer number for which the output shape is to be returned.
       */
      void get_output_shape(int shape[], int i);

      
      /**
       * Randomly intialize input to the neural network.
       */
      void randomise_batch(); //randomise input to the neural network
      
      /**
       * Update the input to the neural network.
       * @param data new input data.
       * @param labels new ground truth labels to the input data.
       */
      void update_batch(float* data, int* labels);

      // void enqueue_batch(float * batch);

      /**
       * Updates the network input to the loop_no index sub_batch. 
       * Used when sub_batch selection algorithm is activated.
       * @param loop_no sub_batch number.
       */
      void enqueue_batch_loop(int loop_no);

      /**
       * Randomly intialize parameters of the neural network.
       */
      void randomise_params();

      /**
       * Iterates over all the network layers calling forward pass of each layer.
       * DEPRICATED
       */
      void forward();

      /**
       * Executes forward pass of a specific layer.
       * @param layer_number Layer number of which forward pass needs to be called. 
       */
      void forward_layer(int layer_number);

      /**
       * Iterates over all the network layers calling backward pass of each layer.
       * DEPRICATED
       */
      void backward();

      /**
       * Executes backward pass of a specific layer.
       * @param layer_number Layer number of which backward pass needs to be called. 
       */
      void backward_layer(int layer_number,float beta);

      /**
       * Executes both forward and backward pass with sub_batch training algorithm.
       */
      void train();

      /**
       * Updates network parameters. 
       * To be called after forward and backward pass so that the network gradients are set before updating parameters. 
       */
      void update_weights();

      /**
       * Returns total device memory occupied by the network.  
       * This may be different from maximum memory if sub batch algorithm is enabled.
       */
      int get_total_memory();

      /**
       * Returns the maximum memory that can be occupied by the network.
       */
      unsigned get_max_memory();

      /**
       * Returns the maximum memory that can be occupied by the network.
       */
      void allocate_all_memory(vmm * mem_manager);

      /**
       * Returns the minimum memory needed to train.
       */
      unsigned getMemoryLowerBound();

      /**
       * Returns subbatch size computed using sub_batch selection algorithm.
       */
      unsigned sub_batch_size();

      /**
       * Creates links between layer buffers during forward pass.
       */
      void link_layer_buffer_fw(int layer_number);
      
      /**
       * Creates links between layer buffers during backward pass.
       */
      void link_layer_buffer_bw(int layer_number);

      /**
       * Returns the no of sub_batchs found after using sub_batch selection algorithm.
       */
      int get_loops();

      /**
       * Returns maximum sub_batch batch size.
       * Same as the batch_size mentioned in the network specification.
       */
      int get_max_batch_size();

      /**
       * Offloads of a specific buffer of the metioned layer.
       * @param layer_number Index of the layer to which the buffer belongs to.
       * @param type Buffer key to choose from the set of buffers belonging to the mentioned layer. (type is one of "output","workspace","input").
       * @param shape Shape of the the buffer
       * @param async Set true to offload the buffer asynchronously.
       */
      float* offload_buffer(int layer_number,std::string type,int shape[],int async=1); 

      /**
       * Prefectches a specific buffer of the metioned layer.
       * @param layer_number Index of the layer to which the buffer belongs to.
       * @param type Buffer key to choose from the set of buffers belonging to the mentioned layer. (type is one of "output","workspace","input").
       * @param shape Shape of the the buffer
       */
      float* prefetch_buffer(int layer_number, std::string type,int shape[]);

      /**
       * Allocate memory for a specific layer(layer buffers) on device for forward pass.
       * @param layer_number Layer index.
       * @param mem_manager Pointer to Virtual memory manager for assigning device memory. 
       */
      void allocate_mem_layer_fw(int layer_number, vmm * mem_manager);
      
      /**
       * Allocate memory for a specific layer((layer buffers)) on device for backward pass.
       * @param layer_number Layer index.
       * @param mem_manager Pointer to Virtual memory manager for assigning device memory. 
       */
      void allocate_mem_layer_bw(int layer_number, vmm * mem_manager);
   
      /**
       * Allocate memory for a specific layer((layer buffers)) on device for backward pass with a prefectching heuristic.
       * @param layer_number Layer index.
       * @param mem_manager Pointer to Virtual memory manager for assigning device memory. 
       */
      void allocate_mem_layer_bw_h1(int layer_number, vmm * mem_manager);

      /**
       * Deallocate memory for a specific layer((layer buffers)) on device for forward pass.
       * @param layer_number Layer index.
       * @param mem_manager Pointer to Virtual memory manager for assigning device memory. 
       */
      void deallocate_mem_layer_fw(int layer_number, vmm * mem_manager,int local=0);

      /**
       * Deallocate memory for a specific layer((layer buffers)) on device for backward pass.
       * @param layer_number Layer index.
       * @param mem_manager Pointer to Virtual memory manager for assigning device memory. 
       */
      void deallocate_mem_layer_bw(int layer_number, vmm * mem_manager,int local=0);


      void offload_and_call_mem_manager(float ** buff, int bytes, std::string misc, vmm * mem_manager,int layer_number,int offload);
      void deallocate_mem_layer_fw2(int layer_number, vmm * mem_manager,int local,int offload);
      void allocate_mem_layer_fw2(int layer_number, vmm * mem_manager);
      void allocate_mem_layer_bw2(int layer_number, vmm * mem_manager);
      /**
       * Allocate memory for a neural network params(conv, fc params).
       * @param mem_manager Pointer to Virtual memory manager for assigning device memory. 
       */
      void allocate_mem_params(vmm * mem_manager);

      /**
       * Destroy seqNetwork.
       */
      ~seqNetwork();

    private:
      /**
       * Iterates over all the network layers calling forward pass of each layer.
       */
      void forward_();

      /**
       * Iterates over all the network layers calling backward pass of each layer.
       * @param beta Set to 1 to add the new gradients to the old gradients. To overwrite set to 0.
       */
      void backward_(float beta);

      /**
       * Creates instances of each layer parsing the specifications and computes the number of bytes needed for each layer buffer.
       */
      void make_nn_objs(unsigned sub_batch_size);
      void link_all_buffers();

      /**
       * Subbatch selection algorithm
       */
      unsigned calculate_sub_batch();

      /**
       * Computes total memory occupied by the network. 
       * Should be called after calling make_nn_objs.
       */
      int get_total_memory_();

      /**
       * Computes the minimum memory needed to train the network.
       */
      unsigned getMemoryLowerBound_();

      /**
       * Helper function for sub batch selection algorithm.
       * Checks if a subbatch is valid and the network occupies memory less than max_allowed_bytes_.
       */
      bool profile_subbatch_validity(unsigned batch_size);

      unsigned max_sub_batch_size_; /*!< Maximum sub batch size. Same as the batch size mentioned in specification. */
      cudaStream_t memory_stream_; /*!< Cuda stream for memory operations.(Is used for Async memory copy) */
      cudaStream_t compute_stream_; /*!< Cuda stream for compute operations. */
      unsigned sub_batch_size_; /*!< Computed subbatch size using subbatch selection algorithm. */
      unsigned max_allowed_bytes_; /*!< Maximum available memory for neural network. */
      unsigned weights_memory_bytes_; /*!< Memory occupied by network params. */
      unsigned total_seqnet_bytes_; /*!< Memory occupied by total network. Different from max_seqnet_memory_ if subbatch selection algorithm is used.  */
      unsigned min_seqnet_bytes_; /*!< Minimum memory required to train the network. */
      unsigned max_seqnet_memory_; /*!< Maximum memory the network can occupy. Computed using max_sub_batch_size_. */
      float* batch_data_; /*!< Pointer to input batch for neural network. */
      int* batch_labels_; /*!< Pointer to ground truth labels for the input batch. */
      int sync_layer_no_; /*!< #Only for Prefecting heuristic# Layer number at which CudaDeviceSynchronization should be called. */
      int prefetch_trigger_layer_no_; /*!< #Only for Prefecting heuristic# Layer number at which prefetching should be triggered. */
      int last_prefetched_layer_no_; /*!< #Only for Prefecting heuristic# Gives the layer number till which prefectching is already submitted in memory stream. */
  };
}

#endif
