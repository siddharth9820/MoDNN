#include "layers/layers.h"
#include <fstream>
#include <math.h>
#include "mnist_dataset/mnist.h"
#include "data_core/data_loader.h"
#include "trainer/trainer.h"
#include <cuda_profiler_api.h>

using namespace layers;
using namespace network;




int main(int argc, const char* argv[])
{
    // main function for executing the code
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    std::ofstream outdata;





    // cudaSetDevice(0);

    std::string images_file_str = "mnist_dataset/data/train-images.idx3-ubyte";
    std::string label_file_str = "mnist_dataset/data/train-labels.idx1-ubyte";
    char * images_file = (char*)images_file_str.c_str();
    char * label_file = (char*)label_file_str.c_str();
    std::cout << images_file << " "<<label_file << std::endl;
    float* data_batch, *label_batch;
    unsigned batch_size = BATCH_SIZE,rows, sub_batch_size;
    unsigned dataset_size, offset;

    std::cout << "Creating Dataset" << std::endl;
    Dataset* dataset= new MNIST(images_file, label_file, true);
    dataset_size = dataset->getDatasetSize();

    std::cout << "Creating DataLoader" << std::endl;

    DataLoader* dataloader = new DataLoader(dataset, batch_size);
    rows = sqrt(dataset->getInputDim());
    std::string input_spec = "input "  + std::to_string(batch_size)+ " " + std::to_string(rows) +" "+std::to_string(rows)+ " " + "1 " +std::to_string(dataset->getLabelDim());
    int* label_batch_integer = (int*)malloc(sizeof(int)*batch_size);



    //std::vector<std::string> specs = {input_spec,"flatten","fc "+std::to_string(dataset->getLabelDim()),"softmax"};
    //le net specification
    std::vector<std::string> specs = {input_spec,
                                      "conv 5 5 20",
                                      "relu",
                                      "maxpool 2 2 2 2",
                                      "conv 5 5 50",
                                      "relu",
                                      "maxpool 2 2 2 2",
                                      "flatten",
                                      "fc 500",
                                      "relu",
                                      "fc "+std::to_string(dataset->getLabelDim()),
                                      "softmax"};



    // int MAX_MEM = 105458240; //25MB
    // int USE_MEM = MAX_MEM; //8MB
    seqNetwork * nn = new seqNetwork(cudnn,cublas,specs,LR,0,0);
    std::cout << (float)nn->get_total_memory()/1000000 << " MB " <<std::endl;
    vmm * mem_manager = new vmm(2*nn->get_total_memory(),&(nn->layer_buffers));


    // cudaProfilerStart();
     train_with_minimal_memory(dataloader,dataset,nn, mem_manager,5);
    // train_with_full_memory(dataloader,dataset,nn,mem_manager,5);
    // cudaProfilerStop();
    return 0;

}
