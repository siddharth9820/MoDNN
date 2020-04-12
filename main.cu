#include "layers/layers.h"
#include <fstream>
#include <math.h>
#include "mnist_dataset/mnist.h"
#include "data_core/data_loader.h"
#include "trainer/trainer.h"

using namespace layers;
using namespace network;




int main(int argc, const char* argv[])
{
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    std::ofstream outdata;





    // cudaSetDevice(0);

    std::string images_file_str = "/content/src/mnist_dataset/data/train-images-idx3-ubyte";
    std::string label_file_str = "/content/src/mnist_dataset/data/train-labels-idx1-ubyte";
    char * images_file = (char*)images_file_str.c_str();
    char * label_file = (char*)label_file_str.c_str();
    std::cout << images_file << " "<<label_file << std::endl;
    float* data_batch, *label_batch;
    unsigned batch_size = 32,rows, sub_batch_size;
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
    std::vector<std::string> specs = {input_spec,"conv 3 3 3","relu","maxpool 2 2 2 2","flatten","fc 50","relu","fc "+std::to_string(dataset->getLabelDim()),"softmax"};
    seqNetwork * nn = new seqNetwork(cudnn,cublas,specs,LR,500000);

    vmm * mem_manager = new vmm(500000);



    train_with_minimal_memory(dataloader,dataset,nn, mem_manager,4);



    return 0;

}
