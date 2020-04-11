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



    std::vector<std::string> specs = {input_spec,"flatten","fc "+std::to_string(dataset->getLabelDim()),"softmax"};
    seqNetwork * nn = new seqNetwork(cudnn,cublas,specs,LR,500000);

    vmm * mem_manager = new vmm(nn->get_total_memory()+20);



    //train_with_full_memory(dataloader,dataset,nn, mem_manager,1);
    train_with_full_memory(dataloader,dataset,nn,mem_manager,10);
    // nn.allocate_all_memory(mem_manager);
    //
    // mem_manager->printNodes();
    //
    // nn.print_network_info();
    // int shape[4];
    // nn.get_output_shape(shape,nn.num_layers-1);
    //
    // std::cout << "Printing output shape of Neural Network" << std::endl;
    // for(int i=0;i<4;i++)
    //   std::cout << shape[i] <<" "<<" ";
    // std::cout<<std::endl;
    //
    //
    // std::cout << "Randomising Parameters of the neural network" << std::endl;
    // nn.randomise_params();
    //
    //
    // float * output,loss;
    //
    // int epochs=50;
    // int num_iters_in_epoch =  dataset_size/batch_size;
    // bool rem = false;
    //
    //
    // if(dataset_size%batch_size!=0){
    //   num_iters_in_epoch+=1;
    //   rem = true;
    //   }
    // if(rem)
    //   std::cout << "Ignoring last batch " << std::endl;
    //
    // std::cout << "Number of iterations in an epoch " << num_iters_in_epoch << std::endl;
    // for(int j=0;j<epochs;j++)
    // {
    //   loss = 0;
    //   for(int i=0;i<num_iters_in_epoch;i++)
    //   {
    //     if(rem && i==num_iters_in_epoch-1)
    //       break;
    //
    //     dataloader->get_next_batch(&data_batch, &label_batch);
    //     label_batch_converter_mnist(label_batch, label_batch_integer, batch_size);
    //
    //
    //     nn.update_batch(data_batch, label_batch_integer);
    //     nn.forward();
    //     nn.backward();
    //     nn.update_weights();
    //     if(j%10==0)
    //     {
    //       output = nn.offload_buffer(nn.num_layers-1,"output",shape);
    //
    //       loss += categorical_cross_entropy_loss(output,shape,label_batch_integer);
    //     }
    //
    //   }
    //
    //   if(j%10==0)
    //   {
    //     loss = loss/(float)(dataset_size);
    //     std::cout << "Epoch number "<<j+1<< " : " << "Loss :- " << loss <<std::endl;
    //   }
    //   dataloader->reset();
    //   dataset->shuffle();
    // }


    return 0;

}
