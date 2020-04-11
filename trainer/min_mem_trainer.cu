#include "trainer.h"

void train_with_minimal_memory(DataLoader * dataloader,Dataset * dataset,seqNetwork * nn, vmm * mem_manager, int epochs)
{
  int dataset_size = dataset->getDatasetSize(),batch_size = 32;
  float* data_batch, *label_batch;
  int* label_batch_integer = (int*)malloc(sizeof(int)*batch_size);
  float * output,loss;
  int shape[4];
  int sub_batch_size = nn->sub_batch_size();
  int offset = ((batch_size/sub_batch_size)-1)*sub_batch_size;

  //nn->allocate_all_memory(mem_manager);
  mem_manager->printNodes();
  nn->print_network_info();


  std::cout << "Allocating Params Memory" << std::endl;
  nn->allocate_mem_params(mem_manager);
  mem_manager->printNodes();

  std::cout << "Randomising Parameters of the neural network" << std::endl;
  nn->randomise_params();


  for(int i=0;i<nn->num_layers;i++){
    std::cout << "Allocating " << nn->layer_info[i][0] << " - Output Memory" << std::endl;
    nn->allocate_mem_layer_fw(i,mem_manager);
    mem_manager->printNodes();
    nn->link_layer_buffer_fw(i);
    std::cout << nn->layer_buffers[i]["input"] <<" " << nn->layer_buffers[i]["output"] << std::endl;
  }



  for(int i=nn->num_layers-1;i>=1;i--){
    std::cout << "Allocating " << nn->layer_info[i][0] << " - dinput Memory" << std::endl;
    nn->allocate_mem_layer_bw(i,mem_manager);
    mem_manager->printNodes();
    nn->link_layer_buffer_bw(i);
    std::cout << nn->layer_buffers[i]["doutput"] <<" " << nn->layer_buffers[i]["dinput"] << std::endl;
  }

  int num_iters_in_epoch =  dataset_size/batch_size;


  for(int epoch_no=0;epoch_no<epochs;epoch_no++)
  {
    loss = 0;
    for(int i=0;i<num_iters_in_epoch;i++)
    {
      //mem_manager->printNodes();
      dataloader->get_next_batch(&data_batch, &label_batch);
      label_batch_converter_mnist(label_batch, label_batch_integer, batch_size);
      nn->update_batch(data_batch, label_batch_integer);
      //std::cout << " " << std::endl;
      nn->train();
      nn->update_weights();
      //mem_manager->printNodes();
      output = nn->offload_buffer(nn->num_layers-1,"output",shape);
      loss += categorical_cross_entropy_loss(output,shape,label_batch_integer+offset);

    }
    std::cout << "Loss - " << loss/dataset_size << std::endl;
  }
}
