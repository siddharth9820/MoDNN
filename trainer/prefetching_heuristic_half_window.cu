#include "trainer.h"
#define ONE_ITER false
#define PRINT_OUTPUT false

void train_with_prefetching_half_window(DataLoader * dataloader,Dataset * dataset,seqNetwork * nn, vmm * mem_manager, int epochs)
{
  int dataset_size = dataset->getDatasetSize(),batch_size = nn->get_max_batch_size();
  float* data_batch, *label_batch;
  int* label_batch_integer = (int*)malloc(sizeof(int)*batch_size);
  float * output,loss;
  int shape[4];
  int sub_batch_size = nn->sub_batch_size();
  int loops = nn->get_loops();
  if(ONE_ITER)
   loops = 1;
  //nn->allocate_all_memory(mem_manager);
  mem_manager->printNodes();
  nn->print_network_info();


  std::cout << "Allocating Params Memory" << std::endl;
  nn->allocate_mem_params(mem_manager);
  mem_manager->printNodes();

  std::cout << "Randomising Parameters of the neural network" << std::endl;
  nn->randomise_params();

  int num_iters_in_epoch = dataset_size/batch_size;
  if(ONE_ITER){
    num_iters_in_epoch = 1;
    epochs = 1;
  }
  bool rem = false;


  if(dataset_size%batch_size!=0){
    num_iters_in_epoch+=1;
    rem = true;
    }
  if(rem)
    std::cout << "Ignoring last batch " << std::endl;

  //int epoch_no = 0;
  for(int epoch_no=0;epoch_no<epochs;epoch_no++)
  {
    float epoch_loss = 0;
    for(int batch_no=0;batch_no<num_iters_in_epoch;batch_no++)
    {
      if(rem && batch_no == num_iters_in_epoch-1)
        continue;
      dataloader->get_next_batch(&data_batch, &label_batch);
      label_batch_converter_mnist(label_batch, label_batch_integer, batch_size);
      nn->update_batch(data_batch, label_batch_integer);
      float beta = 0.0;
      //int loop_no = 0;
      for(int loop_no=0;loop_no<loops;loop_no++)
      {
        //forward
        for(int i=0;i<nn->num_layers;i++)
        {
            //std::cout << "Layer "<<i << " " << nn->layer_info[i][0] << " Forward" << std::endl;
            nn->allocate_mem_layer_fw(i,mem_manager);
            nn->link_layer_buffer_fw(i);
            //mem_manager->printNodes();

            if(i==0)//enqueue_batch
            {
              nn->enqueue_batch_loop(loop_no);
            }
            else//forward pass
            {
              //call forward function
              nn->forward_layer(i);
              nn->deallocate_mem_layer_fw(i,mem_manager,1); //local deletion
              nn->deallocate_mem_layer_fw(i-1,mem_manager); //shared buffer deletion
              nn->link_layer_buffer_fw(i-1);
              if( ONE_ITER && PRINT_OUTPUT)
              {
                output = nn->offload_buffer(i,"output",shape,1);
                cudaDeviceSynchronize();
                print_output(output,shape);
              }
            }
        }
        output = nn->offload_buffer(nn->num_layers-1,"output",shape,1);
        cudaDeviceSynchronize();
        epoch_loss += categorical_cross_entropy_loss(output,shape,label_batch_integer+loop_no*sub_batch_size);
        //backward
        if(loop_no>0)beta=1.0;
        for(int i=nn->num_layers-1;i>=0;i--)
        {
          //std::cout << "Layer "<<i << " " << nn->layer_info[i][0] << " Backward" << std::endl;
          nn->allocate_mem_layer_bw_h1(i,mem_manager);
          nn->link_layer_buffer_bw(i);

          nn->backward_layer(i,beta);

          // cudaDeviceSynchronize();

          // nn->deallocate_mem_layer_bw(i,mem_manager,1);
          // if(i!=nn->num_layers-1)
          // {
          //   //deallocate layer + 1's data

          //   nn->deallocate_mem_layer_bw(i+1,mem_manager);
          //   nn->link_layer_buffer_bw(i+1);
          // }
        }

        cudaDeviceSynchronize();

        for(int i=nn->num_layers-1;i>=0;i--)
        {


          nn->deallocate_mem_layer_bw(i,mem_manager,1);
          if(i!=nn->num_layers-1)
          {
            //deallocate layer + 1's data

            nn->deallocate_mem_layer_bw(i+1,mem_manager);
            nn->link_layer_buffer_bw(i+1);
          }
        }
      }
      nn->update_weights();
    }
    std::cout << "Epoch Number - " << epoch_no << " Loss - " << epoch_loss/dataset_size << std::endl;
  }



}
