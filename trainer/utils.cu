#include "trainer.h"

float categorical_cross_entropy_loss(float * softmax_dinput,int shape[], int * label_batch_integer)
{
  float temp,loss=0;
  for(int i=0;i<shape[0];i++){
      int j = label_batch_integer[i];
      temp = softmax_dinput[i*shape[1]+j];
      loss += -log(temp);

  }
  return loss;
}

void label_batch_converter_mnist(float* batch, int* batch_target, unsigned batch_size)
{
  for (int i = 0; i < batch_size; i++)
  {
    batch_target[i] = int(batch[i]);
  }
}

void print_output(float * layer,int shape[])
{
  for(int i=0;i<shape[0];i++){
    for(int j=0;j<shape[1]*shape[2]*shape[3];j++){
        std::cout << layer[i*shape[1]*shape[2]*shape[3]+j] << " ";
    }
    std::cout << std::endl;
  }

}
