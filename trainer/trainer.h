#ifndef TRAINER_H_
#define TRAINER_H_

#define PRINT_EVERY 5

#include "../layers/layers.h"
#include "../mnist_dataset/mnist.h"
#include "../data_core/data_loader.h"
#include <math.h>

using namespace layers;
using namespace network;

void print_output(float * layer,int shape[]);
float categorical_cross_entropy_loss(float * softmax_dinput,int shape[], int * label_batch_integer);
void label_batch_converter_mnist(float* batch, int* batch_target, unsigned batch_size);
void train_with_full_memory(DataLoader * dataloader,Dataset * dataset,seqNetwork * nn, vmm * mem_manager, int epochs);
void train_with_minimal_memory(DataLoader * dataloader,Dataset * dataset,seqNetwork * nn, vmm * mem_manager, int epochs);


#endif
