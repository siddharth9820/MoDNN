#ifndef TRAINER_H_
#define TRAINER_H_

#define PRINT_EVERY 1

#include "../layers/layers.h"
#include "../mnist_dataset/mnist.h"
#include "../data_core/data_loader.h"
#include <math.h>

using namespace layers;
using namespace network;

/**
 * Prints the soutput of a layer.
 */
void print_output(float * layer,int shape[]);


/**
 * Categorial Cross entropy loss.
 * @param softmax_dinput Output of softmax function.
 * @param shape Shape of softmax output.
 * @param label_batch_integer Ground truth labels.
 */
float categorical_cross_entropy_loss(float * softmax_dinput,int shape[], int * label_batch_integer);


/**
 * Utility function to convert floating point labels array to integer labels array.
 */
void label_batch_converter_mnist(float* batch, int* batch_target, unsigned batch_size);


/**
 * Full memory Trainer function.
 * Used to train using full network memory.
 * @param dataloader DataLoader object pointer for loading data.
 * @param dataset Dataset object pointer.
 * @param nn Neural network pointer.
 * @param mem_manager Virtual Memory Manager pointer.
 * @param epochs No of epochs to train.
 */
void train_with_full_memory(DataLoader * dataloader,Dataset * dataset,seqNetwork * nn, vmm * mem_manager, int epochs);


/**
 * Minimum Memory Trainer function.
 * Used to train using minimum network memory.
 * @param dataloader DataLoader object pointer for loading data.
 * @param dataset Dataset object pointer.
 * @param nn Neural network pointer.
 * @param mem_manager Virtual Memory Manager pointer.
 * @param epochs No of epochs to train.
 */
void train_with_minimal_memory(DataLoader * dataloader,Dataset * dataset,seqNetwork * nn, vmm * mem_manager, int epochs);

/**
 * Heuristic Prefectching Trainer function.
 * Used to train using partial memory with prefetching heuristic.
 * @param dataloader DataLoader object pointer for loading data.
 * @param dataset Dataset object pointer.
 * @param nn Neural network pointer.
 * @param mem_manager Virtual Memory Manager pointer.
 * @param epochs No of epochs to train.
 */
void train_with_prefetching_half_window(DataLoader * dataloader,Dataset * dataset,seqNetwork * nn, vmm * mem_manager, int epochs);

/**
 * Heuristic Prefectching Trainer function.
 * Used to train using partial memory with prefetching next layer.
 * @param dataloader DataLoader object pointer for loading data.
 * @param dataset Dataset object pointer.
 * @param nn Neural network pointer.
 * @param mem_manager Virtual Memory Manager pointer.
 * @param epochs No of epochs to train.
 */
void train_with_prefetching_next(DataLoader * dataloader,Dataset * dataset,seqNetwork * nn, vmm * mem_manager, int epochs);

/**
 * Ondemand offloading Trainer function.
 * Used to train using on demand offloading scheme.
 * @param dataloader DataLoader object pointer for loading data.
 * @param dataset Dataset object pointer.
 * @param nn Neural network pointer.
 * @param mem_manager Virtual Memory Manager pointer.
 * @param epochs No of epochs to train.
 */
void offload_when_needed(DataLoader * dataloader,Dataset * dataset,seqNetwork * nn, vmm * mem_manager, int epochs);

#endif
