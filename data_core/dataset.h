#ifndef DATASET_H_
#define DATASET_H_

#include <cstdlib>

class Dataset{
    protected:
        size_t dataset_size_;       // stores the number of elements in dataset
        size_t input_size_;         // stores the size of a data item without label
        size_t label_size_;         // stores the size of label
    public:
        virtual size_t getInputDim() = 0;       // returns the size of a data item
        virtual size_t getLabelDim() = 0;       // returns the size of a label
        virtual size_t getDatasetSize() = 0;    // returns the number of elements in the dataset
        virtual void get_item(int index, float* input, float* output) = 0;                   // returns the data item and label present at index in pointers data and label respectively
        virtual void get_item_range(int start, int end, float* input, float* output) = 0;    // returns the data items and labels from index start variable to index end variable in pointers data_batch and label_batch respectively   
        virtual void shuffle()=0;               // randomly shuffles the dataset
};

#endif
