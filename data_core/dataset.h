#ifndef DATASET_H_
#define DATASET_H_

#include <cstdlib>

class Dataset{
    protected:
        size_t dataset_size_;
        size_t input_size_;
        size_t label_size_;
    public:
        virtual size_t getInputDim() = 0 ;
        virtual size_t getLabelDim() = 0;
        virtual size_t getDatasetSize() = 0;
        virtual void get_item(int index, float* input, float* output) = 0;
        virtual void get_item_range(int start, int end, float* input, float* output) = 0;
};

#endif