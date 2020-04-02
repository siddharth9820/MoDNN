#ifndef DATA_LOADER_H_
#define DATA_LOADER_H_

#include "dataset.h"

class DataLoader {
    private:
        unsigned batch_size_;
        unsigned index_;
        unsigned max_index_;
        Dataset* dataset_;
    public:
        DataLoader(Dataset* dataset, unsigned batch_size);
        unsigned get_next_batch(float* data, float* labels); // Returns the batch size
        unsigned getBatchSize();
};

#endif