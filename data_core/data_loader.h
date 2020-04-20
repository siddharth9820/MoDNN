#ifndef DATA_LOADER_H_
#define DATA_LOADER_H_

#include "dataset.h"

class DataLoader {
    private:
        unsigned batch_size_;       // number of data items to be read together
        unsigned index_;            // current index of the data in the dataset to be read
        unsigned max_index_;        // maximum number of elements in the dataset
        Dataset* dataset_;          // pointer to the dataset and its methods
        float* data_buffer_;        // stores a data item 
        float* label_buffer_;       // stores the label for the data_buffer_
    public:
        DataLoader(Dataset* dataset, unsigned batch_size);      
        unsigned get_next_batch(float** data, float** labels);  // Returns a batch of data items and their labels in data and labels respectively
        unsigned getBatchSize();                                // Returns the batch size
        void reset();                                           // resets the index to the first element in the dataset
        ~DataLoader();
};

#endif