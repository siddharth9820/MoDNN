#ifndef DATA_LOADER_H_
#define DATA_LOADER_H_

#include "dataset.h"

/*! \class Dataset

This class is used to load the data and store it in the buffer.
*/
class DataLoader {
    private:
        unsigned batch_size_;       /*!< number of data items to be read together. */
        unsigned index_;            /*!< current index of the data in the dataset to be read. */
        unsigned max_index_;        /*!< maximum number of elements in the dataset. */
        Dataset* dataset_;          /*!< pointer to the dataset and its methods. */
        float* data_buffer_;        /*!< stores a data item . */
        float* label_buffer_;       /*!< stores the label for the data_buffer_. */
    public:

        /**
        * Initializes DataLoader class.
        * @param dataset Dataset to initialize to.
        * @param batch_size number of data items to access in a single batch.
        */    
        DataLoader(Dataset* dataset, unsigned batch_size);      

        /**
        * Returns a batch of data items and their labels in data and labels respectively.
        * @param data stores the data items in it for returning.
        * @param labels stores the labels in it for returning.
        */
        unsigned get_next_batch(float** data, float** labels);

        /**
        * Returns the batch size  
        */
        unsigned getBatchSize();                                
        
        /**
        * resets the index to the first element in the dataset
        */
        void reset();                                           
        
        /**
        * Destroy DataLoader
        */
        ~DataLoader();
};

#endif