#ifndef DATASET_H_
#define DATASET_H_

#include <cstdlib>

/*! \class Dataset
 \brief Abstract class Dataset.

 This class is inherited by the datasets used (mnist_dataset for now).
*/
class Dataset{
    protected:
        size_t dataset_size_;       /*!< stores the number of elements in dataset. */
        size_t input_size_;         /*!< stores the size of a data item without label. */
        size_t label_size_;         /*!< stores the size of label. */
    public:

        /**
        * returns the size of a data item. 
        */
        virtual size_t getInputDim() = 0;       

        /**
        * returns the size of a label.
        */
        virtual size_t getLabelDim() = 0;       
        
        /**
        * returns the number of elements in the dataset. 
        */
        virtual size_t getDatasetSize() = 0;    
        
        /**
        * Returns the data item and label present at index.
        * @param index index of the data item and label to fetch .
        * @param input Pointer to the buffer for storing data.
        * @param output Pointer to the buffer for storing label.
        */
        virtual void get_item(int index, float* input, float* output) = 0;                  

        /**
        * Returns all the data items and labels in the range from index start variable to index end variable.
        * @param start start index of the range.
        * @param end end index of the range.
        * @param input Pointer to the buffer for storing data/
        * @param output Pointer to the buffer for storing labels.
        */ 
        virtual void get_item_range(int start, int end, float* input, float* output) = 0;    
        
        /**
        * randomly shuffles the dataset.
        */
        virtual void shuffle()=0;               
};

#endif
