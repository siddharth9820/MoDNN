#ifndef MNIST_H_
#define MNIST_H_

#include "../data_core/dataset.h"
#include <vector>

/*! \class MNIST

This class is used to read the mnist_dataset.
*/
class MNIST : public Dataset {
    private:
        std::vector<std::vector<float> > images;    /*!< stores the images. */
        std::vector<float> labels;                  /*!< stores the labels. */

        /**
        * reads the labels files from the mnist dataset.
        * @param labels_file labels to be read
        */
        void parse_labels_file(char* labels_file);  

        /*
        * reads the image files from the mnist dataset.
        * @param images_file images to be read
        */
        void parse_images_file(char* images_file);  

        /**
        * randomly shuffles the dataset.
        */
        void shuffle();                             
        
        unsigned seed;                              /*!< seed used for shuffling the dataset. */
    
    public:
        /** 
        * reads and stores the dataset.
        * @param images_file image files to be read
        * @param labels_file labels to be read
        * @param shuffle to shuffle the dataset or not
        */
        MNIST(char* images_file, char* labels_file, bool shuffle=false);   

        /** 
        * returns the size of an image.
        */
        size_t getInputDim();                                              

        /** 
        * returns the dimension of the label.
        */
        size_t getLabelDim();                                              

        /** 
        * returns the number of elements in the dataset.
        */
        size_t getDatasetSize();                                           

        /** 
        * returns the data item and label present at index in pointers data and label respectively.
        * @param index index of the dataitem to be accessed
        */
        void get_item(int index, float* data, float* label);               

        /** 
        * returns the data items and labels in the range from index start variable to index end variable in pointers data_batch and label_batch respectively.        
        * @param start start index of the range
        * @param end end index of the range
        */
        void get_item_range(int start, int end, float* data_batch, float* label_batch);     

};

#endif
