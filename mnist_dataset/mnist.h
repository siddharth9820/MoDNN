#ifndef MNIST_H_
#define MNIST_H_

#include "../data_core/dataset.h"
#include <vector>

class MNIST : public Dataset {
    private:
        std::vector<std::vector<float> > images;    // stores the images
        std::vector<float> labels;                  // stores the labels
        void parse_labels_file(char* labels_file);  // reads the labels files from the mnist dataset
        void parse_images_file(char* images_file);  // reads the image files from the mnist dataset
        void shuffle();                             // randomly shuffles the dataset
        unsigned seed;                              // seed used for shuffling the dataset
    public:
        MNIST(char* images_file, char* labels_file, bool shuffle=false);   // reads and stores the dataset 
        size_t getInputDim();                                              // returns the size of an image
        size_t getLabelDim();                                              // returns the dimension of the label
        size_t getDatasetSize();                                           // returns the number of elements in the dataset
        void get_item(int index, float* data, float* label);               // returns the data item and label present at index in pointers data and label respectively
        void get_item_range(int start, int end, float* data_batch, float* label_batch);     // returns the data items and labels from index start variable to index end variable in pointers data_batch and label_batch respectively
};

#endif
