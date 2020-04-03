#ifndef MNIST_H_
#define MNIST_H_

#include "../data_core/dataset.h"
#include <vector>

class MNIST : public Dataset {
    private:
        std::vector<std::vector<float> > images;
        std::vector<float> labels;
        void parse_labels_file(char* labels_file);
        void parse_images_file(char* images_file);
        unsigned seed;
    public:
        MNIST(char* images_file, char* labels_file, bool shuffle=false);
        size_t getInputDim();
        size_t getLabelDim();
        size_t getDatasetSize();
        void get_item(int index, float* data, float* label);
        void get_item_range(int start, int end, float* data_batch, float* label_batch);
};

#endif