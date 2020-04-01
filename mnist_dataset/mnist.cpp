#include "mnist.h"
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring> 
#include <iostream>
#include <fstream>

uint32_t reverseBits(uint32_t value) {
    return (value << 24) | ((value << 8) & 0x00FF0000) | ((value >> 8) & 0X0000FF00) | (value >> 24);
}

MNIST::MNIST(char* images_filename, char* labels_filename, bool shuffle) {
    if (shuffle)
        seed = std::chrono::system_clock::now().time_since_epoch().count(); 

    label_size_ = 10;
    parse_images_file(images_filename);
    parse_labels_file(labels_filename);

    if (shuffle){
        std::shuffle(images.begin(), images.end(), std::default_random_engine(seed));
        std::shuffle(labels.begin(), labels.end(), std::default_random_engine(seed));
    }
}

void MNIST::parse_images_file(char* images_file) {
    std::ifstream fd(images_file, std::ios::in | std::ios::binary);
    char data[4];
    unsigned int magic_number, rows, cols, i = 0;
    char pixel;
    fd.read(data, sizeof(unsigned));
    magic_number = reverseBits(*((unsigned int*) data));
    fd.read(data, sizeof(unsigned));
    dataset_size_ = reverseBits(*((unsigned int*) data));
    fd.read(data, sizeof(unsigned));
    rows = reverseBits(*((unsigned int*) data));
    fd.read(data, sizeof(unsigned));
    cols = reverseBits(*((unsigned int*) data));
    std::cout << "magic number: " << magic_number << "dataset_size: " << dataset_size_ << " rows "<< rows << " cols " << cols <<  std::endl;
    input_size_ = rows*cols;
    images.resize(dataset_size_);
    data[0] = 0;
    data[1] = 0;
    data[2] = 0;
    while(!fd.eof()) {
        images[i].resize(input_size_);
        for (int j = 0; j < input_size_; j++) {
            fd.read(&data[3], 1);
            images[i][j] = reverseBits(*((unsigned int*) data));
        }
        i++;
    }
    fd.close();
}

void MNIST::parse_labels_file(char* label_file) {
    std::fstream fd;
    unsigned int magic_number, i = 0;
    char value;
    char data[4];
    fd.open(label_file, std::ios::in | std::ios::binary);
    fd.read(data, sizeof(unsigned));
    magic_number = reverseBits(*((unsigned int*) data));
    fd.read(data, sizeof(unsigned));
    dataset_size_ = reverseBits(*((unsigned int*) data));

    labels.resize(dataset_size_);
    while(!fd.eof()) {
        labels[i] = std::vector<float>(label_size_, 0);
        fd.read(&value, 1);
        labels[i][unsigned(value)] = 1;
        i++;
    }
    fd.close();
}

size_t MNIST::getInputDim() {
    return input_size_;
}

size_t MNIST::getLabelDim() {
    return label_size_;
}

size_t MNIST::getDatasetSize() {
    return dataset_size_;
}

void MNIST::get_item(int index, float* data, float* label) {
    memcpy(data, images[index].data(), sizeof(float)*input_size_);
    memcpy(label, labels[index].data(), sizeof(float)*label_size_);
}

void MNIST::get_item_range(int start, int end, float* data_batch, float* label_batch) {
    for (int i = start; i < end; i++){
        memcpy(data_batch+(i-start)*input_size_, images[i].data(), sizeof(float)*input_size_);
        memcpy(label_batch+(i-start)*label_size_, labels[i].data(), sizeof(float)*label_size_);
    }
}