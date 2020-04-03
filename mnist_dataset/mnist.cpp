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

    label_size_ = 1;

    parse_images_file(images_filename);

    parse_labels_file(labels_filename);

    if (shuffle){
        std::shuffle(images.begin(), images.end(), std::default_random_engine(seed));
        std::shuffle(labels.begin(), labels.end(), std::default_random_engine(seed));
    }

}

void MNIST::shuffle()
{
  unsigned seed_temp = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(images.begin(), images.end(), std::default_random_engine(seed_temp));
  std::shuffle(labels.begin(), labels.end(), std::default_random_engine(seed_temp));
}

int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
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
    input_size_ = rows*cols;
    std::cout << input_size_ << " " << rows << " " << cols << " " << dataset_size_ << std::endl;
    images.resize(dataset_size_);
    data[0] = 0;
    data[1] = 0;
    data[2] = 0;
    while(!fd.eof()) {
        images[i].resize(input_size_);
        for (int j = 0; j < input_size_; j++) {
            fd.read(&data[3], 1);
            images[i][j] = reverseBits(*((unsigned int*) data))/255.0;
        }
        i++;
    }
    fd.close();
}

// void MNIST::parse_images_file(char* images_file) {
//     std::ifstream fd(images_file, std::ios::in | std::ios::binary);
//     char data[4];
//     unsigned int magic_number, rows, cols, i = 0;
//     char pixel;
//     fd.read(data, sizeof(unsigned));
//     //magic_number = reverseBits(*((unsigned int*) data));
//     magic_number = *((unsigned int *)data);
//
//     fd.read(data, sizeof(unsigned));
//     //dataset_size_ = reverseBits(*((unsigned int*) data));
//     //dataset_size_ = *((unsigned int *)data);
//     dataset_size_ = 60000;
//
//     fd.read(data, sizeof(unsigned));
//     //rows = reverseBits(*((unsigned int*) data));
//     //rows = *((unsigned int *)data);
//     rows = 28;
//
//     fd.read(data, sizeof(unsigned));
//     //cols = reverseBits(*((unsigned int*) data));
//     //cols = *((unsigned int *)data);
//     cols = 28;
//
//     input_size_ = rows*cols;
//
//     std::cout << dataset_size_ <<" "<<input_size_ << std::endl;
//
//     images.resize(dataset_size_);
//     data[0] = 0;
//     data[1] = 0;
//     data[2] = 0;
//     while(i < dataset_size_ +100 && !fd.eof()) {
//         images[i].resize(input_size_);
//         for (int j = 0; j < input_size_; j++) {
//             fd.read(&data[3], 1);
//             //images[i][j] = reverseBits(*((unsigned int*) data))/255.0;
//             images[i][j] = *((unsigned int*) data)/255.0;
//         }
//         i++;
//         std::cout << i << std::endl;
//     }
//     fd.close();
//     // std::ifstream file (/*full_path*/images_file,std::ios::binary);
//     // std::cout << file.is_open() << std::endl;
//     // if (file.is_open())
//     // {
//     //     int magic_number=0;
//     //     int number_of_images=0;
//     //     int n_rows=0;
//     //     int n_cols=0;
//     //     file.read((char*)&magic_number,sizeof(magic_number));
//     //     magic_number= reverseInt(magic_number);
//     //     file.read((char*)&number_of_images,sizeof(number_of_images));
//     //     number_of_images= reverseInt(number_of_images);
//     //     file.read((char*)&n_rows,sizeof(n_rows));
//     //     n_rows= reverseInt(n_rows);
//     //     file.read((char*)&n_cols,sizeof(n_cols));
//     //     n_cols= reverseInt(n_cols);
//     //
//     //     dataset_size_ = number_of_images;
//     //     input_size_ = n_rows*n_cols;
//     //
//     //     std::cout << dataset_size_ << " " << n_rows << " " << n_cols << std::endl;
//     //
//     //     images.resize(dataset_size_);
//     //
//     //     for(int i=0;i<number_of_images;++i)
//     //     {
//     //         images[i].resize(input_size_);
//     //         for(int r=0;r<n_rows;++r)
//     //         {
//     //             for(int c=0;c<n_cols;++c)
//     //             {
//     //                 unsigned char temp=0;
//     //                 file.read((char*)&temp,sizeof(temp));
//     //                 images[r][c] = reverseBits(*((unsigned int*) temp))/255.0;
//     //                 if(i==0)
//     //                   std::cout << images[r][c] << " ";
//     //             }
//     //             std::cout << std::endl;
//     //         }
//     //     }
//     // }
//
// }

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
        fd.read(&value, 1);
        labels[i] = int(value);
        i++;
    }
    fd.close();
}

size_t MNIST::getInputDim() {
    return input_size_;
}

size_t MNIST::getLabelDim() {
    return 10;
}

size_t MNIST::getDatasetSize() {
    return dataset_size_;
}

void MNIST::get_item(int index, float* data, float* label) {
    memcpy(data, images[index].data(), sizeof(float)*input_size_);
    memcpy(label, &labels[index], sizeof(float)*label_size_);
}

void MNIST::get_item_range(int start, int end, float* data_batch, float* label_batch) {
    for (int i = start; i < end; i++){
        memcpy(data_batch+(i-start)*input_size_, images[i].data(), sizeof(float)*input_size_);
        memcpy(label_batch+(i-start)*label_size_, &labels[i], sizeof(float)*label_size_);
    }
}
