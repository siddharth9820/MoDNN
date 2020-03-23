#include <cudnn.h>
#include "layers.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>


using namespace layers;
using namespace network



cv::Mat load_image(const char* image_path) {
  cv::Mat image = cv::imread(image_path, CV_LOAD_IMAGE_COLOR);
  image.convertTo(image, CV_32FC3);
  cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
  std::cerr << "Input Image: " << image.rows << " x " << image.cols << " x "
            << image.channels() << std::endl;
  return image;
}

void save_image(const char* output_filename,
                float* buffer,
                int height,
                int width,
                int channels) {
  cv::Mat output_image(height, width, CV_32FC3, buffer);
  // Make negative values zero.
  cv::threshold(output_image,
                output_image,
                /*threshold=*/0,
                /*maxval=*/0,
                cv::THRESH_TOZERO);
  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image.convertTo(output_image, CV_8UC3);
  cv::imwrite(output_filename, output_image);
  std::cerr << "Wrote output to " << output_filename << std::endl;
}



int main(int argc, const char* argv[]) {

  cv::Mat image = load_image(argv[1]);
  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  padding_type pad = VALID;
  ConvLayer layer = ConvLayer(cudnn,
                  1, /*batch_size*/
                  image.rows, /*input_height*/
                  image.cols,/*input_cols*/
                  image.channels(),/*input_channels*/
                  5,/*kernel_height*/
                  5,/*kernel_width*/
                  3/*output_channels*/,
                  pad
           );


  //Step 1 - Device Pointers
  float *d_kernel{nullptr}, *d_input{nullptr}, *d_output{nullptr};
  void *d_workspace{nullptr};

  //Step 2 - host pointers
  float* h_output;


  int input_shape[4],output_shape[4];

  //Step 3 - allocate internal memory
  int layer_internal_mem = layer.allocate_internal_mem(&d_kernel,&d_workspace);

  //Step 4 - Get input and output information
  int output_bytes = layer.get_output_shape_and_bytes(output_shape);
  int input_bytes = layer.get_input_shape_and_bytes(input_shape);

  std::cout << "Internal memory footprint of the Convolution Layer :- " << " " << layer_internal_mem <<std::endl;

  //Step 5 - Allocate and copy to input and output memory
  cudaMalloc(&d_input, input_bytes);
  cudaMemcpy(d_input, image.ptr<float>(0), input_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(&d_output, output_bytes);
  cudaMemset(d_output, 0, output_bytes);

  //Step 6 - Randomly initialize layer parameters
  layer.populate_filter_params(d_kernel);

  const float alpha = 1.0f, beta = 0.0f;

  //Step 7 - Forward Pass through the CNN
  layer.forward(alpha, beta, d_input, d_kernel, d_workspace, d_output);

  //Step 8 - Copy output from the device
  h_output = new float[output_bytes];
  cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
  save_image("cudnn-out.png", h_output, output_shape[1], output_shape[2], image.channels());



  delete[] h_output;
  cudaFree(d_kernel);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_workspace);
  cudnnDestroy(cudnn);

}
