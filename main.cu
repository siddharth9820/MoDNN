#include "layers/layers.h"
#include <fstream>

using namespace layers;
using namespace network;



cv::Mat load_image(const char* image_path) {
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR );
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

void print_output(float * layer,int shape[])
{
  for(int i=0;i<shape[0];i++){
    for(int j=0;j<shape[1];j++){
        std::cout << layer[i*shape[1]+j] << " ";
    }
    std::cout << std::endl;
  }

}


int main(int argc, const char* argv[])
{
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    std::ofstream outdata;

    std::vector<std::string> specs = {"input 32 28 28 1 8","flatten","fc 10","softmax"};
    seqNetwork nn = seqNetwork(cudnn,cublas,specs);
    nn.print_network_info();
    nn.allocate_memory();
    int shape[4];
    nn.get_output_shape(shape,nn.num_layers-1);

    std::cout << "Printing output shape of Neural Network" << std::endl;
    for(int i=0;i<4;i++)
      std::cout << shape[i] <<" "<<" ";
    std::cout<<std::endl;

    std::cout << "Randomising input to the neural network" << std::endl;
    nn.randomise_batch();

    std::cout << "Randomising Parameters of the neural network" << std::endl;
    nn.randomise_params();

    std::cout << "Forward Pass for the neural network" << std::endl;

    float * output;
    for(int i=0;i<10;i++)
    {
      nn.forward();
      std::cout << "Offloading Ouput " <<std::endl;
      output = nn.offload_buffer(nn.num_layers-1,"output",shape);
      print_output(output,shape);
      nn.backward();
    }


    return 0;

}
