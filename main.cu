#include "layers/layers.h"
#include <fstream>
#include <math.h>

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
    for(int j=0;j<shape[1]*shape[2]*shape[3];j++){
        std::cout << layer[i*shape[1]*shape[2]*shape[3]+j] << " ";
    }
    std::cout << std::endl;
  }

}

float categorical_cross_entropy_loss(float * softmax_dinput,int shape[])
{
  float temp,loss=0;
  for(int i=0;i<shape[0];i++){
    for(int j=0;j<shape[1];j++){
        temp = softmax_dinput[i*shape[1]+j];
        if(temp<=0)
        {
          temp = temp+1;
          loss += -log(temp);
          break;
        }
    }

  }
  return loss/shape[0];
}


int main(int argc, const char* argv[])
{
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    std::ofstream outdata;

    std::vector<std::string> specs = {"input 4 4 5 1 3","conv 3 3 3","relu","maxpool 2 2 2 2","flatten","fc 50","relu","fc 3","softmax"};
    seqNetwork nn = seqNetwork(cudnn,cublas,specs,LR);
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

    float * output,loss;
    for(int i=0;i<10000;i++)
    {
      nn.forward();
      nn.backward();
      if(i%1000==0){
      output = nn.offload_buffer(nn.num_layers-1,"dinput",shape);
      loss = categorical_cross_entropy_loss(output,shape);
      std::cout << "Iteration number "<<i<<" CCE Loss :- "<<loss <<std::endl;
      }
    }


    //test for relu - passed
    // nn.forward();
    // output = nn.offload_buffer(3,"output",shape);
    // print_output(output,shape);
    // nn.backward();
    // output = nn.offload_buffer(3,"doutput",shape);
    // print_output(output,shape);
    // output = nn.offload_buffer(3,"dinput",shape);
    // print_output(output,shape);
    return 0;

}
