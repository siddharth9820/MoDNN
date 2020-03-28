#include "layers/layers.h"


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



int main(int argc, const char* argv[])
{
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    std::vector<std::string> specs = {"input 10 5 5 1 5","conv 3 3 5","maxpool 2 2 2 2","flatten","fc 5","softmax"};
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
    nn.forward();

    float * output;

    std::cout << "Offloading all activations " <<std::endl;


    for(int layer_no=2;layer_no<nn.num_layers;layer_no++)
    { //start from flatten
      output = nn.offload_buffer(layer_no,"output",shape);
      
      //float * output;
      std::cout<< "Printing Output of  Layer " << layer_no << std::endl;
      std::cout << "Layer Type : "<<nn.layer_info[layer_no][0] << std::endl;
      std::cout << "Layer Shape" << shape[0] <<" "<<shape[1] << " "<<shape[2] << " " << shape[3] << std::endl;
      //output = nn.layer_offloaded_buffers[nn.num_layers-1]["output"];

      for(int i=0;i<shape[0];i++){
        for(int j=0;j<shape[1];j++){
            std::cout << output[i*shape[1]+j] << " ";
        }
        std::cout << std::endl;
      }
      if(nn.layer_info[layer_no][0]=="fc")
      {
        std::cout<< "Printing Params " << std::endl;
        output = nn.offload_buffer(layer_no,"params",shape);
        for(int i=0;i<shape[0];i++){
          for(int j=0;j<shape[1];j++){
              std::cout << output[i*shape[1]+j] << " ";
          }
          std::cout << std::endl;
        }
      }
      std::cout << std::endl;
    }




    nn.backward();


    // output = (float*)malloc(shape[0]*shape[1]*sizeof(float));
    // std::cout <<"Printing result of softmax backward" << std::endl;
    // cudaMemcpy(output,nn.layer_buffers[nn.num_layers-1]["dinput"],shape[0]*shape[1]*sizeof(float),
    //   cudaMemcpyDeviceToHost);

    std::cout <<"\n"<<"Dinput for softmax" << std::endl;
    output = nn.offload_buffer(nn.num_layers-1,"dinput",shape);

    for(int i=0;i<shape[0];i++){
      for(int j=0;j<shape[1];j++){
          std::cout << output[i*shape[1]+j] << " ";
      }
      std::cout << std::endl;
    }

    std::cout <<"\n"<<"Doutput for FC" << std::endl;
    std::cout << "Layer Shape " << shape[0] <<" "<<shape[1] << " "<<shape[2] << " " << shape[3] << std::endl;

    output = nn.offload_buffer(nn.num_layers-2,"doutput",shape);

    for(int i=0;i<shape[0];i++){
      for(int j=0;j<shape[1];j++){
          std::cout << output[i*shape[1]+j] << " ";
      }
      std::cout << std::endl;
    }

    std::cout <<"Dinput for FC" << std::endl;
    // cudaMemcpy(output,nn.layer_buffers[nn.num_layers-1]["dinput"],shape[0]*shape[1]*sizeof(float),
    //   cudaMemcpyDeviceToHost);
    output = nn.offload_buffer(nn.num_layers-2,"dinput",shape);
    std::cout << "Layer Shape " << shape[0] <<" "<<shape[1] << " "<<shape[2] << " " << shape[3] << std::endl;



    for(int i=0;i<shape[0];i++){
      for(int j=0;j<shape[1];j++){
          std::cout << output[i*shape[1]+j] << " ";
      }
      std::cout << std::endl;
    }
    //
    std::cout <<"\n"<<"Dparams for FC" << std::endl;
    // cudaMemcpy(output,nn.layer_buffers[nn.num_layers-1]["dinput"],shape[0]*shape[1]*sizeof(float),
    //   cudaMemcpyDeviceToHost);
    output = nn.offload_buffer(nn.num_layers-2,"dparams",shape);
    std::cout << "Layer Shape " << shape[0] <<" "<<shape[1] << " "<<shape[2] << " " << shape[3] << std::endl;

    for(int i=0;i<shape[0];i++){
      for(int j=0;j<shape[1];j++){
          std::cout << output[i*shape[1]+j] << " ";
      }
      std::cout << std::endl;
    }

    return 0;
    // int deviceCount;
    // cudaGetDeviceCount(&deviceCount);
    // int device;
    // for (device = 0; device < deviceCount; ++device) {
    //     cudaDeviceProp deviceProp;
    //     cudaGetDeviceProperties(&deviceProp, device);
    //     printf("Device %d has compute capability %d.%d., conc kernel - %d\n",
    //            device, deviceProp.major, deviceProp.minor,deviceProp.concurrentKernels);
    // }
}
