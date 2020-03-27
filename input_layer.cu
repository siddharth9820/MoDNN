#include "input_layer.h"

using namespace layers;
InputLayer::InputLayer(int batch_size, int height, int width, int channels,int num_classes)
{
  ibatch_size = obatch_size = batch_size;
  iheight = oheight = height;
  iwidth = owidth = width;
  ichannels = ochannels = channels;
  this->num_classes = num_classes;
}

int InputLayer::get_output_shape_and_bytes(int shape[])
{
    //Get Output Shape in NHWC format
    shape[0] = obatch_size;
    shape[1] = oheight;
    shape[2] = owidth;
    shape[3] = ochannels;
    return shape[0]*shape[1]*shape[2]*shape[3]*sizeof(float);
}



void InputLayer::randomly_populate(float *data,float * labels)
{
  float init_params[obatch_size][oheight][owidth][ochannels];
  int init_labels[obatch_size];
  int class_;
  std::normal_distribution<float> distribution(MU,SIGMA);
  std::default_random_engine generator;

  for(int data_point = 0; data_point < obatch_size; data_point++)
    for(int row=0;row<oheight;row++)
      for(int col=0;col<owidth;col++)
        for(int ochannel=0;ochannel < ochannels; ochannel++){
          init_params[data_point][row][col][ochannel] = distribution(generator);
        }

  //std::cout << "Checking random input layer" << std::endl;
  //std::cout << init_params[0][0][0][1] << std::endl;
  cudaMemcpy(data,init_params,sizeof(init_params),cudaMemcpyHostToDevice);

  for (int j = 0; j < obatch_size; j++)
  {
    class_ = rand() % num_classes;
    init_labels[j] = class_;

  }

  cudaMemcpy((void *)(labels),init_labels,sizeof(init_labels),cudaMemcpyHostToDevice);

}