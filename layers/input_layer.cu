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
  float* init_params = (float*)malloc( obatch_size *oheight*owidth*ochannels*sizeof(float));
  int* init_labels = (int*)malloc(obatch_size*sizeof(int));
  int class_;
  std::normal_distribution<float> distribution(MU,SIGMA);
  std::default_random_engine generator;

  int dim1 = owidth*ochannels;
  int dim2 = oheight*dim1;

  for(int data_point = 0; data_point < obatch_size; data_point++)
    for(int row=0;row<oheight;row++)
      for(int col=0;col<owidth;col++)
        for(int ochannel=0;ochannel < ochannels; ochannel++){
          init_params[data_point*dim2 + row*dim1 + col*ochannels + ochannel] = distribution(generator);
        }

  //std::cout << "Checking random input layer" << std::endl;
  //std::cout << init_params[0][0][0][1] << std::endl;
  gpuErrchk(cudaMemcpy(data,init_params, obatch_size *oheight*owidth*ochannels*sizeof(float),cudaMemcpyHostToDevice));

  for (int j = 0; j < obatch_size; j++)
  {
    class_ = rand() % num_classes;
    init_labels[j] = class_;

  }

  gpuErrchk(cudaMemcpy((void *)(labels),init_labels,obatch_size*sizeof(int),cudaMemcpyHostToDevice));

}

void InputLayer::update_batch(float* data, float* labels, float* data_buffer, float* labels_buffer) {
  // std::cout << "Update Batch : Copying into " << data_buffer << std::endl;
  // std::cout << obatch_size *oheight*owidth*ochannels*sizeof(float) << " Bytes into output" << std::endl;
  // std::cout << obatch_size << std::endl;
  gpuErrchk(cudaMemcpy(data_buffer,data, obatch_size *oheight*owidth*ochannels*sizeof(float),cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpy((void *)(labels_buffer),labels,obatch_size*sizeof(int),cudaMemcpyHostToDevice));
}

int InputLayer::get_total_memory()
{
  //std::cout << "In input layer get total  memory " << std::endl;
  int shape[4];
  return get_output_shape_and_bytes(shape);
}
