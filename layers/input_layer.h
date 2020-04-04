#ifndef INPUT_LAYER_H_
#define INPUT_LAYER_H_

#include "layers.h"

namespace layers{

  class InputLayer : public layers::Layer
  {
    public:
      int num_classes;
      InputLayer(int batch_size, int height, int width, int channels, int num_classes);//NHWC format
      void randomly_populate(float * data,float * labels);
      void update_batch(float* data, float* labels, float* data_buffer, float* labels_buffer);
      int get_output_shape_and_bytes(int shape[]);
      int get_total_memory();

  };

}

#endif
