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
      int get_output_shape_and_bytes(int shape[]);

  };

}

#endif