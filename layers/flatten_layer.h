#ifndef FLATTEN_LAYER_H_
#define FLATTEN_LAYER_H_

#include "layers.h"

namespace layers{
class Flatten : public layers::Layer
 {
  public:
   Flatten(int batch_size,int input_height,int input_width,int input_channels);
   int get_output_shape_and_bytes(int shape[]);
   int get_input_shape_and_bytes(int shape[]);
   int get_total_memory();
 };
}

#endif
