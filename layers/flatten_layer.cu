#include "flatten_layer.h"

using namespace layers;

Flatten::Flatten(int batch_size,int input_height,int input_width,int input_channels)
{
  ibatch_size = batch_size;
  ichannels = input_channels;
  iheight = input_height;
  iwidth = input_width;
  obatch_size = batch_size;
  oheight = input_channels*input_height*input_width;

}

int Flatten::get_output_shape_and_bytes(int shape[])
{
  shape[0] = obatch_size;
  shape[1] = oheight;
  shape[2] = 1;
  shape[3] = 1;

  return obatch_size*oheight*sizeof(float);
}

int Flatten::get_input_shape_and_bytes(int shape[])
{
  shape[0] = ibatch_size;
  shape[1] = iheight;
  shape[2] = iwidth;
  shape[3] = ichannels;

  return obatch_size*oheight*sizeof(float);
}

int Flatten::get_total_memory()
{
  return 0;
}
