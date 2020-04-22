#ifndef FLATTEN_LAYER_H_
#define FLATTEN_LAYER_H_

#include "layers.h"

namespace layers{

/*! \class Flatten
  \brief Flatten Layer

  This class is used to create a Flatten layer.
*/  
class Flatten : public layers::Layer
 {
  public:
   
   /**
     * Intializes Flatten Layer Class.
     * @param batch_size Input batch size.
     * @param input_height Input data height. 
     * @param input_width Input data width.
     * @param input_channels Input data channels.
     */
   Flatten(int batch_size,int input_height,int input_width,int input_channels);
   
   /**
    * Return the no of bytes occupied by the output and set the shape of the output in the passed array.
    * @param shape Shape of the output is set in this array.
    */
   int get_output_shape_and_bytes(int shape[]);
   
   /**
    * Return the no of bytes occupied by the intput and set the shape of the input in the passed array.
    * @param shape Shape of the input is set in this array.
    */
   int get_input_shape_and_bytes(int shape[]);
   
   /**
    * Return the total memory occupied by Flatten Layer.
    */
   int get_total_memory();
 };
}

#endif
