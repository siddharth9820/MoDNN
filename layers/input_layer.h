#ifndef INPUT_LAYER_H_
#define INPUT_LAYER_H_

#include "layers.h"

namespace layers{

  /*! \class InputLayer
    \brief Input Layer

    This class is used to create a Input layer. This layer is used to store input data and ground truth labels.
  */  
  class InputLayer : public layers::Layer
  {
    public:
      int num_classes; /*!< Total number of classes / Class labels dimension */

      /**
       * Intializes Input Layer Class.
       * @param batch_size Input batch size.
       * @param height Input data height. 
       * @param width Input data width.
       * @param channels Input data channels.
       * @param num_classes Dimension of class labels. 
       */
      InputLayer(int batch_size, int height, int width, int channels, int num_classes);//NHWC format
      
      /**
       * Randomly generate input data.
       * @param data data buffer pointer.
       * @param labels data labels buffer pointer.
       */
      void randomly_populate(float * data,float * labels);

      /**
       * Update the input to the neural network.
       * @param data new input data.
       * @param labels new ground truth labels to the input data.
       * @param data_buffer device data buffer.
       * @param labels_buffer device labels buffer.
       */
      void update_batch(float* data, float* labels, float* data_buffer, float* labels_buffer);

      /**
      * Return the no of bytes occupied by the output and set the shape of the output in the passed array.
      * @param shape Shape of the output is set in this array.
      */
      int get_output_shape_and_bytes(int shape[]);

      /**
       * Return the total memory occupied by Flattern Layer.
       */
      int get_total_memory();

  };

}

#endif
