#include "pooling_layer.h"

using namespace layers;

PoolingLayer::PoolingLayer(cudnnHandle_t* handle,
    int window_height,
    int window_width,
    int vertical_stride,
    int horizontal_stride,
    int batch_size,
    int input_height,
    int input_width,
    int input_channels,
    padding_type pad,
    cudnnPoolingMode_t mode) {
    // mode -- CUDNN_POOLING_MAX(0), CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING(1)
    //                 CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING(2), CUDNN_POOLING_MAX_DETERMINISTIC(3)

    handle_ = handle;
    ibatch_size = batch_size;
    ichannels = input_channels;
    iheight = input_height;
    iwidth = input_width;

    checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_descriptor));
    if (pad == SAME) {
        checkCUDNN(cudnnSetPooling2dDescriptor(
            /*poolingDesc=*/pooling_descriptor,
            /*mode=*/mode,
            /*maxpoolingNanOpt=*/CUDNN_NOT_PROPAGATE_NAN,
            /*windowHeight*/window_height,
            /*windowWidth*/window_width,
            /*verticalPadding*/window_height/2,
            /*horizontalPadding*/window_width/2,
            /*verticalStride*/vertical_stride,
            /*horizontalStride*/horizontal_stride
        ));
    } else if (pad == VALID) {
        checkCUDNN(cudnnSetPooling2dDescriptor(
            /*poolingDesc=*/pooling_descriptor,
            /*mode=*/mode,
            /*maxpoolingNanOpt=*/CUDNN_NOT_PROPAGATE_NAN,
            /*windowHeight*/window_height,
            /*windowWidth*/window_width,
            /*verticalPadding*/0,
            /*horizontalPadding*/0,
            /*verticalStride*/vertical_stride,
            /*horizontalStride*/horizontal_stride
        ));
    }

    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/ibatch_size,
        /*channels=*/ichannels,
        /*image_height=*/iheight,
        /*image_width=*/iwidth));

    checkCUDNN(cudnnGetPooling2dForwardOutputDim(pooling_descriptor,
        input_descriptor,
        &obatch_size,
        &ochannels,
        &oheight,
        &owidth));

    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/obatch_size,
        /*channels=*/ochannels,
        /*image_height=*/oheight,
        /*image_width=*/owidth
    ));
}

void PoolingLayer::forward(float alpha, float beta, float* d_input, float* d_output) {
    checkCUDNN(cudnnPoolingForward(*handle_,
        pooling_descriptor,
        &alpha,
        input_descriptor,
        d_input,
        &beta,
        output_descriptor,
        d_output
    ));
}

void PoolingLayer::backward(float alpha, float beta, float* d_y, float* d_dy, float* d_x, float* d_dx) {
    checkCUDNN(cudnnPoolingBackward(*handle_,
        pooling_descriptor,
        &alpha,
        output_descriptor,
        d_y,
        output_descriptor,
        d_dy,
        input_descriptor,
        d_x,
        &beta,
        input_descriptor,
        d_dx
    ));
}

int PoolingLayer::get_output_shape_and_bytes(int shape[]) {
    shape[0] = obatch_size;
    shape[1] = oheight;
    shape[2] = owidth;
    shape[3] = ochannels;
    return sizeof(float)*obatch_size*ochannels*oheight*owidth;
}

int PoolingLayer::get_total_memory()
{
  int shape[4];
  return get_output_shape_and_bytes(shape);
}

PoolingLayer::~PoolingLayer() {
    checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
	checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor));
    checkCUDNN(cudnnDestroyPoolingDescriptor(pooling_descriptor));
}
