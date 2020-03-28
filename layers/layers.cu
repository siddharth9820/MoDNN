#include "layers.h"

using namespace layers;

std::map<std::string,float*> init_buffer_map()
{
  std::map<std::string,float*> buffer_map;
  buffer_map["input"] = nullptr;
  buffer_map["output"] = nullptr;
  buffer_map["workspace"] = nullptr;
  buffer_map["params"] = nullptr;

  buffer_map["dinput"] = nullptr;
  buffer_map["doutput"] = nullptr;
  buffer_map["dparams"] = nullptr;

  return buffer_map;
}

const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}


void Layer::forward()
{

}

int calc_bytes_from_shape(int shape[])
{
  int bytes = 1;
  for(int i=0;i<4;i++)
  {
    if(shape[i]==-1)break;
    else bytes*=shape[i];
  }
  return bytes;
}
