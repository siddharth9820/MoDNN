#include "layers.h"
#include "pooling_layer.h"

using namespace network;
using namespace layers;

seqNetwork::seqNetwork(cudnnHandle_t cudnn,cublasHandle_t cublas,std::vector<std::string> &specs)
{
  /*
  Specs is a vector of strings specifying the Neural Network.
  Input -> "input N H W C"
  Conv ->  "conv H W C"
  */
  num_layers = specs.size();
  handle = cudnn;
  blas_handle = cublas;
  for(int i=0;i<num_layers;i++)
    {
      std::stringstream ss(specs[i]);
      std::vector<std::string> info;
      std::string tok;
      while(getline(ss, tok, ' ')){
        info.push_back(tok);
      }
      layer_info.push_back(info);
    }
}

void seqNetwork::print_network_info()
{
  for(int i=0;i<num_layers;i++)
  {
    std::cout << "Layer "<<i+1<<" : "<<layer_info[i][0] << std::endl;
    std::cout << "Specs : ";
    for(int j=1;j<layer_info[i].size();j++)
      std::cout << layer_info[i][j] << " ";
    std::cout << std::endl;
  }
}

void seqNetwork::get_output_shape(int shape[],int i)
{
  Layer *last_layer = layer_objects[i];
  if (layer_info[i][0] == "flatten")
    ((Flatten*)last_layer)->get_output_shape_and_bytes(shape);
  else if(layer_info[i][0] == "conv")
    ((ConvLayer*)last_layer) -> get_output_shape_and_bytes(shape);
  else if(layer_info[i][0] == "fc")
    ((FCLayer*)last_layer) -> get_output_shape_and_bytes(shape);
  else if(layer_info[i][0] == "softmax")
    ((Softmax*)last_layer) -> get_output_shape_and_bytes(shape);
  else if(layer_info[i][0] == "input")
    ((InputLayer*)last_layer) -> get_output_shape_and_bytes(shape);
  else if(layer_info[i][0] == "maxpool" || layer_info[i][0] == "avgpool" )
    ((PoolingLayer*)last_layer)->get_output_shape_and_bytes(shape);
}

void seqNetwork::allocate_memory()
{
  std::string layer_type;
  int shape[4],batch_size,rows,columns,channels;
  int kernel_rows,kernel_cols,kernel_channels,bytes;
  int input_height,output_height;
  int window_height, window_width;
  int vertical_stride, horizontal_stride;
  int pad, pooling_type;

  std::cout << "Allocating memory for the Neural Network" << std::endl;
  layer_buffers.resize(num_layers);
  layer_offloaded_buffers.resize(num_layers);

  for(int i=0;i<num_layers;i++)
  {
    layer_type = layer_info[i][0];
    std::cout << "Layer "<<i+1<<" : "<<layer_type << std::endl;
    layer_offloaded_buffers[i] = init_buffer_map();
    if(layer_type == "input")
    {
      batch_size = atoi(layer_info[i][1].c_str());
      rows = atoi(layer_info[i][2].c_str());
      columns = atoi(layer_info[i][3].c_str());
      channels = atoi(layer_info[i][4].c_str());

      InputLayer * new_ip = new InputLayer(batch_size,rows,columns,channels);
      layer_objects.push_back(new_ip);

      bytes = new_ip->get_output_shape_and_bytes(shape);
      layer_buffers[i] = init_buffer_map();
      cudaMalloc(&(layer_buffers[i]["output"]),bytes);


    }
    else if(layer_type == "conv")
    {
      //batch_size is already fixed in the first input layer
      //std::cout << "Allocating Memory to Conv Layer" << std::endl;
      kernel_rows = atoi(layer_info[i][1].c_str());
      kernel_cols = atoi(layer_info[i][2].c_str());
      kernel_channels = atoi(layer_info[i][3].c_str());

      this->get_output_shape(shape,i-1);

      batch_size = shape[0];
      rows = shape[1];
      columns = shape[2];
      channels = shape[3];

      ConvLayer * new_conv = new ConvLayer(handle,batch_size,rows,columns,channels,kernel_rows,kernel_cols,kernel_channels,VALID);

      bytes =  new_conv->get_output_shape_and_bytes(shape);

      layer_objects.push_back(new_conv);

      layer_buffers[i] = init_buffer_map();
      cudaMalloc(&(layer_buffers[i]["output"]),bytes);
      layer_buffers[i]["input"] = layer_buffers[i-1]["output"];
      new_conv -> allocate_internal_mem(&(layer_buffers[i]["params"]),(void**)&(layer_buffers[i]["workspace"]));

    }
    else if(layer_type == "flatten")
    {
      this->get_output_shape(shape,i-1);

      batch_size = shape[0];
      rows = shape[1];
      columns = shape[2];
      channels = shape[3];
      std::cout << "Setting up flatten layer - "<< batch_size <<" " << rows << " "<<columns <<" "<<channels << std::endl;

      Flatten * new_flat = new Flatten(batch_size,rows,columns,channels);
      layer_objects.push_back(new_flat);

      layer_buffers[i]["input"] = layer_buffers[i-1]["output"];
      layer_buffers[i]["output"] = layer_buffers[i]["input"];
    }
    else if(layer_type == "fc")
    {
      this->get_output_shape(shape,i-1);

      batch_size = shape[0];
      input_height = shape[1];
      output_height = atoi(layer_info[i][1].c_str());

      FCLayer * new_fc = new FCLayer(blas_handle,batch_size,input_height,output_height);

      bytes =  new_fc->get_output_shape_and_bytes(shape);



      layer_buffers[i] = init_buffer_map();
      cudaMalloc(&(layer_buffers[i]["output"]),bytes);
      layer_buffers[i]["input"] = layer_buffers[i-1]["output"];
      new_fc -> allocate_internal_mem(&(layer_buffers[i]["params"]));


      layer_objects.push_back(new_fc);

    }
    else if(layer_type == "softmax")
    {
      this->get_output_shape(shape,i-1);

      batch_size = shape[0];
      input_height = shape[1];

      Softmax * new_softmax = new Softmax(handle,batch_size,input_height);

      bytes =  new_softmax->get_output_shape_and_bytes(shape);

      layer_objects.push_back(new_softmax);

      layer_buffers[i] = init_buffer_map();
      cudaMalloc(&(layer_buffers[i]["output"]),bytes);
      layer_buffers[i]["input"] = layer_buffers[i-1]["output"];


    }
    else if(layer_type == "maxpool" || layer_type == "avgpool") {
      this->get_output_shape(shape, i-1);
      
      window_height = atoi(layer_info[i][1].c_str());
      window_width = atoi(layer_info[i][2].c_str());
      vertical_stride = atoi(layer_info[i][3].c_str());
      horizontal_stride = atoi(layer_info[i][4].c_str());
      pad = VALID;

      if (layer_type == "maxpool")
        pooling_type = CUDNN_POOLING_MAX;
      else if (layer_type == "avgpool"){
        if (pad == VALID)
          pooling_type = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        else
          pooling_type = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
      }

      batch_size = shape[0];
      rows = shape[1];
      columns = shape[2];
      channels = shape[3];

      PoolingLayer* new_pooling = new PoolingLayer(handle, 
        window_height, 
        window_width,
        vertical_stride,
        horizontal_stride,
        batch_size,
        rows,
        columns,
        channels,
        pad,
        pooling_type
      );

      bytes =  new_pooling->get_output_shape_and_bytes(shape);
      layer_buffers[i] = init_buffer_map();
      layer_buffers[i]["input"] = layer_buffers[i-1]["output"];
      cudaMalloc(&(layer_buffers[i]["output"]),bytes);

      layer_objects.push_back(new_pooling);
    }

  }
}

void seqNetwork::randomise_input()
{
  ((InputLayer*)layer_objects[0])->randomly_populate(layer_buffers[0]["output"]);
}

void seqNetwork::randomise_params()
{
  for(int i=0;i<num_layers;i++)
  {
    if(layer_info[i][0]=="conv")
    {
      ((ConvLayer*)layer_objects[i])->populate_filter_params(layer_buffers[i]["params"]);
    }
    else if(layer_info[i][0]=="fc")
    {
      ((FCLayer*)layer_objects[i])->populate_filter_params(layer_buffers[i]["params"]);
    }
  }
}

void seqNetwork::forward()
{
  for(int i=0;i<num_layers;i++)
  {
    std::map<std::string,float*> buffer_map = layer_buffers[i];
    std::string layer_type = layer_info[i][0];

    if(layer_type=="input")continue;
    else if(layer_type=="conv")
    {
      ConvLayer * layer_obj = (ConvLayer*)(layer_objects[i]);
      layer_obj -> forward(1.0,0.0,buffer_map["input"],buffer_map["params"],(void*)buffer_map["workspace"],buffer_map["output"]);
    }
    else if(layer_type=="fc")
    {
      FCLayer * layer_obj = (FCLayer*)(layer_objects[i]);
      layer_obj -> forward(buffer_map["input"],buffer_map["params"],buffer_map["output"]);
    }
    else if(layer_type == "softmax")
    {
      Softmax* layer_obj = (Softmax*)(layer_objects[i]);
      layer_obj -> forward(buffer_map["input"],buffer_map["output"]);
    }
    else if(layer_type == "maxpool" || layer_type == "avgpool"){
      PoolingLayer* layer_obj = (PoolingLayer*) (layer_objects[i]);
      layer_obj->forward(1.0,0.0,buffer_map["input"], buffer_map["output"]);
    }
  }
}

void seqNetwork::offload_buffer(int layer_number, std::string type)
{
  int bytes,shape[4];
  std::string layer_type = layer_info[layer_number][0];
  std::cout << "Offloading " << layer_type << std::endl;
  if(layer_type=="conv")
  {
    ConvLayer * layer_obj = (ConvLayer*)(layer_objects[layer_number]);
    if(type=="output")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
    else if(type == "workspace")
      bytes = layer_obj->get_total_workspace_size();
  }
  else if(layer_type=="fc")
  {
    FCLayer * layer_obj = (FCLayer*)(layer_objects[layer_number]);
    if(type=="output")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
  }
  else if(layer_type=="flatten")
  {
    std::cout << "Offloading " << layer_type << std::endl;
    Flatten * layer_obj = (Flatten*)(layer_objects[layer_number]);
    if(type=="output")
      bytes = layer_obj->get_output_shape_and_bytes(shape);

  }
  else if(layer_type == "softmax")
  {

    Softmax * layer_obj = (Softmax*)(layer_objects[layer_number]);
    if(type=="output")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
  }
  else if(layer_type == "input")
  {
    InputLayer * layer_obj = (InputLayer*)(layer_objects[layer_number]);
    if(type=="output")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
  }
  else if (layer_type == "maxpool" || layer_type == "avgpool"){
    PoolingLayer * layer_obj = (PoolingLayer*)(layer_objects[layer_number]);
    if(type=="output")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
  }

  if(layer_offloaded_buffers[layer_number][type] == nullptr)
    layer_offloaded_buffers[layer_number][type] = (float*)malloc(bytes);

  cudaMemcpy(layer_offloaded_buffers[layer_number][type],layer_buffers[layer_number][type],bytes,
    cudaMemcpyDeviceToHost);


}

void seqNetwork::prefetch_buffer(int layer_number,std::string type)
{

}
