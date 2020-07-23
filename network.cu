#include "layers/pooling_layer.h"
#include "layers/input_layer.h"
#include "layers/conv_layer.h"
#include "layers/softmax_layer.h"
#include "layers/fc_layer.h"
#include "layers/flatten_layer.h"
#include "layers/layers.h"
#include "layers/relu_layer.h"


using namespace network;
using namespace layers;

seqNetwork::seqNetwork(cudnnHandle_t cudnn,cublasHandle_t cublas,std::vector<std::string> &specs,float lr, unsigned max_allowed_bytes,int sub_batch_selection=0)
{
  /*
  Specs is a vector of strings specifying the Neural Network.
  Input -> "input N H W C"
  Conv ->  "conv H W C"
  */

  num_layers = specs.size();
  handle = cudnn;
  blas_handle = cublas;
  this->lr = lr;
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

  max_allowed_bytes_ = max_allowed_bytes;
  max_sub_batch_size_ = atoi(layer_info[0][1].c_str());
  min_seqnet_bytes_ = getMemoryLowerBound_();

  make_nn_objs(max_sub_batch_size_);
  max_seqnet_memory_ = get_total_memory_();

  if(sub_batch_selection == 0)
    sub_batch_size_ = max_sub_batch_size_;
  else
   sub_batch_size_ = calculate_sub_batch();

  std::cout << "=============Subbatch size : ================\n" << sub_batch_size_ << std::endl;
  make_nn_objs(sub_batch_size_);
  total_seqnet_bytes_ = get_total_memory_();

  gpuErrchk(cudaStreamCreate(&memory_stream_));
  gpuErrchk(cudaStreamCreate(&compute_stream_));
  // std::cout << "Compute stream - "<<compute_stream_ << " Memory stream - " << memory_stream_ << std::endl;
  checkCUDNN(cudnnSetStream(cudnn,compute_stream_));
  checkCUBLAS(cublasSetStream(cublas,compute_stream_));

}

unsigned seqNetwork::get_max_memory() {
  return max_seqnet_memory_;
}

unsigned seqNetwork::getMemoryLowerBound() {
  return min_seqnet_bytes_;
}

unsigned seqNetwork::getMemoryLowerBound_() {
  make_nn_objs(1);
  std::string buff_type;
  int buff_bytes;
  int fw_bytes, bw_bytes=0;
  int min_memory = 0;
  for(int i=0;i<num_layers;i++)
  {
    buff_type = layer_info[i][0];
    fw_bytes = bw_bytes = 0;
    std::cout << layer_info[i][0] << std::endl;

    if(buff_type=="input") {
      fw_bytes += layer_buffer_bytes[i]["output"];
      bw_bytes += layer_buffer_bytes[i]["doutput"];
    }
    else if(buff_type=="conv")
    {
      fw_bytes += layer_buffer_bytes[i]["input"] + layer_buffer_bytes[i]["workspace"] + layer_buffer_bytes[i]["output"];
      bw_bytes += layer_buffer_bytes[i]["doutput"]+layer_buffer_bytes[i]["workspace"] + layer_buffer_bytes[i]["input"]+ layer_buffer_bytes[i]["dinput"];
    }
    else if(buff_type=="fc")
    {
      fw_bytes += layer_buffer_bytes[i]["input"]+layer_buffer_bytes[i]["output"];
      bw_bytes += layer_buffer_bytes[i]["input"] +layer_buffer_bytes[i]["dinput"]+ layer_buffer_bytes[i]["doutput"];
    }
    else if(buff_type == "softmax")
    {
      fw_bytes += layer_buffer_bytes[i]["input"] + layer_buffer_bytes[i]["output"];
      bw_bytes += layer_buffer_bytes[0]["labels"]+layer_buffer_bytes[i]["dinput"]+layer_buffer_bytes[i]["output"];
    }
    else if(buff_type == "maxpool" || buff_type == "avgpool"){
      fw_bytes += layer_buffer_bytes[i]["input"] + layer_buffer_bytes[i]["output"];
      bw_bytes += layer_buffer_bytes[i]["output"]+ layer_buffer_bytes[i]["doutput"] +layer_buffer_bytes[i]["input"]+ layer_buffer_bytes[i]["dinput"];
    }
    else if(buff_type=="relu")
    {
      fw_bytes += layer_buffer_bytes[i]["input"] + layer_buffer_bytes[i]["output"];
      bw_bytes += layer_buffer_bytes[i]["input"]+layer_buffer_bytes[i]["output"]+layer_buffer_bytes[i]["dinput"]+layer_buffer_bytes[i]["doutput"];
    }

    std::cout << buff_type << " fw_bytes :  " << fw_bytes << " bw_bytes :  "<< bw_bytes << std::endl;

    if (fw_bytes > min_memory) {
      min_memory = fw_bytes;
    }
    if (bw_bytes > min_memory) {
      min_memory = bw_bytes;
    }
  }
  std::cout << "Weights memory : " << weights_memory_bytes_ << std::endl;
  std::cout << "Minimum memory requirement : " << min_memory + weights_memory_bytes_ << std::endl;
  return min_memory + weights_memory_bytes_;
}

unsigned seqNetwork::sub_batch_size() {
  return sub_batch_size_;
}

bool seqNetwork::profile_subbatch_validity(unsigned batch_size) {
  make_nn_objs(batch_size);
  int alphaT = ceil(0.15*2*num_layers);
  std::queue<unsigned> window_layers_bytes;
  unsigned running_window_bytes = weights_memory_bytes_;
  unsigned max_memory_requirement=0, temp, old_index_bytes;
  int index;
  bool is_already_present = false;
  std::string buff_type;

  std::cout << "alpha T : " << alphaT << std::endl;
  for(int i = 0; i < num_layers; i++) {
    buff_type = layer_info[i][0];
    temp=0;
    if(buff_type=="input") {
      temp = layer_buffer_bytes[i]["output"];
    } else if(buff_type=="conv"){
      temp = layer_buffer_bytes[i]["workspace"] + layer_buffer_bytes[i]["output"];
    } else if(buff_type=="fc") {
      temp = layer_buffer_bytes[i]["output"];
    } else if(buff_type == "softmax") {
      temp = layer_buffer_bytes[i]["output"];
    } else if(buff_type == "maxpool" || buff_type == "avgpool") {
      temp = layer_buffer_bytes[i]["output"];
    } else if(buff_type=="relu") {
      temp = layer_buffer_bytes[i]["output"];
    }

    if (i <= alphaT) {
      running_window_bytes += temp;
    } else {
      old_index_bytes = window_layers_bytes.front();
      window_layers_bytes.pop();
      running_window_bytes = running_window_bytes - old_index_bytes + temp;
    }
    if (running_window_bytes > max_memory_requirement) {
      max_memory_requirement = running_window_bytes;
    }
    window_layers_bytes.push(temp);
  }

  // Backward Pass
  for(int i = num_layers-1; i > 0; i--) {
    temp=0;
    buff_type = layer_info[i][0];
    if (2*num_layers -1 - alphaT < 2*i) {
      is_already_present = true;
      std::cout << buff_type << " Is already present\n";
    }
    if(buff_type=="conv") {
      temp = layer_buffer_bytes[i]["workspace"] +  layer_buffer_bytes[i]["dinput"];
    }
    else if(buff_type=="fc") {
      temp = layer_buffer_bytes[i]["dinput"]+ layer_buffer_bytes[i]["doutput"];
    }
    else if(buff_type == "softmax") {
      temp = layer_buffer_bytes[0]["labels"]+layer_buffer_bytes[i]["dinput"];
    }
    else if(buff_type == "maxpool" || buff_type == "avgpool"){
      temp =  layer_buffer_bytes[i]["dinput"];
    }
    else if(buff_type=="relu") {
      temp = layer_buffer_bytes[i]["dinput"];
    }
    if (!is_already_present && buff_type != "softmax" ) {
      temp += layer_buffer_bytes[i]["input"];
    }

    old_index_bytes = window_layers_bytes.front();
    window_layers_bytes.pop();
    running_window_bytes = running_window_bytes - old_index_bytes + temp;

    if (running_window_bytes > max_memory_requirement) {
      max_memory_requirement = running_window_bytes;
    }
    window_layers_bytes.push(temp);
    is_already_present = false;
  }

  //std::cout << "Max memory requirement for batch_size : " << batch_size << " : " << max_memory_requirement << std::endl;
  return (max_memory_requirement < max_allowed_bytes_);

}

unsigned seqNetwork::calculate_sub_batch() {
  unsigned lower = 1;
  unsigned upper = max_sub_batch_size_;
  unsigned index, power_count = 0, result = 1;
  while (upper > lower) {
    index = (upper + lower)/2 + (upper + lower)%2;
    if (profile_subbatch_validity(index)) {
      lower = index;
    } else {
      upper = index-1;
    }
  }
  std::cout << "Profiled Subbatch size : " << lower << std::endl;

  // Regularization
  while(lower != 1) {
    lower = lower >> 1;
    result = result << 1;
  }

  std::cout << "Profiled Subbatch size (regularized): " << result << std::endl;

  return result;
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

int seqNetwork::get_total_memory() {
  return total_seqnet_bytes_;
}

int seqNetwork::get_total_memory_()
{
  std::map<std::string,int>::iterator it;
  std::string buff_type;
  int buff_bytes;
  int total_bytes=0;
  for(int i=0;i<num_layers;i++)
  {
    std::cout << layer_info[i][0] << std::endl;
    it = layer_buffer_bytes[i].begin();
    while(it!=layer_buffer_bytes[i].end())
    {
      buff_type = it->first;
      buff_bytes = it->second;
      if (buff_type != "input" && buff_type != "dinput") {
        total_bytes += buff_bytes;
        std::cout << buff_type << " " << buff_bytes << std::endl;
      }
      it++;
    }
  }
  std::cout << "Total memory of network : " << total_bytes << std::endl;
  return total_bytes;

}

void seqNetwork::make_nn_objs(unsigned sub_batch_size)
{
  std::string layer_type;
  int shape[4],batch_size,rows,columns,channels,num_classes;
  int kernel_rows,kernel_cols,kernel_channels,bytes;
  int input_height,output_height;
  int window_height, window_width;
  int vertical_stride, horizontal_stride;
  padding_type pad;
  cudnnPoolingMode_t pooling_type;
  weights_memory_bytes_ = 0;
  //std::cout << "Allocating memory for the Neural Network" << std::endl;
  layer_buffers.resize(num_layers);
  layer_offloaded_buffers.resize(num_layers);
  layer_buffer_bytes.resize(num_layers);
  layer_buffer_redundant_bytes.resize(num_layers);
  layer_objects.clear();

  for(int i=0;i<num_layers;i++)
  {
    layer_type = layer_info[i][0];
    //std::cout << "Layer "<<i+1<<" : "<<layer_type << std::endl;
    layer_offloaded_buffers[i] = init_buffer_map();
    layer_buffers[i] = init_buffer_map();
    if(layer_type == "input")
    {
      batch_size = sub_batch_size;
      rows = atoi(layer_info[i][2].c_str());
      columns = atoi(layer_info[i][3].c_str());
      channels = atoi(layer_info[i][4].c_str());
      num_classes = atoi(layer_info[i][5].c_str());
      this->batch_size = batch_size;

      //std::cout << "Setting up input layer - "<< batch_size <<" " << rows << " "<<columns <<" "<<channels << std::endl;

      InputLayer * new_ip = new InputLayer(batch_size,rows,columns,channels,num_classes);
      layer_objects.push_back(new_ip);

      bytes = new_ip->get_output_shape_and_bytes(shape);
      //layer_buffers[i] = init_buffer_map();


      layer_buffer_redundant_bytes[i]["output"] = layer_buffer_bytes[i]["output"]=bytes;//cudaMalloc(&(layer_buffers[i]["output"]),bytes);
      layer_buffer_redundant_bytes[i]["doutput"] =layer_buffer_bytes[i]["doutput"]=bytes;//cudaMalloc(&(layer_buffers[i]["doutput"]),bytes);




      layer_buffers[i]["labels"] = nullptr;
      layer_buffer_redundant_bytes[i]["labels"] = layer_buffer_bytes[i]["labels"]=batch_size*sizeof(int);//cudaMalloc(&(layer_buffers[i]["labels"]),batch_size*sizeof(int));



      //std::cout << "finished with input layer" << std::endl;
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

      //std::cout << "Setting up conv layer - "<< batch_size <<" " << kernel_rows << " "<< kernel_cols <<" "<< kernel_channels << std::endl;

      ConvLayer * new_conv = new ConvLayer(handle,batch_size,rows,columns,channels,kernel_rows,kernel_cols,kernel_channels,VALID);

      bytes =  new_conv->get_output_shape_and_bytes(shape);

      layer_objects.push_back(new_conv);

      //layer_buffers[i] = init_buffer_map();

      layer_buffer_redundant_bytes[i]["output"] = layer_buffer_bytes[i]["output"]=bytes;//cudaMalloc(&(layer_buffers[i]["output"]),bytes);
      layer_buffer_redundant_bytes[i]["doutput"] =layer_buffer_bytes[i]["doutput"]=bytes;//cudaMalloc(&(layer_buffers[i]["doutput"]),bytes);



      layer_buffer_bytes[i]["input"] = layer_buffer_bytes[i-1]["output"];
      layer_buffer_bytes[i]["dinput"] = layer_buffer_bytes[i-1]["doutput"];



      //new_conv -> allocate_internal_mem(&(layer_buffers[i]["params"]),(void**)&(layer_buffers[i]["workspace"]),&(layer_buffers[i]["dparams"]));
      layer_buffer_bytes[i]["params"] = new_conv -> get_params_shape_and_bytes(shape);
      layer_buffer_bytes[i]["dparams"] = layer_buffer_bytes[i]["params"];
      layer_buffer_bytes[i]["workspace"] = new_conv -> get_total_workspace_size();

      weights_memory_bytes_ += layer_buffer_bytes[i]["params"];
      weights_memory_bytes_ += layer_buffer_bytes[i]["dparams"];

      bytes =  new_conv->get_input_shape_and_bytes(shape);
      layer_buffer_redundant_bytes[i]["input"] = layer_buffer_redundant_bytes[i]["dinput"] = bytes;
      layer_buffer_redundant_bytes[i]["workspace"] = new_conv -> get_total_workspace_size();

    }
    else if(layer_type == "flatten")
    {
      this->get_output_shape(shape,i-1);

      batch_size = shape[0];
      rows = shape[1];
      columns = shape[2];
      channels = shape[3];
      //std::cout << "Setting up flatten layer - "<< batch_size <<" " << rows << " "<<columns <<" "<<channels << std::endl;

      Flatten * new_flat = new Flatten(batch_size,rows,columns,channels);
      layer_objects.push_back(new_flat);

      // layer_buffers[i]["input"] = layer_buffers[i-1]["output"];
      // layer_buffers[i]["dinput"] = layer_buffers[i-1]["doutput"];
      // layer_buffers[i]["output"] = layer_buffers[i]["input"];
      // layer_buffers[i]["doutput"] = layer_buffers[i]["dinput"];

      layer_buffer_bytes[i]["output"]=0;
      layer_buffer_bytes[i]["doutput"]=0;

      bytes =  new_flat->get_input_shape_and_bytes(shape);
      layer_buffer_redundant_bytes[i]["input"] = layer_buffer_redundant_bytes[i]["dinput"] = bytes;

      bytes =  new_flat->get_output_shape_and_bytes(shape);
      layer_buffer_redundant_bytes[i]["output"] = bytes;
      layer_buffer_redundant_bytes[i]["doutput"] = bytes;

    }
    else if(layer_type == "fc")
    {
      this->get_output_shape(shape,i-1);

      batch_size = shape[0];
      input_height = shape[1];
      output_height = atoi(layer_info[i][1].c_str());

      //std::cout << "Setting up fc layer - "<< batch_size <<" " << input_height << std::endl;

      FCLayer * new_fc = new FCLayer(blas_handle,batch_size,input_height,output_height);

      bytes =  new_fc->get_output_shape_and_bytes(shape);



      layer_buffer_bytes[i]["output"]=bytes;//cudaMalloc(&(layer_buffers[i]["output"]),bytes);
      layer_buffer_bytes[i]["doutput"]=bytes;//cudaMalloc(&(layer_buffers[i]["output"]),bytes);


      layer_buffer_redundant_bytes[i]["output"] = bytes;
      layer_buffer_redundant_bytes[i]["doutput"] = bytes;

      layer_buffer_bytes[i]["input"] = layer_buffer_bytes[i-1]["output"];
      layer_buffer_bytes[i]["dinput"] = layer_buffer_bytes[i-1]["doutput"];

      bytes =  new_fc->get_input_shape_and_bytes(shape);
      layer_buffer_redundant_bytes[i]["input"] = layer_buffer_redundant_bytes[i]["dinput"] = bytes;


      //new_fc -> allocate_internal_mem(&(layer_buffers[i]["params"]),&(layer_buffers[i]["dparams"]));
      layer_buffer_bytes[i]["params"] = new_fc -> get_params_shape_and_bytes(shape);
      layer_buffer_bytes[i]["dparams"] = new_fc -> get_params_shape_and_bytes(shape);

      weights_memory_bytes_ += layer_buffer_bytes[i]["params"];
      weights_memory_bytes_ += layer_buffer_bytes[i]["dparams"];

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

      //layer_buffers[i] = init_buffer_map();

      layer_buffer_bytes[i]["output"]=bytes;//cudaMalloc(&(layer_buffers[i]["output"]),bytes);
      layer_buffer_bytes[i]["doutput"]=bytes;//cudaMalloc(&(layer_buffers[i]["output"]),bytes);


      layer_buffer_redundant_bytes[i]["output"] = bytes;
      layer_buffer_redundant_bytes[i]["doutput"] = bytes;

      layer_buffer_bytes[i]["input"] = layer_buffer_bytes[i-1]["output"];
      layer_buffer_bytes[i]["dinput"] = layer_buffer_bytes[i-1]["doutput"];

      bytes =  new_softmax->get_input_shape_and_bytes(shape);
      layer_buffer_redundant_bytes[i]["input"] = layer_buffer_redundant_bytes[i]["dinput"] = bytes;

    }
    else if(layer_type == "relu")
    {
      this->get_output_shape(shape,i-1);
      batch_size = shape[0];
      channels = shape[1];
      rows = shape[2];
      columns = shape[3];

      relu * new_relu =  new relu(handle,batch_size,channels,rows,columns);
      bytes = new_relu->get_output_shape_and_bytes(shape);

      layer_objects.push_back(new_relu);


      layer_buffer_bytes[i]["output"]=bytes;//cudaMalloc(&(layer_buffers[i]["output"]),bytes);
      layer_buffer_bytes[i]["doutput"]=bytes;//cudaMalloc(&(layer_buffers[i]["output"]),bytes);


      layer_buffer_redundant_bytes[i]["output"] = bytes;
      layer_buffer_redundant_bytes[i]["doutput"] = bytes;

      layer_buffer_bytes[i]["input"] = layer_buffer_bytes[i-1]["output"];
      layer_buffer_bytes[i]["dinput"] = layer_buffer_bytes[i-1]["doutput"];

      bytes =  new_relu->get_input_shape_and_bytes(shape);
      layer_buffer_redundant_bytes[i]["input"] = layer_buffer_redundant_bytes[i]["dinput"] = bytes;


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

      //std::cout << "Setting up pooling layer - "<< batch_size <<" " << rows << " "<<columns <<" "<<channels << std::endl;

      PoolingLayer* new_pooling = new PoolingLayer(&handle,
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

      layer_buffer_bytes[i]["output"]=bytes;//cudaMalloc(&(layer_buffers[i]["output"]),bytes);
      layer_buffer_bytes[i]["doutput"]=bytes;//cudaMalloc(&(layer_buffers[i]["output"]),bytes);


      layer_buffer_redundant_bytes[i]["output"] = bytes;
      layer_buffer_redundant_bytes[i]["doutput"] = bytes;

      layer_buffer_bytes[i]["dinput"] = layer_buffer_bytes[i-1]["doutput"];
      layer_buffer_bytes[i]["input"] = layer_buffer_bytes[i-1]["output"];

      layer_objects.push_back(new_pooling);

      bytes =  new_pooling->get_input_shape_and_bytes(shape);
      layer_buffer_redundant_bytes[i]["input"] = layer_buffer_redundant_bytes[i]["dinput"] = bytes;
    }

  }
}


void seqNetwork::link_all_buffers()
{
  for(int i=1;i<num_layers;i++)
  {
    layer_buffers[i]["input"] = layer_buffers[i-1]["output"];
    layer_buffers[i]["dinput"] = layer_buffers[i-1]["doutput"];

    if(layer_info[i][0]=="flatten")
    {
        layer_buffers[i]["output"] = layer_buffers[i]["input"];
        layer_buffers[i]["doutput"] = layer_buffers[i]["dinput"];
    }

  }
}

void seqNetwork::randomise_batch()
{
  ((InputLayer*)layer_objects[0])->randomly_populate(layer_buffers[0]["output"],layer_buffers[0]["labels"]);
}

void seqNetwork::update_batch(float* data, int* labels)
{
  batch_data_ = data;
  batch_labels_ = labels;
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

void seqNetwork::train() {
  int loops = max_sub_batch_size_/sub_batch_size_;
  int shape[4];
  ((InputLayer*)layer_objects[0]) -> get_output_shape_and_bytes(shape);
  int offset = shape[0]*shape[1]*shape[2]*shape[3];
  //std::cout << "Offset " << offset << std::endl;


  ((InputLayer*)layer_objects[0])->update_batch((batch_data_), (float*)(batch_labels_),layer_buffers[0]["output"],layer_buffers[0]["labels"]);
  forward_();
  backward_(0.0);
  //std::cout << "Number of Loops" << loops << std::endl;
  for (int i = 1; i < loops; i++) {
    ((InputLayer*)layer_objects[0])->update_batch((batch_data_ + i*(offset)), (float*)(batch_labels_+i*sub_batch_size_),layer_buffers[0]["output"],layer_buffers[0]["labels"]);
    forward_();
    backward_(1.0);

  }
}

int seqNetwork::get_loops()
{
  return max_sub_batch_size_/sub_batch_size_;
}

void seqNetwork::enqueue_batch_loop(int loop_no)
{
  int i = loop_no;
  int shape[4];
  ((InputLayer*)layer_objects[0]) -> get_output_shape_and_bytes(shape);
  int offset = shape[0]*shape[1]*shape[2]*shape[3];
  ((InputLayer*)layer_objects[0])->update_batch((batch_data_ + i*(offset)), (float*)(batch_labels_+i*sub_batch_size_),layer_buffers[0]["output"],layer_buffers[0]["labels"]);
}

void seqNetwork::forward() {
  int loops = max_sub_batch_size_/sub_batch_size_;
  int shape[4];
  ((InputLayer*)layer_objects[0]) -> get_output_shape_and_bytes(shape);
  int offset = shape[0]*shape[1]*shape[2]*shape[3];
  for (int i = 0; i < loops; i++) {
    ((InputLayer*)layer_objects[0])->update_batch((batch_data_ + i*offset), (float*)(batch_labels_+i*sub_batch_size_),layer_buffers[0]["output"],layer_buffers[0]["labels"]);
    forward_();
  }
}

void seqNetwork::forward_()
{
  for(int i=0;i<num_layers;i++)
  {
    //std::cout << "Forward " << i << " " << layer_info[i][0] << std::endl;
    std::map<std::string,float*> buffer_map = layer_buffers[i];
    std::string layer_type = layer_info[i][0];

    //cudaDeviceSynchronize();
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
    else if(layer_type=="relu")
    {
      relu * layer_obj = (relu*)(layer_objects[i]);
      layer_obj -> forward(buffer_map["input"],buffer_map["output"]);
    }

  }


}

void seqNetwork::forward_layer(int layer_number)
{
  int i = layer_number;
  //std::cout << "Forward " << i << " " << layer_info[i][0] << std::endl;
  std::map<std::string,float*> buffer_map = layer_buffers[i];
  std::string layer_type = layer_info[i][0];

  //cudaDeviceSynchronize();
  if(layer_type=="input")return;
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
  else if(layer_type=="relu")
  {
    relu * layer_obj = (relu*)(layer_objects[i]);
    layer_obj -> forward(buffer_map["input"],buffer_map["output"]);
  }




}

void seqNetwork::backward() {
  int loops = max_sub_batch_size_/sub_batch_size_;
  backward_(0);
  for (int i = 1; i < loops; i++) {
    backward_(1);
  }
}

 void seqNetwork::backward_(float beta)
{

  for(int i=num_layers-1;i>=0;i--)
  {
    std::map<std::string,float*> buffer_map = layer_buffers[i];
    std::string layer_type = layer_info[i][0];
    //cudaDeviceSynchronize();
    //std::cout << "Backward " << i << " " << layer_info[i][0] << std::endl;
    if(layer_type=="input")continue;
    else if(layer_type=="conv")
    {
      ConvLayer * layer_obj = (ConvLayer*)(layer_objects[i]);
      layer_obj -> backward(1.0,beta,0.0,buffer_map["doutput"],(void*)buffer_map["workspace"], buffer_map["params"], buffer_map["input"], buffer_map["dinput"], buffer_map["dparams"],lr);
    }
    else if(layer_type=="fc")
    {
      FCLayer * layer_obj = (FCLayer*)(layer_objects[i]);
      layer_obj -> backward(1.0,beta,0.0,buffer_map["input"], buffer_map["params"],buffer_map["dparams"],buffer_map["dinput"], buffer_map["doutput"],lr);
    }
    else if(layer_type == "softmax")
    {
      Softmax* layer_obj = (Softmax*)(layer_objects[i]);
      layer_obj -> backward((int*)layer_buffers[0]["labels"],buffer_map["dinput"],buffer_map["output"]);
      //gradients are stored in buffer_map["labels"]
    }
    else if(layer_type == "maxpool" || layer_type == "avgpool")
    {
      PoolingLayer* layer_obj = (PoolingLayer*) (layer_objects[i]);
      layer_obj->backward(1.0,0.0,buffer_map["output"], buffer_map["doutput"] ,buffer_map["input"], buffer_map["dinput"]);
    }
    else if(layer_type=="relu")
    {
      relu * layer_obj = (relu*)(layer_objects[i]);
      layer_obj -> backward(buffer_map["input"],buffer_map["output"],buffer_map["dinput"],buffer_map["doutput"]);
    }
  }
}

void seqNetwork::backward_layer(int layer_number,float beta)
{
   int i = layer_number;
   std::map<std::string,float*> buffer_map = layer_buffers[i];
   std::string layer_type = layer_info[i][0];
   //cudaDeviceSynchronize();
   //std::cout << "Backward " << i << " " << layer_info[i][0] << std::endl;
   if(layer_type=="input")return;
   else if(layer_type=="conv")
   {
     ConvLayer * layer_obj = (ConvLayer*)(layer_objects[i]);
     layer_obj -> backward(1.0,beta,0.0,buffer_map["doutput"],(void*)buffer_map["workspace"], buffer_map["params"], buffer_map["input"], buffer_map["dinput"], buffer_map["dparams"],lr);
   }
   else if(layer_type=="fc")
   {
     FCLayer * layer_obj = (FCLayer*)(layer_objects[i]);
     layer_obj -> backward(1.0,beta,0.0,buffer_map["input"], buffer_map["params"],buffer_map["dparams"],buffer_map["dinput"], buffer_map["doutput"],lr);
   }
   else if(layer_type == "softmax")
   {
     Softmax* layer_obj = (Softmax*)(layer_objects[i]);
     layer_obj -> backward((int*)layer_buffers[0]["labels"],buffer_map["dinput"],buffer_map["output"]);
     //gradients are stored in buffer_map["labels"]
   }
   else if(layer_type == "maxpool" || layer_type == "avgpool")
   {
     PoolingLayer* layer_obj = (PoolingLayer*) (layer_objects[i]);
     layer_obj->backward(1.0,0.0,buffer_map["output"], buffer_map["doutput"] ,buffer_map["input"], buffer_map["dinput"]);
   }
   else if(layer_type=="relu")
   {
     relu * layer_obj = (relu*)(layer_objects[i]);
     layer_obj -> backward(buffer_map["input"],buffer_map["output"],buffer_map["dinput"],buffer_map["doutput"]);
   }

}

void seqNetwork::update_weights() {
  for(int i=num_layers-1;i>=0;i--)
  {
    std::map<std::string,float*> buffer_map = layer_buffers[i];
    std::string layer_type = layer_info[i][0];
    //cudaDeviceSynchronize();
    if(layer_type=="conv")
    {
      ConvLayer * layer_obj = (ConvLayer*)(layer_objects[i]);
      layer_obj -> update_weights( buffer_map["params"], buffer_map["dparams"],lr,compute_stream_);
    }
    else if(layer_type=="fc")
    {
      FCLayer * layer_obj = (FCLayer*)(layer_objects[i]);
      layer_obj -> update_weights( buffer_map["params"],buffer_map["dparams"],lr,compute_stream_);
    }

  }
}

float* seqNetwork::offload_buffer(int layer_number, std::string type,int shape[],int async)
{
  int bytes;
  std::string layer_type = layer_info[layer_number][0];
  //std::cout << "Offloading layer number - " << layer_number <<" " <<layer_type <<" "<<type <<std::endl;
  if(layer_type=="conv")
  {
    ConvLayer * layer_obj = (ConvLayer*)(layer_objects[layer_number]);
    if(type=="output" || type=="doutput")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
    else if(type == "workspace")
      bytes = layer_obj->get_total_workspace_size();
    else if(type == "input" || type == "dinput")
      bytes = layer_obj->get_input_shape_and_bytes(shape);

  }
  else if(layer_type=="fc")
  {
    FCLayer * layer_obj = (FCLayer*)(layer_objects[layer_number]);
    if(type=="output" || type=="doutput")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
    else if(type == "input" || type == "dinput")
      bytes = layer_obj->get_input_shape_and_bytes(shape);
    else if(type=="params" || type == "dparams")
      bytes = layer_obj -> get_params_shape_and_bytes(shape);

  }
  else if(layer_type=="flatten")
  {

    Flatten * layer_obj = (Flatten*)(layer_objects[layer_number]);
    if(type=="output" || type=="doutput")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
    else if(type == "input" || type == "dinput")
      bytes = layer_obj->get_input_shape_and_bytes(shape);

  }
  else if(layer_type == "softmax")
  {

    Softmax * layer_obj = (Softmax*)(layer_objects[layer_number]);
    if(type=="output" || type=="doutput")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
    else if(type == "input" || type == "dinput")
      bytes = layer_obj->get_input_shape_and_bytes(shape);
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
  else if(layer_type == "relu")
  {
    relu * layer_obj = (relu*)(layer_objects[layer_number]);
    if(type == "output" || type == "doutput")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
    else if(type=="input" || type == "dinput")
      bytes = layer_obj->get_input_shape_and_bytes(shape);
  }

  if(layer_offloaded_buffers[layer_number][type] == nullptr){
    //std::cout << "Allocating bytes to the layer buffer " << layer_number <<" " << type<<std::endl;
    //layer_offloaded_buffers[layer_number][type] = (float*)malloc(bytes);
    gpuErrchk(cudaMallocHost((void**)&(layer_offloaded_buffers[layer_number][type]), bytes));
  }

  if(async)
  {
    gpuErrchk(cudaMemcpyAsync(layer_offloaded_buffers[layer_number][type],layer_buffers[layer_number][type],bytes,
      cudaMemcpyDeviceToHost, memory_stream_));
  }
  else
  {
    gpuErrchk(cudaMemcpy(layer_offloaded_buffers[layer_number][type],layer_buffers[layer_number][type],bytes,
      cudaMemcpyDeviceToHost));
  }


  if(type == "output" && layer_number < num_layers-1)
    layer_offloaded_buffers[layer_number+1]["input"] = layer_offloaded_buffers[layer_number]["output"];
  if(type == "doutput" && layer_number < num_layers-1)
    layer_offloaded_buffers[layer_number+1]["dinput"] = layer_offloaded_buffers[layer_number]["doutput"];

  if(type == "input" && layer_number > 0)
    layer_offloaded_buffers[layer_number-1]["output"] = layer_offloaded_buffers[layer_number]["input"];
  if(type == "dinput" && layer_number > 0)
    layer_offloaded_buffers[layer_number-1]["doutput"] = layer_offloaded_buffers[layer_number]["dinput"];

  if(layer_type=="flatten")
  {
    if(type == "output")
    {
      layer_offloaded_buffers[layer_number]["input"] = layer_offloaded_buffers[layer_number]["output"];
      if(layer_number>0)layer_offloaded_buffers[layer_number-1]["output"] = layer_offloaded_buffers[layer_number]["input"];
    }
    if(type == "input")
    {
      layer_offloaded_buffers[layer_number]["output"] = layer_offloaded_buffers[layer_number]["input"];
      if(layer_number+1<num_layers)layer_offloaded_buffers[layer_number+1]["input"] = layer_offloaded_buffers[layer_number]["output"];
    }
    if(type == "doutput")
    {
      layer_offloaded_buffers[layer_number]["dinput"] = layer_offloaded_buffers[layer_number]["doutput"];
      if(layer_number>0)layer_offloaded_buffers[layer_number-1]["doutput"] = layer_offloaded_buffers[layer_number]["dinput"];
    }
    if(type == "dinput")
    {
      layer_offloaded_buffers[layer_number]["doutput"] = layer_offloaded_buffers[layer_number]["dinput"];
      if(layer_number+1<num_layers)layer_offloaded_buffers[layer_number+1]["dinput"] = layer_offloaded_buffers[layer_number]["doutput"];
    }
  }

  return layer_offloaded_buffers[layer_number][type];

  // cudaFree(layer_buffers[layer_number][type]);
  // layer_buffers[layer_number][type] = nullptr;


}

float* seqNetwork::prefetch_buffer(int layer_number, std::string type,int shape[])
{
  int bytes;
  std::string layer_type = layer_info[layer_number][0];
  //std::cout << "Prefetching layer number - " << layer_number <<" " <<layer_type <<" "<<type <<std::endl;
  if(layer_type=="conv")
  {
    ConvLayer * layer_obj = (ConvLayer*)(layer_objects[layer_number]);
    if(type=="output" || type=="doutput")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
    else if(type == "workspace")
      bytes = layer_obj->get_total_workspace_size();
    else if(type == "input" || type == "dinput")
      bytes = layer_obj->get_input_shape_and_bytes(shape);

  }
  else if(layer_type=="fc")
  {
    FCLayer * layer_obj = (FCLayer*)(layer_objects[layer_number]);
    if(type=="output" || type=="doutput")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
    else if(type == "input" || type == "dinput")
      bytes = layer_obj->get_input_shape_and_bytes(shape);
    else if(type=="params" || type == "dparams")
      bytes = layer_obj -> get_params_shape_and_bytes(shape);

  }
  else if(layer_type=="flatten")
  {

    Flatten * layer_obj = (Flatten*)(layer_objects[layer_number]);
    if(type=="output" || type=="doutput")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
    else if(type == "input" || type == "dinput")
      bytes = layer_obj->get_input_shape_and_bytes(shape);

  }
  else if(layer_type == "softmax")
  {

    Softmax * layer_obj = (Softmax*)(layer_objects[layer_number]);
    if(type=="output" || type=="doutput")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
    else if(type == "input" || type == "dinput")
      bytes = layer_obj->get_input_shape_and_bytes(shape);
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
  else if(layer_type == "relu")
  {
    relu * layer_obj = (relu*)(layer_objects[layer_number]);
    if(type == "output" || type == "doutput")
      bytes = layer_obj->get_output_shape_and_bytes(shape);
    else if(type=="input" || type == "dinput")
      bytes = layer_obj->get_input_shape_and_bytes(shape);
  }

  assert (layer_offloaded_buffers[layer_number][type] != nullptr); //nonempty source
  assert (layer_buffers[layer_number][type] != nullptr);           //non empty destination

  gpuErrchk(cudaMemcpyAsync(layer_buffers[layer_number][type],layer_offloaded_buffers[layer_number][type],bytes,
    cudaMemcpyHostToDevice, memory_stream_));

  return layer_buffers[layer_number][type];

  // cudaFree(layer_buffers[layer_number][type]);
  // layer_buffers[layer_number][type] = nullptr;


}

void seqNetwork::allocate_all_memory(vmm * mem_manager)
{
  std::map<std::string,int>::iterator it;
  std::string buff_type;
  int buff_bytes;
  int total_bytes=0;
  for(int i=0;i<num_layers;i++)
  {
    //std::cout << layer_info[i][0] << std::endl;
    it = layer_buffer_bytes[i].begin();
    while(it!=layer_buffer_bytes[i].end())
    {
      buff_type = it->first;
      buff_bytes = it->second;
      if ((buff_type != "input" && buff_type != "dinput")){
        total_bytes += buff_bytes;
      //std::cout << buff_type << " " << buff_bytes << std::endl;
        mem_manager->allocate(&layer_buffers[i][buff_type],buff_bytes,layer_info[i][0]+" "+buff_type);
      }
      it++;
    }
  }
  //std::cout << total_bytes << std::endl;
  link_all_buffers();
}

void seqNetwork::allocate_mem_params(vmm * mem_manager)
{
  int bytes;
  for(int i=0;i<num_layers;i++)
  {
    if(layer_info[i][0] == "conv" || layer_info[i][0] == "fc")
    {
      bytes = layer_buffer_bytes[i]["params"];
      mem_manager->allocate(&layer_buffers[i]["params"],bytes,layer_info[i][0]+" params");

      bytes = layer_buffer_bytes[i]["dparams"];
      mem_manager->allocate(&layer_buffers[i]["dparams"],bytes,layer_info[i][0]+" dparams");

    }
  }
}

void seqNetwork::allocate_mem_layer_fw(int layer_number, vmm * mem_manager)
{
  int i = layer_number,bytes;
  int shape[4];

  if(layer_info[i][0]!="flatten")
  {
    assert(layer_buffers[i]["output"] == nullptr);
    bytes = layer_buffer_redundant_bytes[i]["output"];
    mem_manager->allocate(&layer_buffers[i]["output"],bytes,layer_info[i][0]+" layer - output");
  }

  if(layer_info[i][0] == "input")
  {
    //allocate labels memory
    assert(layer_buffers[i]["labels"] == nullptr);
    bytes = layer_buffer_redundant_bytes[i]["labels"];
    mem_manager->allocate(&layer_buffers[i]["labels"],bytes,"input layer - labels");
  }
  if(layer_info[i][0] == "conv")
  {
    assert(layer_buffers[i]["workspace"] == nullptr);
    bytes = layer_buffer_redundant_bytes[i]["workspace"];
    mem_manager->allocate(&layer_buffers[i]["workspace"],bytes,"conv layer - workspace");
  }

}

void seqNetwork::offload_and_call_mem_manager(float ** buff, int bytes, std::string misc, vmm * mem_manager,int layer_number,int offload)
{
  int sync=0;
  std::vector<int>local_pop;
  std::vector<int>global_pop;
  while(mem_manager->allocate(buff,bytes,misc) == INSUFF_MEM)
  {
    //std::cout << "Requested bytes - " << bytes << " by " << misc << " and free bytes - " << mem_manager->freeSize << std::endl;

    assert(!(locally_allocated_layers.empty()) || !(globally_allocated_layers.empty()));

    if(!locally_allocated_layers.empty()){
      int layer_local_deletion = locally_allocated_layers.front();
      if(layer_local_deletion != layer_number-1){
        //std::cout << " Local Mem Deletion of Layer - "<<layer_local_deletion << " - " <<layer_info[layer_local_deletion][0] << std::endl;
        this->deallocate_mem_layer_fw2(layer_local_deletion,mem_manager,1,offload);
        locally_allocated_layers.pop();
      }
      else
      {
        local_pop.push_back(layer_local_deletion);
        locally_allocated_layers.pop();
      }
    }
    //while(!globally_allocated_layers.empty()){
    else{
      sync=1;
      int layer_global_deletion = globally_allocated_layers.front();
      if(layer_global_deletion!=layer_number-2 && layer_global_deletion!= layer_number-1 && layer_global_deletion!=layer_number){
        //std::cout << " Shared Mem (Output) Deletion of Layer - "<<layer_global_deletion << " - " <<layer_info[layer_global_deletion][0] << std::endl;
        this->deallocate_mem_layer_fw2(layer_global_deletion,mem_manager,0,offload);
        globally_allocated_layers.pop();
        this->link_layer_buffer_fw(layer_global_deletion);
      }
      else
      {
        global_pop.push_back(layer_global_deletion);
        globally_allocated_layers.pop();
      }
    }
    //mem_manager->printNodes();
  }

  for(int i=0;i<local_pop.size();i++)locally_allocated_layers.push(local_pop[i]);
  for(int i=0;i<global_pop.size();i++)globally_allocated_layers.push(global_pop[i]);

  if(sync)
    cudaDeviceSynchronize();

}

void seqNetwork::deallocate_mem_layer_fw2(int layer_number, vmm * mem_manager,int local,int offload)
{
  int i = layer_number,bytes;
  int shape[4];

  if(local==0){
    if(i+1 == num_layers || layer_info[i+1][0]!="flatten")
    {
      assert(layer_buffers[i]["output"] != nullptr);
      bytes = layer_buffer_redundant_bytes[i]["output"];
      if(offload == 1)
        offload_buffer(i,"output",shape);
      //cudaDeviceSynchronize();
      mem_manager->deleteMem(layer_buffers[i]["output"]);
      layer_buffers[i]["output"] = nullptr;
      if(layer_info[i][0] == "flatten"){
        layer_buffers[i]["input"] = nullptr;
        if(i>0)
          layer_buffers[i-1]["output"] = nullptr;
      }
    }
    return;
  }

  if(layer_info[i][0] == "input")
  {
    //allocate labels memory
    assert(layer_buffers[i]["labels"] != nullptr);
    bytes = layer_buffer_redundant_bytes[i]["labels"];
    if(offload == 1)
      offload_buffer(i,"labels",shape);
    //cudaDeviceSynchronize();
    mem_manager->deleteMem(layer_buffers[i]["labels"]);
    layer_buffers[i]["labels"] = nullptr;
  }
  if(layer_info[i][0] == "conv")
  {
    //assert(layer_buffers[i]["workspace"] != nullptr);
    if(layer_buffers[i]["workspace"] != nullptr){
      bytes = layer_buffer_redundant_bytes[i]["workspace"];
      mem_manager->deleteMem(layer_buffers[i]["workspace"]);
      layer_buffers[i]["workspace"] = nullptr;
    }
  }

}


void seqNetwork::allocate_mem_layer_fw2(int layer_number, vmm * mem_manager)
{
  int i = layer_number,bytes;
  int shape[4];

  if(layer_info[i][0]!="flatten")
  {
    assert(layer_buffers[i]["output"] == nullptr);
    bytes = layer_buffer_redundant_bytes[i]["output"];
    this->offload_and_call_mem_manager(&layer_buffers[i]["output"],bytes,layer_info[i][0]+" layer - output",mem_manager,layer_number,1);
    //float * buff, int bytes, std::string misc, vmm * mem_manager,int layer_number,int offload
  }

  if(layer_info[i][0] == "input")
  {
    //allocate labels memory
    assert(layer_buffers[i]["labels"] == nullptr);
    bytes = layer_buffer_redundant_bytes[i]["labels"];
    this->offload_and_call_mem_manager(&layer_buffers[i]["labels"],bytes,"input layer - labels",mem_manager,layer_number,1);
    //mem_manager->allocate(&layer_buffers[i]["labels"],bytes,"input layer - labels");
  }
  if(layer_info[i][0] == "conv")
  {
    assert(layer_buffers[i]["workspace"] == nullptr);
    bytes = layer_buffer_redundant_bytes[i]["workspace"];
    this->offload_and_call_mem_manager(&layer_buffers[i]["workspace"],bytes,layer_info[i][0]+"conv layer - workspace",mem_manager,layer_number,1);
    //mem_manager->allocate(&layer_buffers[i]["workspace"],bytes,"conv layer - workspace");
  }
  if(layer_number!=0)
    locally_allocated_layers.push(layer_number);
  globally_allocated_layers.push(layer_number);

}


void seqNetwork::link_layer_buffer_fw(int layer_number)
{
  int i = layer_number;
  if(layer_info[i][0]=="flatten")
    layer_buffers[i]["output"] = layer_buffers[i]["input"];

  if(i < num_layers-1)
    layer_buffers[i+1]["input"] = layer_buffers[i]["output"];

}

void seqNetwork::deallocate_mem_layer_fw(int layer_number, vmm * mem_manager,int local)
{
  int i = layer_number,bytes;
  int shape[4];

  if(local==0){
    if(i+1 == num_layers || layer_info[i+1][0]!="flatten")
    {
      assert(layer_buffers[i]["output"] != nullptr);
      bytes = layer_buffer_redundant_bytes[i]["output"];
      offload_buffer(i,"output",shape);
      cudaDeviceSynchronize();
      mem_manager->deleteMem(layer_buffers[i]["output"]);
      layer_buffers[i]["output"] = nullptr;
      if(layer_info[i][0] == "flatten"){
        layer_buffers[i]["input"] = nullptr;
        if(i>0)
          layer_buffers[i-1]["output"] = nullptr;
      }
    }
    return;
  }

  if(layer_info[i][0] == "input")
  {
    //allocate labels memory
    assert(layer_buffers[i]["labels"] != nullptr);
    bytes = layer_buffer_redundant_bytes[i]["labels"];
    offload_buffer(i,"labels",shape);
    cudaDeviceSynchronize();
    mem_manager->deleteMem(layer_buffers[i]["labels"]);
    layer_buffers[i]["labels"] = nullptr;
  }
  if(layer_info[i][0] == "conv")
  {
    //assert(layer_buffers[i]["workspace"] != nullptr);
    if(layer_buffers[i]["workspace"] != nullptr){
      bytes = layer_buffer_redundant_bytes[i]["workspace"];
      mem_manager->deleteMem(layer_buffers[i]["workspace"]);
      layer_buffers[i]["workspace"] = nullptr;
    }
  }

}

void seqNetwork::allocate_mem_layer_bw(int layer_number, vmm * mem_manager)
{
  int i = layer_number,bytes;
  int shape[4];

  //dinput
  if(layer_info[i][0]!="flatten" && layer_info[i][0]!="input")
  {
    assert(layer_buffers[i]["dinput"] == nullptr);
    bytes = layer_buffer_redundant_bytes[i]["dinput"];
    mem_manager->allocate(&layer_buffers[i]["dinput"],bytes,layer_info[i][0]+" layer - dinput");
  }
  //workspace
  if(layer_info[i][0] == "conv")
  {
    assert(layer_buffers[i]["workspace"] == nullptr);
    bytes = layer_buffer_redundant_bytes[i]["workspace"];
    mem_manager->allocate(&layer_buffers[i]["workspace"],bytes,"conv layer - workspace");
  }
  //output
  if(layer_info[i][0] == "maxpool" || layer_info[i][0] == "avgpool" || layer_info[i][0] == "relu")
  {
    assert(layer_buffers[i]["output"] == nullptr);
    bytes = layer_buffer_redundant_bytes[i]["output"];
    mem_manager->allocate(&layer_buffers[i]["output"],bytes,layer_info[i][0]+ " output");
    //prefetch output
    prefetch_buffer(i,"output",shape);
  }
  //input
  if(layer_info[i][0] == "conv" || layer_info[i][0] == "fc"|| layer_info[i][0] == "relu"||layer_info[i][0] == "maxpool" || layer_info[i][0] == "avgpool")
  {
    assert(layer_buffers[i]["input"] == nullptr);
    bytes = layer_buffer_redundant_bytes[i]["input"];
    mem_manager->allocate(&layer_buffers[i]["input"],bytes,layer_info[i][0]+ " input");
    prefetch_buffer(i,"input",shape);
    //prefetch input
  }
  if(layer_info[i][0] == "conv" || layer_info[i][0] == "fc"|| layer_info[i][0] == "relu"||layer_info[i][0] == "maxpool" || layer_info[i][0] == "avgpool")
    cudaDeviceSynchronize();

}

void seqNetwork::allocate_mem_layer_bw2(int layer_number, vmm * mem_manager)
{
  int i = layer_number,bytes;
  int shape[4];
  int to_sync = 0;

  //dinput
  if(layer_info[i][0]!="flatten" && layer_info[i][0]!="input")
  {
    assert(layer_buffers[i]["dinput"] == nullptr);
    bytes = layer_buffer_redundant_bytes[i]["dinput"];
    //mem_manager->allocate(&layer_buffers[i]["dinput"],bytes,layer_info[i][0]+" layer - dinput");
    offload_and_call_mem_manager(&layer_buffers[i]["dinput"], bytes, layer_info[i][0]+" layer - dinput", mem_manager,layer_number,0);
  }
  //workspace
  if(layer_info[i][0] == "conv")
  {
    if(layer_buffers[i]["workspace"] == nullptr){
      bytes = layer_buffer_redundant_bytes[i]["workspace"];
      //mem_manager->allocate(&layer_buffers[i]["workspace"],bytes,"conv layer - workspace");
      offload_and_call_mem_manager(&layer_buffers[i]["workspace"], bytes, "conv layer - workspace", mem_manager,layer_number,0);
    }
  }
  //output
  if(layer_info[i][0] == "maxpool" || layer_info[i][0] == "avgpool" || layer_info[i][0] == "relu")
  {
    if(layer_buffers[i]["output"] == nullptr){
      bytes = layer_buffer_redundant_bytes[i]["output"];
      //mem_manager->allocate(&layer_buffers[i]["output"],bytes,layer_info[i][0]+ " output");
      offload_and_call_mem_manager(&layer_buffers[i]["output"], bytes, layer_info[i][0]+ " output", mem_manager,layer_number,0);
      //std::cout << "Prefetching output " << std::endl;
      prefetch_buffer(i,"output",shape);
      to_sync = 1;
    }
  }
  //input
  if(layer_info[i][0] == "conv" || layer_info[i][0] == "fc"|| layer_info[i][0] == "relu"||layer_info[i][0] == "maxpool" || layer_info[i][0] == "avgpool")
  {
    if(layer_buffers[i]["input"] == nullptr){
      bytes = layer_buffer_redundant_bytes[i]["input"];
      //mem_manager->allocate(&layer_buffers[i]["input"],bytes,layer_info[i][0]+ " input");
      offload_and_call_mem_manager(&layer_buffers[i]["input"], bytes, layer_info[i][0]+ " input", mem_manager,layer_number,0);
      //std::cout << "Prefetching input " << std::endl;
      prefetch_buffer(i,"input",shape);
      //prefetch input
      to_sync = 1;
    }
  }
  //if(layer_info[i][0] == "conv" || layer_info[i][0] == "fc"|| layer_info[i][0] == "relu"||layer_info[i][0] == "maxpool" || layer_info[i][0] == "avgpool")

  cudaDeviceSynchronize();

}

void seqNetwork::allocate_mem_layer_bw_h1(int layer_number, vmm * mem_manager)
{
  int i = layer_number,bytes;
  int shape[4];

  int alphaT = ceil(0.15*2*num_layers);
  if (layer_number == num_layers-1) {
    prefetch_trigger_layer_no_ = layer_number;
    last_prefetched_layer_no_ = num_layers;
  }

  if (layer_number == prefetch_trigger_layer_no_) {
    prefetch_trigger_layer_no_ = last_prefetched_layer_no_-1 - (alphaT/2);
    if (layer_number == num_layers-1)
      sync_layer_no_ = last_prefetched_layer_no_ - 2;
    else
      sync_layer_no_ = last_prefetched_layer_no_ - 1;

    for (int i = last_prefetched_layer_no_-1; i >= 0 && i > last_prefetched_layer_no_-1-alphaT; i--) {
    //dinput
      if(layer_info[i][0]!="flatten" && layer_info[i][0]!="input")
      {
        assert(layer_buffers[i]["dinput"] == nullptr);
        bytes = layer_buffer_redundant_bytes[i]["dinput"];
        mem_manager->allocate(&layer_buffers[i]["dinput"],bytes,layer_info[i][0]+" layer - dinput");
      }
      //workspace
      if(layer_info[i][0] == "conv")
      {
        assert(layer_buffers[i]["workspace"] == nullptr);
        bytes = layer_buffer_redundant_bytes[i]["workspace"];
        mem_manager->allocate(&layer_buffers[i]["workspace"],bytes,"conv layer - workspace");
      }
      //output
      if(layer_info[i][0] == "maxpool" || layer_info[i][0] == "avgpool" || layer_info[i][0] == "relu")
      {
        assert(layer_buffers[i]["output"] == nullptr);
        bytes = layer_buffer_redundant_bytes[i]["output"];
        mem_manager->allocate(&layer_buffers[i]["output"],bytes,layer_info[i][0]+ " output");
        //prefetch output
        prefetch_buffer(i,"output",shape);
      }
      //input
      if(layer_info[i][0] == "conv" || layer_info[i][0] == "fc"|| layer_info[i][0] == "relu"||layer_info[i][0] == "maxpool" || layer_info[i][0] == "avgpool")
      {
        assert(layer_buffers[i]["input"] == nullptr);
        bytes = layer_buffer_redundant_bytes[i]["input"];
        mem_manager->allocate(&layer_buffers[i]["input"],bytes,layer_info[i][0]+ " input");
        prefetch_buffer(i,"input",shape);
        //prefetch input
      }

      link_layer_buffer_bw(i);

      // if(layer_info[i][0] == "conv" || layer_info[i][0] == "fc"|| layer_info[i][0] == "relu" || layer_info[i][0] == "maxpool" || layer_info[i][0] == "avgpool")
      //   cudaDeviceSynchronize();

    }

    last_prefetched_layer_no_ = last_prefetched_layer_no_ - alphaT;

    // std::cout << "prefetch_trigger_layer_no : " << prefetch_trigger_layer_no_ << " sync_layer_no_ : "<<sync_layer_no_<< " last_prefetch_layer_no : " << last_prefetched_layer_no_ << std::endl;
  }

  if (layer_number == sync_layer_no_)
    cudaDeviceSynchronize();

}

void seqNetwork::deallocate_mem_layer_bw(int layer_number, vmm * mem_manager, int local)
{
  int i = layer_number,bytes;
  // cudaDeviceSynchronize();
  //dinput
  if(local==0){
    if(layer_info[i][0]!="input")
    {
      if(i-1 <0 || layer_info[i-1][0] != "flatten")
      {
        assert(layer_buffers[i]["dinput"] != nullptr);
        mem_manager->deleteMem(layer_buffers[i]["dinput"]);
        layer_buffers[i]["dinput"] = nullptr;
      }
      if(layer_info[i][0]=="flatten")
      {
        layer_buffers[i]["doutput"] = nullptr;
        if(i+1<num_layers)
          layer_buffers[i+1]["dinput"] = nullptr;
      }

    }
    return;
  }

  //workspace
  if(layer_info[i][0] == "conv")
  {
    if(layer_buffers[i]["workspace"] != nullptr){
    mem_manager->deleteMem(layer_buffers[i]["workspace"]);
    layer_buffers[i]["workspace"] = nullptr;
    }
  }
  //output
  if(layer_info[i][0] == "maxpool" || layer_info[i][0] == "avgpool" || layer_info[i][0] == "relu" || layer_info[i][0] == "softmax")
  {
    assert(layer_buffers[i]["output"] != nullptr);
    mem_manager->deleteMem(layer_buffers[i]["output"]);
    layer_buffers[i]["output"] = nullptr;

  }
  //input
  if(layer_info[i][0] == "conv" || layer_info[i][0] == "fc"|| layer_info[i][0] == "relu"||layer_info[i][0] == "maxpool" || layer_info[i][0] == "avgpool")
  {
    assert(layer_buffers[i]["input"] != nullptr);
    mem_manager->deleteMem(layer_buffers[i]["input"]);
    layer_buffers[i]["input"] = nullptr;
    //prefetch input
  }
  //labels
  if(layer_info[i][0]=="softmax")
  {
    assert(layer_buffers[0]["labels"] != nullptr);
    mem_manager->deleteMem(layer_buffers[0]["labels"]);
    layer_buffers[0]["labels"] = nullptr;
  }

}

void seqNetwork::link_layer_buffer_bw(int layer_number)
{
  int i = layer_number;
  if(layer_info[i][0]=="flatten")
    layer_buffers[i]["dinput"] = layer_buffers[i]["doutput"];

  if(i > 0)
    layer_buffers[i-1]["doutput"] = layer_buffers[i]["dinput"];

}

int seqNetwork::get_max_batch_size()
{
  return this->max_sub_batch_size_;
}

seqNetwork::~seqNetwork()
{
  cudnnDestroy(handle);
  cublasDestroy(blas_handle);
  for(int i=0;i<num_layers;i++)
  {
    if(layer_buffers[i]["input"]!=nullptr)
      cudaFree(layer_buffers[i]["input"]);
    if(layer_buffers[i]["workspace"]!=nullptr)
      cudaFree(layer_buffers[i]["workspace"]);
    if(layer_buffers[i]["output"]!=nullptr)
      cudaFree(layer_buffers[i]["output"]);
    if(layer_buffers[i]["params"]!=nullptr)
      cudaFree(layer_buffers[i]["params"]);

    if(layer_info[i][0]=="input")
      cudaFree(layer_buffers[i]["labels"]);



  }

}
