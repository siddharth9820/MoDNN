#include "layers.h"


using namespace network;
using namespace layers;

seqNetwork::seqNetwork(cudnnHandle_t cudnn,std::vector<std::string> &specs)
{
  /*
  Specs is a vector of strings specifying the Neural Network.
  Input -> "input N H W C"
  Conv ->  "conv H W C"
  */
  num_layers = specs.size();
  handle = cudnn;
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
    std::cout << "Dimensions : ";
    for(int j=1;j<layer_info[i].size();j++)
      std::cout << layer_info[i][j] << " ";
    std::cout << std::endl;
  }
}

void seqNetwork::get_output_shape(int shape[])
{
  Layer *last_layer = layer_objects[num_layers-1];
  last_layer->get_output_shape_and_bytes(shape);


}

void seqNetwork::allocate_memory()
{
  std::string layer_type;
  int shape[4],batch_size,rows,columns,channels;
  int kernel_rows,kernel_cols,kernel_channels,bytes;


  std::cout << "Allocating memory for the Neural Network" << std::endl;
  layer_buffers.resize(num_layers);

  for(int i=0;i<num_layers;i++)
  {
    layer_type = layer_info[i][0];
    std::cout << "Layer "<<i+1<<" : "<<layer_type << std::endl;

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

      bytes = layer_objects[i-1]->get_output_shape_and_bytes(shape);

      batch_size = shape[0];
      rows = shape[1];
      columns = shape[2];
      channels = shape[3];

      ConvLayer * new_conv = new ConvLayer(handle,batch_size,rows,columns,channels,kernel_rows,kernel_cols,kernel_channels,VALID);
      layer_objects.push_back(new_conv);

      layer_buffers[i] = init_buffer_map();
      cudaMalloc(&(layer_buffers[i]["output"]),bytes);
      layer_buffers[i]["input"] = layer_buffers[i-1]["output"];
      new_conv -> allocate_internal_mem(&(layer_buffers[i]["params"]),(void**)&(layer_buffers[i]["workspace"]));



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
  }
}


int main(int argc, const char* argv[])
{
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    std::vector<std::string> specs = {"input 10 28 28 3","conv 3 3 5","conv 3 3 4","conv 3 3 3"};
    seqNetwork nn = seqNetwork(cudnn,specs);
    nn.print_network_info();
    nn.allocate_memory();
    int shape[4];
    nn.get_output_shape(shape);

    std::cout << "Printing output shape of Neural Network" << std::endl;
    for(int i=0;i<4;i++)
      std::cout << shape[i] <<" "<<" ";
    std::cout<<std::endl;

    std::cout << "Randomising input to the neural network" << std::endl;
    nn.randomise_input();

    std::cout << "Randomising Parameters of the neural network" << std::endl;
    nn.randomise_params();

    std::cout << "Forward Pass for the neural network" << std::endl;
    nn.forward();

}
