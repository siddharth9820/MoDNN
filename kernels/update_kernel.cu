__global__ void update(float * weights, float * grad,float lr,int N)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  if(x<N)
    weights[x] -= lr*grad[x];
  grad[x] = 0.0;
}
