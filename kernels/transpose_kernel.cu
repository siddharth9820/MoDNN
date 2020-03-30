#include "../layers/layers.h"

__global__ void transposeCoalesced(float *odata, const float *idata,int idata_rows,int idata_cols)
{
  __shared__ float tile[TILE_SIZE][TILE_SIZE+1];

  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;
  //int width = gridDim.x * TILE_SIZE;

  for (int j = 0; j < TILE_SIZE; j += BLOCK_SIZE){
    if((y+j)<idata_rows && x<idata_cols)
      tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*idata_cols + x];
   }
  __syncthreads();

  x = blockIdx.y * TILE_SIZE + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_SIZE + threadIdx.y;

  for (int j = 0; j < TILE_SIZE; j += BLOCK_SIZE){
    if((y+j)<idata_cols && x<idata_rows)
      odata[(y+j)*idata_rows + x] = tile[threadIdx.x][threadIdx.y + j];
   }
}

__global__ void transposeNaive(float *odata, const float *idata,int idata_rows,int idata_cols)
{

  int x = blockIdx.x * TILE_SIZE + threadIdx.x;
  int y = blockIdx.y * TILE_SIZE + threadIdx.y;
  //int width = gridDim.x * TILE_SIZE;

  if(y<idata_rows && x<idata_cols)
    odata[x*idata_rows+y] = idata[y*idata_cols+x];
}
