#include "../layers/layers.h"

__global__ void matrixMultiplyShared(float * A, float * B, float * C,
                                    int numARows, int numAColumns,
                                    int numBRows, int numBColumns,
                                    int numCRows, int numCColumns)
{
    __shared__ float sA[TILE_SIZE][TILE_SIZE];   // Tile size to store elements in shared memory
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int Row = blockDim.y*blockIdx.y + threadIdx.y; //To generate ids of threads.
    int Col = blockDim.x*blockIdx.x + threadIdx.x;
    float Cvalue = 0.0;
    sA[threadIdx.y][threadIdx.x] = 0.0;
    sB[threadIdx.y][threadIdx.x] = 0.0;

    for (int k = 0; k < (((numAColumns - 1)/ TILE_SIZE) + 1); k++)
    {
        if ( (Row < numARows) && (threadIdx.x + (k*TILE_SIZE)) < numAColumns)//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        {
            sA[threadIdx.y][threadIdx.x] = A[(Row*numAColumns) + threadIdx.x + (k*TILE_SIZE)];
        }
        else
        {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if ( Col < numBColumns && (threadIdx.y + k*TILE_SIZE) < numBRows)//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
        {
            sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k*TILE_SIZE)*numBColumns + Col];
        }
        else
        {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j)//Multiplying Elements present in tile
        {
            Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
    }
    if (Row < numCRows && Col < numCColumns)//Saving Final result into Matrix C
    {
        C[Row*numCColumns + Col] = Cvalue;
    }
}


__global__ void matrixMultiplyNaive(float * A, float * B, float * C,
                                    int N,int K,int M)
{

    int Row = blockDim.y*blockIdx.y + threadIdx.y; //To generate ids of threads.
    int Col = blockDim.x*blockIdx.x + threadIdx.x;

    if(Row<N && Col<M)
    {
      float Cvalue = 0.0;
      int k;
      for(k=0;k<K;k++)
      {
        Cvalue += A[Row*K+k] * B[k*M+Col];
      }
      C[Row*M+Col] = Cvalue;
    }
}
