#include "vmm.h"

void random_populate_float(float *a,int n)
{
  float *b = (float*)malloc(n*sizeof(float));
  for(int i=0;i<n;i++)
    b[i] = 789.0+i;

  //std::cout << "Copying data" << std::endl;
  gpuErrchk(cudaMemcpy(a,b,n*sizeof(float),cudaMemcpyHostToDevice));

}

void print_arr(float *a,int n)
{
  float *b = (float*)malloc(n*sizeof(float));
  gpuErrchk(cudaMemcpy(b,a,n*sizeof(float),cudaMemcpyDeviceToHost));
  for(int i=0;i<n;i++)
    std::cout << b[i] << " ";
  std::cout << std::endl;
}


int main()
{
    vmm * mem_manager = new vmm(100*sizeof(float));
    mem_manager->printNodes();

    float *a, *b,*c,*d;

    checkvmm(mem_manager -> allocate(&a,25*sizeof(float)));
    checkvmm(mem_manager -> allocate(&b,25*sizeof(float)));
    checkvmm(mem_manager -> allocate(&c,50*sizeof(float)));

    mem_manager->printNodes();

    random_populate_float(c,25);
    print_arr(c,25);
    mem_manager->deleteMem(b);
    mem_manager->defragmentMem();
    print_arr(c,25);

    return 0;
}
