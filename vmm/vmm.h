#ifndef VMM
#define VMM


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda.h>

struct memoryNode{
  float* startAddrCuda;			// stores the address in GPU memory
	float** accessPointer;
	unsigned long long int size;   // size of the block in the GPU memory
	bool isFree;				  // stores if the node is free or not
  std::string misc;     //stores misc info
  struct memoryNode* next;
	memoryNode(){
    next = NULL;
  }
};

enum allocstatus_t{
  VMM_SUCCESS,
  INSUFF_MEM,
  DEFRAG_SUCCESS
};

#define checkvmm(expression)                                 \
  {                                                          \
    allocstatus_t status = (expression);                     \
    if (status != VMM_SUCCESS) {                             \
      if(status == DEFRAG_SUCCESS){                          \
          std::cerr << "Defragmented Memory" << std::endl;    \
        }                                                     \
      else if (status==INSUFF_MEM){                           \
        std::cerr << "Insufficient Memory" << std::endl;     \
      }                                                      \
      else{                                                  \
        std::cerr << "Unknown Error" << std::endl;           \
      }                                                      \
    }                                                        \
  }                                                          \

class vmm{
  private:
    int freeSize;
    struct memoryNode* head;
    float* allocateHelper(float** ptr,int bytes,std::string misc);

  public:
		vmm(int bytes);
		// this method just joins contiguous free blocks
		void defragmentMemSimple();
		// this method moves all free blocks in the end and combines them into a single contiguous block
		void defragmentMem();
		allocstatus_t allocate(float** ptr,int bytes,std::string misc = "None");
		void deleteMem(float* ptr);
		void printNodes();
    ~vmm();
};

#endif
