#ifndef VMM
#define VMM


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <vector>
#include <map>
#include <string>

/**
* represents a memory node in the linkedlist
*/
struct memoryNode{
  float* startAddrCuda;			/*!<  stores the address in GPU memory. */
	float** accessPointer;    /*!<  stores the address of the pointer which used to access this memory node . */
	unsigned long long int size;   /*!<  size of the block in the GPU memory. */
	bool isFree;				  /*!<  stores if the node is free or not. */
  std::string misc;     /*!< stores misc info. */
  struct memoryNode* next;  /*!<  stores the address of the next node. */
	memoryNode(){
    next = NULL;
  }
};

/**
* an enumerator for memory allocation status
*/
enum allocstatus_t{   
  VMM_SUCCESS,     /*!< memory allocation successful. */
  INSUFF_MEM,      /*!< memory allocation not possible due to insufficient memory. */
  DEFRAG_SUCCESS   /*!< memory allocation and defragmentation successful. */
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

/*! \class vmm

This class is used to represent the GPU memory as a linked list memory model.
*/
class vmm{
  private:
    int freeSize;               /*!< total size of the free memory. */
    struct memoryNode* head;    /*!< pointer to the head of the linked list of the memory nodes. */
    
    /**
    * helper function to allocate memory of size bytes.
    */
    float* allocateHelper(float** ptr,int bytes,std::string misc);  
    
    std::vector<std::map<std::string,float*> > *buffers;

  public:
    /**
    * constructor for declaring bytes amount of memory
    */
		vmm(int bytes,std::vector<std::map<std::string,float*> > *layer_buffers);

		/**
    * this method just joins contiguous free blocks
    */
    void defragmentMemSimple();

		/**
    * this method moves all free blocks in the end and combines them into a single contiguous block
    */
    void defragmentMem();

    /**
    * this method is used to allocate bytes amount of memory and store the access pointer in *ptr
    * @param bytes amount of memory to be allocated
    */
		allocstatus_t allocate(float** ptr,int bytes,std::string misc = "None");

		/**
    * this method frees up the memory pointed by ptr
    */
    void deleteMem(float* ptr);

    /**
    * this method prints all the memory nodes with their details
    */
		void printNodes();

    /**
    * Destroys vmm
    */
    ~vmm();

};

#endif
