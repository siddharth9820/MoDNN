#include "vmm.h"



vmm::vmm(int bytes){

  freeSize = bytes;
  float * memStartAddress;
  cudaMalloc(&memStartAddress,bytes);

  head = new struct memoryNode;
  head->startAddrCuda = memStartAddress;
  head->accessPointer = NULL;
  head->size = freeSize;
  head->isFree = true;

}

void vmm::defragmentMemSimple(){
  struct memoryNode* iterator = head;
  struct memoryNode* prev = NULL;
  struct memoryNode* temp;

  while(iterator){
    if(iterator->isFree && iterator->next && iterator->next->isFree){
      iterator->size+=iterator->next->size;
      temp = iterator->next;
      iterator->next = iterator->next->next;
      delete(temp);
    }else{
      iterator = iterator->next;
    }
  }
}

void vmm::defragmentMem()
{
  std::cout<<" Defragmentation required "<<std::endl;

  struct memoryNode* prev = NULL;
  struct memoryNode* temp;
  struct memoryNode* iterator = head;

  while(iterator){

    if(iterator->next && iterator->isFree && !iterator->next->isFree){
    // swap the contents to push the free block to one side

      // this method needs to be replaced with the equivalent CUDA memory move method
      //memmove(iterator->startAddrCuda,iterator->next->startAddrCuda,iterator->next->size);
      cudaMemcpy(iterator->startAddrCuda,
                  iterator->next->startAddrCuda,
                  iterator->next->size,
                  cudaMemcpyDeviceToDevice);


      unsigned long long int total_size = iterator->size+iterator->next->size;
      iterator->size = iterator->next->size;
      iterator->next->size = total_size-iterator->size;

      iterator->accessPointer = iterator->next->accessPointer;
      *(iterator->accessPointer) = iterator->startAddrCuda;

      iterator->isFree = false;
      iterator->next->isFree = true;
      iterator->next->startAddrCuda = (float*)((unsigned long long int)iterator->startAddrCuda+iterator->size);
      iterator->next->accessPointer = NULL;
      iterator = iterator->next;

    }else if(iterator->next && iterator->isFree && iterator->next->isFree){
    // merge the free blocks
      iterator->size+=iterator->next->size;
      temp = iterator->next;
      iterator->next = iterator->next->next;
      delete(temp);
    }else{
      iterator = iterator->next;
    }
  }
  std::cout<<" Defragmentation complete "<<std::endl;
  printNodes();
}

allocstatus_t vmm::allocate(float** ptr,int bytes, std::string misc){
  if(bytes==0)
  {
    *ptr = NULL;
    return VMM_SUCCESS;
  }

  if(bytes>freeSize){
    std::cout<<"Requested memory more than free memory"<<std::endl;
    return INSUFF_MEM;
  }
  *ptr = NULL;

  defragmentMemSimple(); //join contiguous free blocks

  *ptr = allocateHelper(ptr,bytes,misc);

  if(*ptr!=NULL){
    return VMM_SUCCESS;
  }

  defragmentMem();

  *ptr = allocateHelper(ptr,bytes,misc);

  if(*ptr!=NULL){
    return DEFRAG_SUCCESS;
  }

  std::cout<<" Unable to allocate memory : Reason unknown"<<std::endl;
  return INSUFF_MEM;
}

float* vmm::allocateHelper(float** ptr,int bytes, std::string misc)
{
  struct memoryNode* iterator = head;

  while(iterator){
    if(iterator->isFree){
      if(iterator->size == bytes){
        *ptr = iterator->startAddrCuda;
        iterator->accessPointer = ptr;
        iterator->isFree = false;
        iterator->misc = misc;
        this->freeSize-=bytes;
        return *ptr;
      }else if(iterator->size>bytes){
        *ptr=iterator->startAddrCuda;
        iterator->accessPointer= &*ptr;
        iterator->isFree = false;
        struct memoryNode* temp = new struct memoryNode;

        temp->isFree = true;
        temp->accessPointer = NULL;
        temp->startAddrCuda = (float*)((unsigned long long int)iterator->startAddrCuda+bytes);
        temp->size = iterator->size-bytes;

        iterator->size = bytes;
        temp->next = iterator->next;
        iterator->next = temp;
        iterator -> misc = misc;
        this->freeSize-=bytes;
        return *ptr;
      }
    }
    iterator = iterator->next;
  }
  return NULL;
}

void vmm::deleteMem(float* ptr)
{
  struct memoryNode* iterator = this->head;
  while(iterator){
    std::cout<<(iterator->startAddrCuda)<<"\t"<<ptr<<std::endl;
    if((iterator->startAddrCuda)==ptr){
      if(!iterator->isFree){
        iterator->isFree = true;
        iterator->accessPointer = NULL;
        this->freeSize+=iterator->size;
      }
      return;
    }
    iterator = iterator->next;
  }
}

void vmm::printNodes(){

  struct memoryNode* iterator = head;
  std::cout<<"\n\n ====== Memory Status =======\n";
  while(iterator){
    std::cout<<"Size: "<<iterator->size<<"\tFree: "<<iterator->isFree<<"\tCuda Address: "<<iterator->startAddrCuda<<"\tAccess Pointer: "<<(iterator->accessPointer==NULL?NULL:(iterator->accessPointer))<<"\tMISC - " << iterator -> misc <<std::endl;
    iterator = iterator->next;
  }
  std::cout<<" ============================\n\n";
}

vmm::~vmm()
{
  cudaFree(head->startAddrCuda);
}
