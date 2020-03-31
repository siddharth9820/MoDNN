// #include <cuda.h>
// #include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
using namespace std;

struct memoryNode{

	float* startAddrCuda;			// stores the address in GPU memory
	float** accessPointer;
	unsigned long long int size;   // size of the block in the GPU memory
	bool isFree;				  // stores if the node is free or not
	struct memoryNode* next;
	memoryNode(){
		next = NULL;
	}
};
// float**
class vmm{
	
	private:
		
		int freeSize;
		struct memoryNode* head; 

		float* allocateHelper(float** ptr,int bytes){

			struct memoryNode* iterator = head;
			
			while(iterator){
				if(iterator->isFree){
					if(iterator->size == bytes){
						*ptr = iterator->startAddrCuda;
						iterator->accessPointer = ptr;
						iterator->isFree = false;
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
						this->freeSize-=bytes;
						return *ptr;
					}
				}
				iterator = iterator->next;
			}
			return NULL;
		}

	public:
		
		vmm(int bytes){

			freeSize = bytes;

			// to be replaced with CUDA malloc
		    float* memStartAddress = (float*)malloc(bytes);
		    
		    if(memStartAddress==NULL){
		    	cout<<"Unable to allocate the requested memory size."<<endl;
		    }else{
			    head = new struct memoryNode;
			    head->startAddrCuda = memStartAddress;
			    head->accessPointer = NULL;
			    head->size = freeSize;
			    head->isFree = true;
		    }
		}

		// this method just joins contiguous free blocks
		void defragmentMemSimple(){
			
			struct memoryNode* iterator = head;
			struct memoryNode* prev = NULL;
			struct memoryNode* temp;
			
			while(iterator){
				if(iterator->isFree and iterator->next and iterator->next->isFree){
					iterator->size+=iterator->next->size;
					temp = iterator->next;
					iterator->next = iterator->next->next;
					delete(temp);
				}else{
					iterator = iterator->next;
				}
			}
		}

		// this method moves all free blocks in the end and combines them into a single contiguous block
		void defragmentMem(){

			cout<<" Defragmentation required "<<endl;
			
			struct memoryNode* prev = NULL;
			struct memoryNode* temp;
			struct memoryNode* iterator = head;
			
			while(iterator){

				if(iterator->next and iterator->isFree and !iterator->next->isFree){
				// swap the contents to push the free block to one side	
					
					// this method needs to be replaced with the equivalent CUDA memory move method
					memmove(iterator->startAddrCuda,iterator->next->startAddrCuda,iterator->next->size);
					
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

				}else if(iterator->next and iterator->isFree and iterator->next->isFree){
				// merge the free blocks
					iterator->size+=iterator->next->size;
					temp = iterator->next;
					iterator->next = iterator->next->next;
					delete(temp);
				}else{
					iterator = iterator->next;
				}
			}
			cout<<" Defragmentation complete "<<endl;
			printNodes();
		}

		float* allocate(float** ptr,int bytes){

			if(bytes>freeSize){
				cout<<"Requested memory more than free memory"<<endl;
				return NULL;
			}
			*ptr = NULL;
			
			defragmentMemSimple();
			
			*ptr = allocateHelper(ptr,bytes);

			if(*ptr!=NULL){
				return *ptr;
			}

			defragmentMem();
			
			*ptr = allocateHelper(ptr,bytes);

			if(*ptr!=NULL){
				return *ptr;
			}
			
			cout<<" Unable to allocate memory : Reason unknown"<<endl;
			return NULL;
		}

		void deleteMem(float* ptr){

			struct memoryNode* iterator = this->head;
			while(iterator){
				cout<<(iterator->startAddrCuda)<<"\t"<<ptr<<endl;
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

		void printNodes(){

			struct memoryNode* iterator = head;
			cout<<"\n\n ====== Memory Status =======\n";
			while(iterator){
				cout<<"Size: "<<iterator->size<<"\tFree: "<<iterator->isFree<<"\tCuda Address: "<<iterator->startAddrCuda<<"\tAccess Pointer: "<<(iterator->accessPointer==NULL?NULL:*(iterator->accessPointer))<<endl;
				iterator = iterator->next;
			}
			cout<<" ============================\n\n";
		}
};

int main(){
	
	float *p1 = new float,*p2 = new float,*p3 = new float,*p4 = new float,*p5 = new float;
	
	vmm* myMemory = new vmm(8);
	
	p1 = myMemory->allocate(&p1,2);
	p2 = myMemory->allocate(&p2,2);
	p3 = myMemory->allocate(&p3,2);
	p4 = myMemory->allocate(&p4,2);
	myMemory->deleteMem(p2);
	myMemory->deleteMem(p4);
	p5 = myMemory->allocate(&p5,4);
	myMemory->printNodes();

	vmm* myMemory1 = new vmm(8);
	
	p1 = myMemory1->allocate(&p1,2);
	p2 = myMemory1->allocate(&p2,2);
	p3 = myMemory1->allocate(&p3,2);
	p4 = myMemory1->allocate(&p4,2);
	myMemory1->deleteMem(p2);
	myMemory1->deleteMem(p3);
	p5 = myMemory1->allocate(&p5,4);
	myMemory1->printNodes();
}