// #include <cuda.h>
// #include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
using namespace std;

struct memoryNode{
	float* startAddrCuda;
	float** accessPointer;
	unsigned long long int size;
	bool isFree;
	struct memoryNode* next;
	memoryNode(){
		next = NULL;
	}
};
// float**
class vmm{
	
	private:
		int freeSize;

	public:
		struct memoryNode* head;
		vmm(int bytes){

			freeSize = bytes;

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
		float* allocate(float** ptr,int bytes){
			if(bytes>freeSize){
				cout<<"Requested memory more than free memory"<<endl;
				return NULL;
			}
			*ptr = NULL;
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
	vmm* myMemory = new vmm(8);
	float *p1 = new float,*p2 = new float,*p3 = new float,*p4 = new float,*p5 = new float;
	p1 = myMemory->allocate(&p1,2);
	p2 = myMemory->allocate(&p2,2);
	myMemory->printNodes();
	p3 = myMemory->allocate(&p3,2);
	myMemory->printNodes();
	p4 = myMemory->allocate(&p4,2);
	myMemory->deleteMem(p2);
	myMemory->printNodes();
	p5 = myMemory->allocate(&p5,2);
	// cout<<p1<<"\t"<<p5<<"\t"<<*(myMemory->head->accessPointer)<<endl;
	// *(myMemory->head->accessPointer) = p5;
	// *(myMemory->head->next->accessPointer) = p5;
	// cout<<p1<<"\t"<<p5<<"\t"<<*(myMemory->head->accessPointer)<<endl;
	myMemory->printNodes();
}