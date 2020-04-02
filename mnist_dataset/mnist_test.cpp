#include "mnist.h"
#include "../data_core/data_loader.h"
#include <iostream>
#include <cmath>

using namespace std;
int main() {
    char* images_file = "data/train-images.idx3-ubyte";
    char* label_file = "data/train-labels.idx1-ubyte";
    
    Dataset* dataset= new MNIST(images_file, label_file, true);
    DataLoader* dataloader = new DataLoader(dataset, 2);

    int input_dim = dataset->getInputDim();
    int dataset_size = dataset->getDatasetSize();
    int label_dim = dataset->getLabelDim();
    int rows = sqrt(input_dim);

    cout << "input dim: " << input_dim << endl;
    cout << "dataset size: " << dataset_size << endl;
    cout << "label dim: " << label_dim << endl;

    float* image = (float*) malloc(2*input_dim*sizeof(float));
    float* label = (float*) malloc(2*label_dim*sizeof(float));
    float *image1, *label1;

    dataset->get_item_range(0,2,image, label);
    image1 = image+input_dim;
    label1 = label+label_dim;

    cout << "\nImage";
    for(int i = 0; i < input_dim; i++) {
        if(i%rows==0)
            cout << endl;
        cout << image1[i] << " ";
    }

    cout << "\nLabel\n";
    for(int i =0; i < label_dim; i++) {
       cout << label1[i] << " ";
    }

    dataloader->get_next_batch(image, label);

    cout << "\nImage";
    for(int i = 0; i < input_dim; i++) {
        if(i%rows==0)
            cout << endl;
        cout << image[i] << " ";
    }

    cout << "\nLabel\n";
    for(int i =0; i < label_dim; i++) {
       cout << label[i] << " ";
    }


}