#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <bits/stdc++.h>
#include <string.h>
#include <iomanip>
using namespace std;
/*
module purge
module load gcc/4.9.0
module load cmake/3.9.1
module load cuda
qsub -I -q coc-ice -l nodes=1:ppn=2:gpus=1,walltime=2:00:00,pmem=2gb
*/

/*
k = float
timesteps = integer
width/height = integer
default starting temp = float
location_x, location_y, location_z, width, height = integer
fixed_temperatures = float
*/

// CPU = Host
// GPU = Device


// global vars
#define T_P_B 512
string dimension;
float k;
int timesteps;
int width_i, height_i;
int depth_i = 1;
int location_x, location_y, location_z, width, height, depth;
float init_temp;
float fixed_temp;

float *h_grid;
float *h_fixed;

// struct block {
//     int x;
//     int y;
//     int z;
//     int width;
//     int height;
//     int depth;
//     int size;
//     float temp;
// };

// struct conf {
//     string dimension;
//     float k;
//     int timesteps;
//     int width_i;
//     int height_i;
//     int depth_i;
//     float init_temp;
//     vector<block> blocks;
// }

// global = call by HOST, run on DEVICE
// device = call by DEVICE, run on DEVICE
// HOST = no qualifier
__device__ int up(int idx, int width_i, int height_i) {
    int idx_next = idx - width_i;
    if(idx_next < 0)
        idx_next = idx;
    return idx_next;
}

__device__ int down(int idx, int width_i, int height_i) {
    int idx_next = idx + width_i;
    if(idx_next > width_i * height_i - 1)
        idx_next = idx;
    return idx_next;
}

__device__ int left(int idx, int width_i) {
    if (idx % width_i != 0) {
        return idx - 1;
    } else {
        return idx;
    }
}

__device__ int right(int idx, int width_i) {
    if (idx % width_i != (width_i - 1)) {
        return idx + 1;
    } else {
        return idx;
    }
}

__device__ int front(int idx, int width_i, int height_i) {
    return idx - width_i * height_i;
}

__device__ int back(int idx, int width_i, int height_i) {
    return idx + width_i * height_i;
}


// __device__ void swapPtr(float **oldPtr, float **newPtr) {
//     int temp = *oldPtr;
//     *oldPtr = *newPtr;
//     *newPtr = temp;
// }

__global__ void heat2d(float *arr, float *temp, int width, int height, int size, float k) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;


    // temp[idx] = arr[left(idx, width)] +
    //             arr[right(idx, width)] +
    //             arr[up(idx, width, height)] +
    //             arr[down(idx, width, height)];

    temp[idx] = arr[idx] + k * (arr[left(idx, width)] +
                                arr[right(idx, width)] +
                                arr[up(idx, width, height)] +
                                arr[down(idx, width, height)] -
                                arr[idx] * 4);
    // temp[idx] = arr[idx]  + k * ((
    //                             arr[left(idx, width)] +
    //                             arr[right(idx, width)] +
    //                             arr[up(idx, width, height)] +
    //                             arr[down(idx, width, height)])/4.0-arr[idx]);

    // filter
    // x(t) --> filter ---> y(t)
    // 1st low pass filter
    // y(n) = y(n) * (1-k) + x(n)*k;
    // k: 0-1

}


string deleteSpace(string str)
{
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    return str;
}

void parseConf(string filename) {

    ifstream inFile;
    inFile.open(filename.c_str());

    if(!inFile) {
        cerr << "Unable to open file datafile.txt";
        exit(1);   // call system to stop
    }

    // process conf file
    string line;
    int count = 0;
    while(getline(inFile, line)) {
        if(line[0] != '#' && (!line.empty())) {
            if (count == 0) {
                dimension = deleteSpace(line);
                cout << dimension << endl;
            } else if (count == 1) {
                k = atof(deleteSpace(line).c_str());
                cout << k << endl;
            } else if (count == 2) {
                timesteps = atoi(deleteSpace(line).c_str());
                cout << timesteps << endl;
            } else if (count == 3) {
                // replace ',' with ' ' for easier parsing
                replace(line.begin(), line.end(), ',', ' ');
                if (dimension == "2D") {
                    vector<string> vec;
                    istringstream iss(line);
                    for(string line; iss >> line;) {
                        vec.push_back(line);
                    }

                    width_i = atoi(vec.at(0).c_str());
                    height_i = atoi(vec.at(1).c_str());

                    h_grid = new float[width_i * height_i];
                    h_fixed = new float[width_i * height_i];

                    cout << width_i << " " << height_i << endl;
                } else {
                    vector<string> vec;
                    istringstream iss(line);
                    for(string line; iss >> line;) {
                        vec.push_back(line);
                    }

                    width_i = atoi(vec.at(0).c_str());
                    height_i = atoi(vec.at(1).c_str());
                    depth_i = atoi(vec.at(2).c_str());

                    h_grid = new float[width_i * height_i * depth_i];
                    h_fixed = new float[width_i * height_i * depth_i];

                    cout << width_i << " " << height_i << " " << depth_i << endl;
                }
            } else if (count == 4) {
                init_temp = atof(deleteSpace(line).c_str());

                if (dimension == "2D") {
                    for (int i = 0; i < width_i * height_i; i++) {
                        h_grid[i] = init_temp;
                        h_fixed[i] = -1;
                    }
                } else {
                    for (int i = 0; i < width_i * height_i * depth_i; i++) {
                        h_grid[i] = init_temp;
                        h_fixed[i] = -1;
                    }
                }

                cout << init_temp << endl;
            } else if (count >= 5 && !inFile.eof()) {
                // replace ',' with ' ' for easier parsing
                replace(line.begin(), line.end(), ',', ' ');
                if (dimension == "2D") {
                    vector<string> vec;
                    istringstream iss(line);
                    for(string line; iss >> line;) {
                        vec.push_back(line);
                    }

                    location_x = atoi(vec.at(0).c_str());
                    location_y = atoi(vec.at(1).c_str());
                    width = atoi(vec.at(2).c_str());
                    height = atoi(vec.at(3).c_str());
                    fixed_temp = atof(vec.at(4).c_str());

                    int start_point = location_y * width_i + location_x;

                    cout << location_x << " " << location_y << " " << width << " " << height << " " << fixed_temp << endl;

                    int row = start_point / width_i;
                    for(int i = start_point; i < start_point + width + (height - 1) * width_i; i++) {
                        // cout << "debug: " << width + row * width_i + start_point % width_i << endl;
                        if (i < width + row * width_i + start_point % width_i && i >= row * width_i + start_point % width_i) {
                            h_fixed[i] = fixed_temp;
                            cout << i << endl;

                        }
                        if (i == -1 + (row + 1) * width_i) {
                            row++;
                        }
                    }

                    // (location_y * width_i + location_x)

                } else {
                    vector<string> vec;
                    istringstream iss(line);
                    for(string line; iss >> line;) {
                        vec.push_back(line);
                    }

                    location_x = atoi(vec.at(0).c_str());
                    location_y = atoi(vec.at(1).c_str());
                    location_z = atoi(vec.at(2).c_str());
                    width = atoi(vec.at(3).c_str());
                    height = atoi(vec.at(4).c_str());
                    depth = atoi(vec.at(5).c_str());
                    fixed_temp = atof(vec.at(6).c_str());

                    cout << location_x << " " << location_y << " " << location_z << " " << width << " " << height << " " << depth << " " << fixed_temp << endl;
                }
            }


            count++;

        }
    }

    inFile.close();


}


void printToCSV(float *h_grid) {
    std::ofstream myFile("heatOutput.csv");

    for (int i = 0; i < width_i * height_i; i++) {

        // cout.setprecision(1);
        // myFile.setprecision(1);
        cout << h_grid[i] << ", ";
        myFile << h_grid[i];
        if (((i+1) % width_i) == 0 && i != 0) {
            cout << endl;
            myFile << endl;
        } else {
            myFile << ", ";
        }
    }
}

void set_fixed(float *d_new, float *h_fixed, int size) {
    for (int i = 0; i < size; i++) {
        if (h_fixed[i] != -1) {
            d_new[i] = h_fixed[i];
        }
    }
}




// __global__ void vectorAdd(int *a, int *b, int *c, int n) {
//     int id = threadIdx.x + blockIdx.x * blockDim.x;
//     int i = threadIdx.x;
//     if (i < n) {
//         c[i] = a[i] + b[i];
//     }

// }

int main(int argc, char** argv) {
    string filename = argv[1];
    parseConf(filename);

    int area = width_i * height_i;
    int size = width_i * height_i * depth_i;
    int N = (size + T_P_B - 1) / T_P_B;
    float *d_old, *d_new;

    cout << endl << endl << endl;
    for (int i = 0; i < width_i * height_i; i++) {
        cout << h_fixed[i] << " ";
        if (((i+1) % width_i) == 0 && i != 0) {
            cout << endl;
        }
    }
    cout << endl << endl;
    for (int i = 0; i < width_i * height_i; i++) {
        if (h_fixed[i] != -1) {
            h_grid[i] = h_fixed[i];
        }
        cout << h_grid[i] << " ";
        if (((i+1) % width_i) == 0 && i != 0) {
            cout << endl;
        }
    }
    cout << endl << endl;



    // cudaMalloc(void **devPtr, size_t sizeInBytes);
    cudaMalloc((void **) &d_old, size * sizeof(float));
    cudaMalloc((void **) &d_new, size * sizeof(float));



    cout << "T_P_B: " << (width_i*height_i + 32 - 1)/32 << endl;

    // cudaMemcpy(void *dest, void *src, size_t, sizeinBytes, enum direction);
    // cudaMemcpy(d_old, h_grid, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_new, h_grid, size * sizeof(float), cudaMemcpyHostToDevice);

    cout << timesteps << endl;
    // <<<numberoOfBlocks, numberOfThreadsperBlock>>>
    for (int i = 0; i < timesteps; i++) {
        cudaMemcpy(d_old, h_grid, size * sizeof(float), cudaMemcpyHostToDevice);
        cout << i << endl;
        // if (strcmp(dimension.c_str(), "2D")) {
            // copy array elements
            heat2d<<<N, T_P_B>>>(d_old, d_new, width_i, height_i, size, k);
            cudaDeviceSynchronize();
            // set_fixed(d_new, h_fixed, size);
        // }
        cudaMemcpy(h_grid, d_new, size * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < width_i * height_i; i++) {
            if (h_fixed[i] != -1) {
                h_grid[i] = h_fixed[i];
            }
            printf("%.2f ", h_grid[i]);
            if (((i+1) % width_i) == 0 && i != 0) {
                cout << endl;
            }
        }
        cout << endl << endl;
    }

    // cudaMemcpy(h_grid, d_old, N * sizeof(float), cudaMemcpyDeviceToHost);
    printToCSV(h_grid);

    // print to debug



    // cudaFree(void **devPtr);
    cudaFree(d_old);
    cudaFree(d_new);


    return 0;
}