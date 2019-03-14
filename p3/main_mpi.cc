#include <iostream>
#include "complex.h"
#include "input_image.h"
#include <mpi.h>

#include <cmath>
#include <vector>
#include <thread>

using namespace std;

const float PI = 3.14159265358979f;

void transpose(Complex* arr, int size) {
    int n = sqrt(size);
    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            swap(arr[n * i + j], arr[n * j + i]);
        }
    }        
}

void cpuDFT1D(Complex* H, Complex* h, int row, int N) {
    int offset = row * N;
    int realN;
    for (int n = offset; n < offset + N; n++) {
        realN = n - offset;
        for (int k = 0; k < N; k++) {
            Complex W(cos(2 * PI * realN * k / float(N)), -sin(2 * PI * realN * k / float(N)));
            H[n] = H[n] + W * h[k + offset];
        }
    }
}

void cpuDFT1DCol(Complex* H, Complex* h, int num, int N) {
    int p = 0;

    for (int n = num; n < N * (N - 1) + num + 1; n = n + N) {
        for (int k = 0; k < N; k++) {
            Complex W(cos(2 * M_PI * p * k / N), -sin(2 * M_PI * p * k / N));
            H[n] = H[n] + W * h[k * N + num];
        }
        p++;
    }
}

void cpuDFT2D(const char* fileName) {
    unsigned int numOfThreads = 4;

    // read in the file
    InputImage inputImage(fileName);

    int width = inputImage.get_width(); // N
    int N = width;
    int height = inputImage.get_height();

    int widthIter = width / numOfThreads;

    // vector to hold data
    vector<Complex> H(width * height);

    // data
    Complex* h = inputImage.get_image_data();

    thread t1([&] {
        for (int i = 0; i < widthIter; i++) {
            cpuDFT1D(&H[0], h, i, N);
        }
    });
    thread t2([&] {
        for (int i = widthIter; i < widthIter * 2; i++) {
            cpuDFT1D(&H[0], h, i, N);
        }
    });
    thread t3([&] {
        for (int i = widthIter * 2; i < widthIter * 3; i++) {
            cpuDFT1D(&H[0], h, i, N);
        }
    });
    thread t4([&] {
        for (int i = widthIter * 3; i < widthIter * 4; i++) {
            cpuDFT1D(&H[0], h, i, N);
        }
    });      
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // column
    vector<Complex> H2(width * height);
    thread t5([&] {
        for (int i = 0; i < widthIter; i++) {
            cpuDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread t6([&] {
        for (int i = widthIter; i < widthIter * 2; i++) {
            cpuDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread t7([&] {
        for (int i = widthIter * 2; i < widthIter * 3; i++) {
            cpuDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread t8([&] {
        for (int i = widthIter * 3; i < widthIter * 4; i++) {
            cpuDFT1DCol(&H2[0], &H[0], i, height);
        }
    });   
    t5.join();
    t6.join();
    t7.join();
    t8.join();

    Complex* result1D = H2.data();

    inputImage.save_image_data("result.txt", result1D, width, height);
}

void mpiDFT1D(Complex* H, Complex* h, int N) {
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < N; k++) {
            Complex W(cos(2 * PI * n * k / float(N)), -sin(2 * PI * n * k / float(N)));
            H[n] = H[n] + W * h[k];            
        }
    }
}

void mpiIDFT1D(Complex* H, Complex* h, int N) {
    Complex divN(1 / (double) N, 0);
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < N; k++) {
            Complex W(cos(-(2 * PI * n * k / float(N))), -sin(-(2 * PI * n * k / float(N))));

            H[n] = H[n] + W * h[k];            
        }
        H[n] = divN * H[n];
    }
}

void mpiFFT1D(Complex* H, Complex* E, Complex* O, Complex* h, int N) {
    for (int n = 0; n < N; n++) {
        Complex Wn(cos(2 * PI * n / float(N)), -sin(2 * PI * n / float(N)));
        for (int k = 0; k < (N / 2); k++) {
            // Complex W(cos(2 * PI / float(N)), -sin(2 * PI / float(N)));
            Complex W(cos(2 * PI * 2 * n * k / float(N)), -sin(2 * PI * 2 * n * k / float(N)));
            E[n] = E[n] + W * h[2 * k];
            O[n] = O[n] + W * h[2 * k + 1];
        }
        H[n] = E[n] + Wn * O[n];
    }
}

void mpiDFT2D(InputImage inputImage, Complex* h, int width, int height, string outputName) {
    int N = width;

    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int rowsPerProc = height / world_size; 
    int row = rowsPerProc * world_rank; // starting row for each process
    int lastRows = rowsPerProc + (height % world_size);
    if (world_rank == world_size - 1) {
        rowsPerProc = lastRows;
    }

    Complex* H;
    Complex* E;
    Complex* O;

    if (world_rank == 0) {
        H = new Complex[width * height];
    } else {
        H = new Complex[width * rowsPerProc];
    }
    E = new Complex[width * rowsPerProc];
    O = new Complex[width * rowsPerProc];

    // row * N: offset for each processor
    // i * N: offset for each loop1
    for (int i = 0; i < rowsPerProc; i++) {
        mpiFFT1D(H + (i * N), E + (i * N), O + (i * N), h + (row * N) + (i * N), N);
        // mpiDFT1D(H + (i * N), h + (row * N) + (i * N), N);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        for (int i = 1; i < world_size; i++) {
            if (i == world_size - 1) {
                // cout << "waiting for cpu: " << i << endl;
                // cout << "lastrows: " << lastRows << endl;
                MPI_Recv(H + rowsPerProc * i * width, lastRows * width * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(H + rowsPerProc * i * width, rowsPerProc * width * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // cout << "cpu 0 received from cpu: " << i << endl;
        }
    } else {
        // cout << "cpu " << world_rank << ": preparing 2D DFT column chunks to cpu 0. " << rowsPerProc << endl;
        MPI_Send(H, rowsPerProc * width * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        // cout << "cpu " << world_rank << ": Sending 2D DFT column chunks to cpu 0." << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        // cout << "cpu 0: Saving 2D DFT matrix into MyAfter1d.txt." << endl;
        // inputImage.save_image_data(outputName.c_str(), H, width, height);
        transpose(H, width * height);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // column
    int colsPerProc = width / world_size;
    int col = colsPerProc * world_rank; // starting col for each process
    int lastCols = colsPerProc + (width % world_size);
    if (world_rank == world_size - 1) {
        colsPerProc = lastCols;
    }

    Complex* H2;
    Complex* E2;
    Complex* O2;
    if (world_rank == 0) {
        H2 = new Complex[width * height];
    } else {
        H2 = new Complex[height * colsPerProc];
    }
    E2 = new Complex[height * colsPerProc];
    O2 = new Complex[height * colsPerProc];

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        for (int i = 1; i < world_size; i++) {
            if (i == world_size - 1) {
                MPI_Send(H + colsPerProc * i * height, lastCols * height * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD);
            } else {
                MPI_Send(H + colsPerProc * i * height, colsPerProc * height * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        MPI_Recv(H + col * height, colsPerProc * height * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // if (world_rank != 0) {
    //     cout << "cpu " << world_rank << " receiving" << endl;
    //     MPI_Recv(H + col * height, colsPerProc * height * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //     cout << "cpu " << world_rank << " received" << endl;  
    // } else {
    //     for (int i = 1; i < world_size; i++) {
    //         if (i == world_size - 1) {
    //             MPI_Send(H + colsPerProc * i * height, lastCols * height * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD);
    //         } else {
    //             MPI_Send(H + colsPerProc * i * height, colsPerProc * height * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD);
    //         }
    //         cout << "cpu 0: Sending 1D DFT column chunks to cpu " << i << "." << endl;
    //     }
    // }
    MPI_Barrier(MPI_COMM_WORLD);


    for (int i = 0; i < colsPerProc; i++) {
        mpiFFT1D(H2 + (i * height), E2 + (i * height), O2 + (i * height), H + (col * height) + (i * height), height);
        // mpiDFT1D(H2 + (i * height), H + (col * height) + (i * height), height);
    }    
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        for (int i = 1; i < world_size; i++) {
            if (i == world_size - 1) {
                MPI_Recv(H2 + colsPerProc * i * height, lastCols * height * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(H2 + colsPerProc * i * height, colsPerProc * height * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        MPI_Send(H2, colsPerProc * height * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    if (world_rank == 0) {
        transpose(H2, width * height);
        inputImage.save_image_data(outputName.c_str(), H2, width, height);
    }
}


int main(int argc, char** argv) {
    string forward = argv[1];
    string inputName = argv[2];
    string outputName = argv[3];

    InputImage inputImage(inputName.c_str());

    int width = inputImage.get_width(); // N
    int height = inputImage.get_height();

    Complex* h = inputImage.get_image_data();





        //////////// Time Code that is inside this block ////////////

        //////////// Time Code that is inside this block ////////////






    if (forward.compare("forward") == 0) {
        // mpiDFT2D(inputImage, h, width, height, outputName);


    MPI_Init(&argc, &argv);









    int N = width;

    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int rowsPerProc = height / world_size; 
    int row = rowsPerProc * world_rank; // starting row for each process
    int lastRows = rowsPerProc + (height % world_size);
    if (world_rank == world_size - 1) {
        rowsPerProc = lastRows;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    Complex* H;
    Complex* E;
    Complex* O;

    if (world_rank == 0) {
        H = new Complex[width * height];
    } else {
        H = new Complex[width * rowsPerProc];
    }
    E = new Complex[width * rowsPerProc];
    O = new Complex[width * rowsPerProc];

    // row * N: offset for each processor
    // i * N: offset for each loop1
    for (int i = 0; i < rowsPerProc; i++) {
        mpiFFT1D(H + (i * N), E + (i * N), O + (i * N), h + (row * N) + (i * N), N);
        // mpiDFT1D(H + (i * N), h + (row * N) + (i * N), N);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        for (int i = 1; i < world_size; i++) {
            if (i == world_size - 1) {
                // cout << "waiting for cpu: " << i << endl;
                // cout << "lastrows: " << lastRows << endl;
                MPI_Recv(H + rowsPerProc * i * width, lastRows * width * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(H + rowsPerProc * i * width, rowsPerProc * width * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            // cout << "cpu 0 received from cpu: " << i << endl;
        }
    } else {
        // cout << "cpu " << world_rank << ": preparing 2D DFT column chunks to cpu 0. " << rowsPerProc << endl;
        MPI_Send(H, rowsPerProc * width * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        // cout << "cpu " << world_rank << ": Sending 2D DFT column chunks to cpu 0." << endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        // cout << "cpu 0: Saving 2D DFT matrix into MyAfter1d.txt." << endl;
        // inputImage.save_image_data(outputName.c_str(), H, width, height);
        transpose(H, width * height);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // column
    int colsPerProc = width / world_size;
    int col = colsPerProc * world_rank; // starting col for each process
    int lastCols = colsPerProc + (width % world_size);
    if (world_rank == world_size - 1) {
        colsPerProc = lastCols;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    Complex* H2;
    Complex* E2;
    Complex* O2;
    if (world_rank == 0) {
        H2 = new Complex[width * height];
    } else {
        H2 = new Complex[height * colsPerProc];
    }
    E2 = new Complex[height * colsPerProc];
    O2 = new Complex[height * colsPerProc];

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank != 0) {
        MPI_Recv(H, colsPerProc * height * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        for (int i = 1; i < world_size; i++) {
            if (i == world_size - 1) {
                MPI_Send(H + colsPerProc * i * height, lastCols * height * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD);
            } else {
                MPI_Send(H + colsPerProc * i * height, colsPerProc * height * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < colsPerProc; i++) {
        // mpiFFT1D(H2 + (i * height), E2 + (i * height), O2 + (i * height), H + (col * height) + (i * height), height);
        mpiFFT1D(H2 + (i * height), E2 + (i * height), O2 + (i * height), H + (i * height), height);
        // mpiDFT1D(H2 + (i * height), H + (col * height) + (i * height), height);
    }    
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        for (int i = 1; i < world_size; i++) {
            if (i == world_size - 1) {
                MPI_Recv(H2 + colsPerProc * i * height, lastCols * height * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(H2 + colsPerProc * i * height, colsPerProc * height * sizeof(Complex), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        MPI_Send(H2, colsPerProc * height * sizeof(Complex), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }


    if (world_rank == 0) {
        transpose(H2, width * height);


        inputImage.save_image_data(outputName.c_str(), H2, width, height);
        
    }



    } else if (forward.compare("reverse") == 0) {

    }
    MPI_Finalize();
    


    
    return 0;
}