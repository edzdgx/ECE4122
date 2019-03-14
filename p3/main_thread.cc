#include <iostream>
#include "complex.h"
#include "input_image.h"
#include <mpi.h>

#include <cmath>
#include <vector>
#include <thread>

using namespace std;

const float PI = 3.14159265358979f;

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
            Complex W(cos(2 * PI * p * k / N), -sin(2 * PI * p * k / N));
            H[n] = H[n] + W * h[k * N + num];
        }
        p++;
    }
}

void cpuDFT2D(const char* inFile, const char* outFile) {
    unsigned int numOfThreads = 8;

    // read in the file
    InputImage inputImage(inFile);

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
    thread t5([&] {
        for (int i = widthIter * 4; i < widthIter * 5; i++) {
            cpuDFT1D(&H[0], h, i, N);
        }
    });
    thread t6([&] {
        for (int i = widthIter * 5; i < widthIter * 6; i++) {
            cpuDFT1D(&H[0], h, i, N);
        }
    });
    thread t7([&] {
        for (int i = widthIter * 6; i < widthIter * 7; i++) {
            cpuDFT1D(&H[0], h, i, N);
        }
    });
    thread t8([&] {
        for (int i = widthIter * 7; i < N; i++) {
            cpuDFT1D(&H[0], h, i, N);
        }
    });       

    t1.join(); t2.join(); t3.join(); t4.join();
    t5.join(); t6.join(); t7.join(); t8.join();

    // Column
    vector<Complex> H2(width * height);
    thread tt1([&] {
        for (int i = 0; i < widthIter; i++) {
            cpuDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt2([&] {
        for (int i = widthIter; i < widthIter * 2; i++) {
            cpuDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt3([&] {
        for (int i = widthIter * 2; i < widthIter * 3; i++) {
            cpuDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt4([&] {
        for (int i = widthIter * 3; i < widthIter * 4; i++) {
            cpuDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt5([&] {
        for (int i = widthIter * 4; i < widthIter * 5; i++) {
            cpuDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt6([&] {
        for (int i = widthIter * 5; i < widthIter * 6; i++) {
            cpuDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt7([&] {
        for (int i = widthIter * 6; i < widthIter * 7; i++) {
            cpuDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt8([&] {
        for (int i = widthIter * 7; i < N; i++) {
            cpuDFT1DCol(&H2[0], &H[0], i, height);
        }
    });    

    tt1.join(); tt2.join(); tt3.join(); tt4.join();
    tt5.join(); tt6.join(); tt7.join(); tt8.join();

    Complex* result1D = H2.data();
 
    inputImage.save_image_data(outFile, result1D, width, height);

}


void cpuIDFT1D(Complex* H, Complex* h, int row, int N) {
    int offset = row * N;
    int realN;
    Complex divN(1 / (double) N, 0);    
    for (int n = offset; n < offset + N; n++) {
        realN = n - offset;
        for (int k = 0; k < N; k++) {
            Complex W(cos(2 * PI * realN * k / float(N)), sin(2 * PI * realN * k / float(N)));
            H[n] = H[n] + W * h[k + offset];
        }
        H[n] = divN * H[n];
    }
}

void cpuIDFT1DCol(Complex* H, Complex* h, int num, int N) {
    int p = 0;

    Complex divN(1 / (double) N, 0);    
    for (int n = num; n < N * (N - 1) + num + 1; n = n + N) {
        for (int k = 0; k < N; k++) {
            Complex W(cos(2 * PI * p * k / N), sin(2 * PI * p * k / N));
            H[n] = H[n] + W * h[k * N + num];
        }
        H[n] = divN * H[n];            
        p++;
    }
}

void cpuIDFT2D(const char* inFile, const char* outFile) {
    unsigned int numOfThreads = 8;

    // read in the file
    InputImage inputImage(inFile);

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
            cpuIDFT1D(&H[0], h, i, N);
        }
    });
    thread t2([&] {
        for (int i = widthIter; i < widthIter * 2; i++) {
            cpuIDFT1D(&H[0], h, i, N);
        }
    });
    thread t3([&] {
        for (int i = widthIter * 2; i < widthIter * 3; i++) {
            cpuIDFT1D(&H[0], h, i, N);
        }
    });
    thread t4([&] {
        for (int i = widthIter * 3; i < widthIter * 4; i++) {
            cpuIDFT1D(&H[0], h, i, N);
        }
    });     
    thread t5([&] {
        for (int i = widthIter * 4; i < widthIter * 5; i++) {
            cpuIDFT1D(&H[0], h, i, N);
        }
    });
    thread t6([&] {
        for (int i = widthIter * 5; i < widthIter * 6; i++) {
            cpuIDFT1D(&H[0], h, i, N);
        }
    });
    thread t7([&] {
        for (int i = widthIter * 6; i < widthIter * 7; i++) {
            cpuIDFT1D(&H[0], h, i, N);
        }
    });
    thread t8([&] {
        for (int i = widthIter * 7; i < N; i++) {
            cpuIDFT1D(&H[0], h, i, N);
        }
    });       

    t1.join(); t2.join(); t3.join(); t4.join();
    t5.join(); t6.join(); t7.join(); t8.join();

    // Column
    vector<Complex> H2(width * height);
    thread tt1([&] {
        for (int i = 0; i < widthIter; i++) {
            cpuIDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt2([&] {
        for (int i = widthIter; i < widthIter * 2; i++) {
            cpuIDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt3([&] {
        for (int i = widthIter * 2; i < widthIter * 3; i++) {
            cpuIDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt4([&] {
        for (int i = widthIter * 3; i < widthIter * 4; i++) {
            cpuIDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt5([&] {
        for (int i = widthIter * 4; i < widthIter * 5; i++) {
            cpuIDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt6([&] {
        for (int i = widthIter * 5; i < widthIter * 6; i++) {
            cpuIDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt7([&] {
        for (int i = widthIter * 6; i < widthIter * 7; i++) {
            cpuIDFT1DCol(&H2[0], &H[0], i, height);
        }
    });
    thread tt8([&] {
        for (int i = widthIter * 7; i < N; i++) {
            cpuIDFT1DCol(&H2[0], &H[0], i, height);
        }
    });

    tt1.join(); tt2.join(); tt3.join(); tt4.join();
    tt5.join(); tt6.join(); tt7.join(); tt8.join();

    Complex* result1D = H2.data();

    inputImage.save_image_data_real(outFile, result1D, width, height);


}


int main(int argc, char** argv) {

    if(argc != 4) {
        cout << "Exiting Program... Expected Input Sequence of: './p31 forward/reverse [INPUTFILE] [OUTPUTFILE]'" << endl;
        return 1;        
    }

    string forward_reverse = argv[1];
    string inputName = argv[2];
    string outputName = argv[3];

    string inFile(inputName);
    string outFile(outputName);
    
    if (forward_reverse.compare("forward") == 0) {
        cpuDFT2D(inFile.c_str(), outFile.c_str());
    } else if (forward_reverse.compare("reverse") == 0) {
        cpuIDFT2D(inFile.c_str(), outFile.c_str());
    }

    return 0;
}