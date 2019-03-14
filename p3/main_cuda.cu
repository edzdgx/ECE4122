#include <fstream>
#include <sstream>
#include <iostream>
#include <cuda.h>

#include <cmath>
#include <vector>

using namespace std;
const int T_P_B = 256;
const float PI = 3.14159265358979f;
/*
module purge
module load gcc/4.9.0
module load cmake/3.9.1
module load cuda

qsub -I -q coc-ice -l nodes=1:ppn=2:gpus=1,walltime=12:00:00,pmem=2gb

./p3 forward Tower256.txt out256.txt
*/

class gComplex {
public:
    __device__ __host__ gComplex() : real(0.0f), imag(0.0f) {}
    __device__ __host__ gComplex(float r, float i) : real(r), imag(i) {}
    __device__ __host__ gComplex(float r) : real(r), imag(0.0f) {}
    __device__ __host__ gComplex operator+(const gComplex& b) const {
        gComplex res;
        res.real = this->real + b.real;
        res.imag = this->imag + b.imag;

        return res;
    }
    __device__ __host__ gComplex operator-(const gComplex& b) const {
        gComplex res;
        res.real = this->real - b.real;
        res.imag = this->imag - b.imag;

        return res;
    }
    __device__ __host__ gComplex operator*(const gComplex& b) const {
        gComplex res;
        res.real = this->real * b.real - this->imag * b.imag;
        res.imag = this->real * b.imag + this->imag * b.real;

        return res;
    }

    __device__ __host__ gComplex mag() const {
        gComplex res;

        res.real = sqrt((this->real * this->real) + (this->imag * this->imag));
        res.imag = 0;

        return res;
    }
    __device__ __host__ gComplex angle() const {
        gComplex res;

        // (0, 0)
        if (this->real == 0 && this->imag == 0) {
            res.real = 0;
            res.imag = 0;
            return res;
        }

        float temp = this->imag / this->real;
        res.real = atan(temp);

        if (this->real < 0 && this->imag >= 0) { // 2nd quadrant
            res.real = res.real + PI;
        } else if (this->real < 0 && this->imag < 0) { // 3rd quadrant
            res.real = res.real - PI;
        } else if (this->real > 0 && this->imag < 0) { // 4th quadrant
            res.real = -1 * res.real;
        }
        res.imag = 0;

        return res;
    }
    __device__ __host__ gComplex conj() const {
        gComplex res;

        res.real = this->real;
        res.imag = -1 * this->imag;

        return res;
    }

    float real;
    float imag;
};

std::ostream& operator<<(std::ostream& os, const gComplex& rhs) {
    gComplex c(rhs);
    if(fabsf(rhs.imag) < 1e-10) c.imag = 0.0f;
    if(fabsf(rhs.real) < 1e-10) c.real = 0.0f;

    if(c.imag == 0) {
        os << c.real;
    }
    else {
        os << "(" << c.real << "," << c.imag << ")";
    }
    return os;
}

class InputImage {
public:

    InputImage(const char* filename) {
        std::ifstream ifs(filename);
        if(!ifs) {
            std::cout << "Can't open image file " << filename << std::endl;
            exit(1);
        }

        ifs >> w >> h;
        data = new gComplex[w * h];
        for(int r = 0; r < h; ++r) {
            for(int c = 0; c < w; ++c) {
                float real;
                ifs >> real;
                data[r * w + c] = gComplex(real);
            }
        }
    }
    int get_width() const {
        return w;
    }
    int get_height() const {
        return h;
    }

    //returns a pointer to the image data.  Note the return is a 1D
    //array which represents a 2D image.  The data for row 1 is
    //immediately following the data for row 0 in the 1D array
    gComplex* get_image_data() const {
        return data;
    }

    //use this to save output from forward DFT
    void save_image_data(const char* filename, gComplex* d, int w, int h) {
        std::ofstream ofs(filename);
        if(!ofs) {
            std::cout << "Can't create output image " << filename << std::endl;
            return;
        }

        ofs << w << " " << h << std::endl;

        for(int r = 0; r < h; ++r) {
            for(int c = 0; c < w; ++c) {
                ofs << d[r * w + c] << " ";
            }
            ofs << std::endl;
        }
    }
    //use this to save output from reverse DFT
    void save_image_data_real(const char* filename, gComplex* d, int w, int h) {
        std::ofstream ofs(filename);
        if(!ofs) {
            std::cout << "Can't create output image " << filename << std::endl;
            return;
        }

        ofs << w << " " << h << std::endl;

        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                ofs << d[r * w + c].real << " ";
            }
            ofs << std::endl;
        }
    }

private:
    int w;
    int h;
    gComplex* data;
};



void transpose(gComplex* arr, int size) {
    int n = sqrt(size);
    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            swap(arr[n * i + j], arr[n * j + i]);
        }
    }
}


__global__ void cudaDFT1D(gComplex* H, gComplex* h, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int col = idx % N;
    int row = idx / N;

    int start = row * N;

    for (int k = 0; k < N; k++) {
        gComplex W(cos(2 * PI * col * k / float(N)), -sin(2 * PI * col * k / float(N)));
        H[idx] = H[idx] + W * h[start + k];
    }
}

__global__ void cudaDFT1DCol(gComplex* H, gComplex* h, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int col = idx % N;
    int row = idx / N;

    int start = col;

    for (int k = 0; k < N; k++) {
        gComplex W(cos(2 * PI * row * k / float(N)), -sin(2 * PI * row * k / float(N)));
        H[idx] = H[idx] + W * h[start + N*k];
    }
}

void cudaDFT2D(InputImage inputImage, gComplex* h, int width, int height, string outputName) {

    int size = width * height;
    gComplex* d_H;
    gComplex* d_h;

    cudaMalloc((void **) &d_h, size * sizeof(gComplex));
    cudaMalloc((void **) &d_H, size * sizeof(gComplex));

    cudaMemcpy(d_h, h, size * sizeof(gComplex), cudaMemcpyHostToDevice);

    cudaDFT1D<<<size / T_P_B, T_P_B>>>(d_H, d_h, width);
    cudaDeviceSynchronize();

    gComplex* d_H2;
    cudaMalloc((void **) &d_H2, size * sizeof(gComplex));

    cudaDFT1DCol<<<size / T_P_B, T_P_B>>>(d_H2, d_H, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h, d_H2, size * sizeof(gComplex), cudaMemcpyDeviceToHost);

    inputImage.save_image_data(outputName.c_str(), h, width, height);

    cudaFree(d_h);
    cudaFree(d_H);
    cudaFree(d_H2);
}


int main(int argc, char** argv) {

    string forward = argv[1];
    string inputName = argv[2];
    string outputName = argv[3];

    InputImage inputImage(inputName.c_str());

    int width = inputImage.get_width(); // N
    int height = inputImage.get_height();

    gComplex* h = inputImage.get_image_data();

    cudaDFT2D(inputImage, h, width, height, outputName);

    return 0;
}
