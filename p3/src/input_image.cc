//
// Created by brian on 11/20/18.
//

#include "input_image.h"
#include "complex.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h> 

InputImage::InputImage(const char* filename) {
    std::ifstream ifs(filename);
    if(!ifs) {
        std::cout << "Can't open image file " << filename << std::endl;
        exit(1);
    }

    ifs >> w >> h;
    data = new Complex[w * h];
    float real;
    float imag;

    for(int r = 0; r < h; ++r) {
        for(int c = 0; c < w; ++c) {
            std::string temp;
            ifs >> temp;

            // ifs >> real;
            if (temp[0] == '(') {
                int comma_index = temp.find(',');
                real = std::stof(temp.substr(1,comma_index - 1));
                imag = std::stof(temp.substr(comma_index + 1, temp.length() - comma_index - 2));

                data[r * w + c] = Complex(real, imag);
            } else {
                real = std::stof(temp);
                data[r * w + c] = Complex(real);
            }

            // std::cout << temp << std::endl;
            // data[r * w + c] = Complex(real);
        }
    }
}

int InputImage::get_width() const {
    return w;
}

int InputImage::get_height() const {
    return h;
}

Complex* InputImage::get_image_data() const {
    return data;
}

void InputImage::save_image_data(const char *filename, Complex *d, int w, int h) {
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

void InputImage::save_image_data_real(const char* filename, Complex* d, int w, int h) {
    std::ofstream ofs(filename);
    if(!ofs) {
        std::cout << "Can't create output image " << filename << std::endl;
        return;
    }

    ofs << w << " " << h << std::endl;

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            // //// ADDED THIS FOR IDFT ////
            // double intpart;
            // float  decimal;
            // decimal = modf(d[r * w + c].real, &intpart);

            // if (decimal >= 0.97) {
            //     d[r * w + c].real = round(d[r * w + c].real);              
            // } else if (decimal <= .01) {
            //     d[r * w + c].real = round(d[r * w + c].real);             
            // }
            // //// ADDED THIS FOR IDFT ////
            ofs << d[r * w + c].real << " ";
        }
        ofs << std::endl;
    }
}
