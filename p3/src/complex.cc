//
// Created by brian on 11/20/18.
//
#include <iostream>
#include "complex.h"

#include <cmath>

const float PI = 3.14159265358979f;

Complex::Complex() : real(0.0f), imag(0.0f) {}

Complex::Complex(float r) : real(r), imag(0.0f) {}

Complex::Complex(float r, float i) : real(r), imag(i) {}

Complex Complex::operator+(const Complex &b) const {
    Complex res;
    res.real = this->real + b.real;
    res.imag = this->imag + b.imag;

    return res;
}

Complex Complex::operator-(const Complex &b) const {
    Complex res;
    res.real = this->real - b.real;
    res.imag = this->imag - b.imag;

    return res;
}

Complex Complex::operator*(const Complex &b) const {
    Complex res;
    res.real = this->real * b.real - this->imag * b.imag;
    res.imag = this->real * b.imag + this->imag * b.real;

    return res;
}

Complex Complex::mag() const {
    Complex res;

    res.real = sqrt((this->real * this->real) + (this->imag * this->imag));
    res.imag = 0;

    return res;
}

Complex Complex::angle() const {
    Complex res;

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

Complex Complex::conj() const {
    Complex res;

    res.real = this->real;
    res.imag = -1 * this->imag;

    return res;    
}

std::ostream& operator<< (std::ostream& os, const Complex& rhs) {
    Complex c(rhs);
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