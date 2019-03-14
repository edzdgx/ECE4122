#ifndef VECTOR3_H
#define VECTOR3_H

struct Vector3 {
    float x;
    float y;
    float z;
    //methods go here
    Vector3() = default; //default constructor
    Vector3(float xyz); //set x, y, and z to xyz
    Vector3(float x, float y, float z); //set component by name
    Vector3 operator+(const Vector3& rhs); //component-wise add
    Vector3 operator-(const Vector3& rhs); //component-wise subtract
    Vector3 operator*(const Vector3& rhs); //component-wise multiplication
    Vector3 operator/(const Vector3& rhs); //component-wise division
    Vector3 operator+(float rhs); //add rhs to each component
    Vector3 operator-(float rhs); //subtract rhs from each component
    Vector3 operator*(float rhs); //multiply each component by rhs
    Vector3 operator/(float rhs); //divide each component by rhs
    float operator|(const Vector3& rhs); // dot product
    Vector3 operator^(const Vector3& rhs); // cross product
    Vector3& operator+=(const Vector3& rhs); //component-wise add
    Vector3& operator-=(const Vector3& rhs); //component-wise subtract
    Vector3& operator*=(const Vector3& rhs); //component-wise multiplication
    Vector3& operator/=(const Vector3& rhs); //component-wise division

    // Vector3++ and ++Vector3 rotate xyz to the right
    // i.e. make x = z, y = x, z = y
    // Make sure they function correctly ++v vs v++
    Vector3& operator++();
    Vector3 operator++(int __unused);
    // Vector3-- and --Vector3 rotate xyz to the left
    // i.e. make x = y, y = z, z = x
    Vector3& operator--();
    Vector3 operator--(int __unused);

    bool operator==(const Vector3& rhs); //component-wise equality
    bool operator!=(const Vector3& rhs); //component-wise inequality
};

#endif // VECTOR3_H