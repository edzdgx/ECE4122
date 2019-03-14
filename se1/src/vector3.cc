#include "vector3.h"
#include <iostream>
using namespace std;

// default constructor
//Vector3::Vector3(): x(0), y(0), z(0) {}

//set x, y, and z to xyz
Vector3::Vector3(float xyz) {
   x = y = z = xyz;
}

//set component by name
Vector3::Vector3(float x, float y, float z)
    :x(x), y(y), z(z) {}

//component-wise add
Vector3 Vector3::operator+(const Vector3& rhs) {
    return Vector3(this->x + rhs.x, this->y + rhs.y, this->z + rhs.z);
}

//component-wise subtract
Vector3 Vector3::operator-(const Vector3& rhs) {
    return Vector3(this->x - rhs.x, this->y - rhs.y, this->z - rhs.z);
}

//component-wise multiplication
Vector3 Vector3::operator*(const Vector3& rhs) {
    return Vector3(this->x * rhs.x, this->y * rhs.y, this->z * rhs.z);
}

//component-wise division
Vector3 Vector3::operator/(const Vector3& rhs) {
    return Vector3(this->x / rhs.x, this->y / rhs.y, this->z / rhs.z);
}

//add rhs to each component
Vector3 Vector3::operator+(float rhs) {
    return Vector3(this->x + rhs, this->y + rhs, this->z + rhs);
}

//subtract rhs to each component
Vector3 Vector3::operator-(float rhs) {
    return Vector3(this->x - rhs, this->y - rhs, this->z - rhs);
}

//multiply rhs to each component
Vector3 Vector3::operator*(float rhs) {
    return Vector3(this->x * rhs, this->y * rhs, this->z * rhs);
}

//divide rhs to each component
Vector3 Vector3::operator/(float rhs) {
    return Vector3(this->x / rhs, this->y / rhs, this->z / rhs);
}

// dot product
float Vector3::operator|(const Vector3& rhs) {
    return this->x * rhs.x + this->y * rhs.y + this->z * rhs.z;
}

// cross product
Vector3 Vector3::operator^(const Vector3& rhs) {
    float x = this->y * rhs.z - rhs.y * this->z;
    float y = this->z * rhs.x - rhs.z * this->x;
    float z = this->x * rhs.y - rhs.x * this->y;
    return Vector3(x, y, z);
}

//component-wise add
Vector3& Vector3::operator+=(const Vector3& rhs) {
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    return *this;
}

//component-wise subtract
Vector3& Vector3::operator-=(const Vector3& rhs) {
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    return *this;
}

//component-wise multiply
Vector3& Vector3::operator*=(const Vector3& rhs) {
    this->x *= rhs.x;
    this->y *= rhs.y;
    this->z *= rhs.z;
    return *this;
}

//component-wise divide
Vector3& Vector3::operator/=(const Vector3& rhs) {
    this->x /= rhs.x;
    this->y /= rhs.y;
    this->z /= rhs.z;
    return *this;
}

// Vector3++ and ++Vector3 rotate xyz to the right
// i.e. make x = z, y = x, z = y
// Make sure they function correctly ++v vs v++
//rotate (pre-increment)
Vector3& Vector3::operator++() {
    float temp = this->x;
    this->x = this->z;
    this->z = this->y;
    this->y = temp;
    return *this;
}

//rotate (post-increment)
Vector3 Vector3::operator++(int __unused) {
    Vector3 temp = *this;
    float t = this->x;
    this->x = this->z;
    this->z = this->y;
    this->y = t;
    return temp;
}

// Vector3-- and --Vector3 rotate xyz to the left
// i.e. make x = y, y = z, z = x
Vector3& Vector3::operator--() {
    float temp = this->x;
    this->x = this->y;
    this->y = this->z;
    this->z = temp;
    return *this;
}

//rotate (post-decrement)
Vector3 Vector3::operator--(int __unused) {
    Vector3 temp = *this;
    float t = this->x;
    this->x = this->y;
    this->y = this->z;
    this->z = t;
    return temp;
}

//component-wise equality
bool Vector3::operator==(const Vector3& rhs) {
    return ((this->x == rhs.x) && (this->y == rhs.y) && (this->z == rhs.z));
}

//component-wise inequality
bool Vector3::operator!=(const Vector3& rhs) {
    return ((this->x != rhs.x) || (this->y != rhs.y) || (this->z != rhs.z));
}