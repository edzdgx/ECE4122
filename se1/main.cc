#include <iostream>
#include "vector3.h"

int main() {
    // initialize v0, v1, v2
    Vector3 v0;
    Vector3 v1(3, 4, 5);
    std::cout << "v1: x = " << v1.x <<" y = " << v1.y << " z = " << v1.z << "\n";
    Vector3 v2(-3, 4, 5);
    std::cout << "v2: x = " << v2.x <<" y = " << v2.y << " z = " << v2.z << "\n";

    // component-wise calculation
    std::cout<< "\n\nComponent-wise calculation\n";
    v0 = v1 + v2;
    std::cout << "x = " << v0.x <<" y = " << v0.y << " z = " << v0.z << "\n";
    v0 = v1 - v2;
    std::cout << "x = " << v0.x <<" y = " << v0.y << " z = " << v0.z << "\n";
    v0 = v1 * v2;
    std::cout << "x = " << v0.x <<" y = " << v0.y << " z = " << v0.z << "\n";
    v0 = v1 / v2;
    std::cout << "x = " << v0.x <<" y = " << v0.y << " z = " << v0.z << "\n";

    // calc rhs to each component
    std::cout << "\n\nCalc rhs to each component\n";
    v0 = v1 + 2;
    std::cout << "x = " << v0.x <<" y = " << v0.y << " z = " << v0.z << "\n";
    v0 = v1 - 2;
    std::cout << "x = " << v0.x <<" y = " << v0.y << " z = " << v0.z << "\n";
    v0 = v1 * 2;
    std::cout << "x = " << v0.x <<" y = " << v0.y << " z = " << v0.z << "\n";
    v0 = v1 / 2;
    std::cout << "x = " << v0.x <<" y = " << v0.y << " z = " << v0.z << "\n";

    // dot-product
    float f;
    std::cout << "\n\ndot product: ";
    f = v1 | v2;
    std::cout << "f = " << f << "\n";

    // cross-product
    v0 = v1 ^ v2;
    std::cout << "cross product: ";
    std::cout << "x = " << v0.x <<" y = " << v0.y << " z = " << v0.z << "\n";

    //component-wise add
    v1 += v2;
    std::cout << "\n\n************************************************\n";
    std::cout << "x = " << v1.x <<" y = " << v1.y << " z = " << v1.z << "\n";
    v1 = Vector3(3, 4, 5);
    v1 -= v2;
    std::cout << "x = " << v1.x <<" y = " << v1.y << " z = " << v1.z << "\n";
    v1 = Vector3(3, 4, 5);
    v1 *= v2;
    std::cout << "x = " << v1.x <<" y = " << v1.y << " z = " << v1.z << "\n";
    v1 = Vector3(3, 4, 5);
    v1 /= v2;
    std::cout << "x = " << v1.x <<" y = " << v1.y << " z = " << v1.z << "\n";
    v1 = Vector3(3, 4, 5);

    std::cout << "\n\n****************increment***********************\n";
    ++v1;
    std::cout << "x = " << v1.x <<" y = " << v1.y << " z = " << v1.z << "\n";
    v1 = Vector3(3, 4, 5);
    v2++;
    std::cout << "x = " << v2.x <<" y = " << v2.y << " z = " << v2.z << "\n";
    v2 = Vector3(-3, 4, 5);

    std::cout << "\n\n****************decrement*******************\n";
    --v1;
    std::cout << "x = " << v1.x <<" y = " << v1.y << " z = " << v1.z << "\n";
    v1 = Vector3(3, 4, 5);
    v2--;
    std::cout << "x = " << v2.x <<" y = " << v2.y << " z = " << v2.z << "\n";
    v2 = Vector3(-3, 4, 5);

    //component-wise equality
    bool b = v1 == v2;
    std::cout << "v1 == v2?  " << b << "\n\n";

    b = v1 != v2;
    std::cout << "v1 != v2?  " << b << "\n\n";

    Vector3 test = ++v1;
    Vector3 test = v1++;

    return 0;
}
