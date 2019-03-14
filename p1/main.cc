#include <iostream>
#include <vector>
#include "src/simple_string.h"
#include "src/array.h"

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//NOTE THIS IS NOT A COMPLETE LISTING OF TESTS THAT WILL BE RUN ON YOUR CODE
//Just a sample to help get you started and give you an idea of how i'll be testing
//Above each test gives the counts for std::vector and the solution i've written for your array
//As well as checking totals ensure your array doesn't leak memory
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//comment/uncomment these lines to enable tests
#define TEST_PUSH_BACK_NEW_VEC
#define TEST_CLEAR
#define TEST_PUSH_FRONT_VEC
#define TEST_PUSH_FRONT_WITH_RESERVE
#define TEST_POP_BACK
#define TEST_INITIALIZER_LIST
#define TEST_POP_FRONT

using std::vector;
//test your code here

int main() {

    {
        simple_string a("a");
        simple_string b;
        simple_string c;

        std::cout << "Vector" << std::endl;
        simple_string::initialize_counts();
        vector<simple_string> vec1(5, a);
        vector<simple_string> vec2(vec1);
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        simple_string::initialize_counts();
        array<simple_string> arr1(5, a);
        array<simple_string> arr2(arr1);
        simple_string::print_counts();
    }

    {
        simple_string a("a");
        simple_string b("b");
        simple_string c("c");
        simple_string d("d");

        std::cout << "Vector" << std::endl;
        simple_string::initialize_counts();
        vector<simple_string> vec1(5, a), vec2({a, b}), vec3; // 5cp, 4cp+2destr
        std::cout<<vec1.size() << " "<< vec2.size() << " " << vec3.size() << std::endl;
        //vector<simple_string> vec2({a, b, b});
        vec2.push_back(std::move(c));
        std::cout<<vec1.size() << " "<< vec2.size() << " " << vec3.size() << std::endl;
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        simple_string::initialize_counts();
        array<simple_string> arr1(5, a), arr2({a, b}), arr3;
        arr2.push_back(std::move(d));
        //array<simple_string> arr3(std::move(c));
        simple_string::print_counts();
    }

        {
        simple_string a("Goober");

        std::cout << "Vector" << std::endl;
        vector<simple_string> vec(5);
        vec.push_back(a);
        simple_string::initialize_counts();
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr(5);
        arr.push_back(a);
        simple_string::initialize_counts();
        simple_string::print_counts();
    }

#ifdef TEST_CLEAR
    //Vector                    Array
    //Default: 0                Default: 0
    //Create: 0                 Create: 0
    //Copy: 0                   Copy: 0
    //Assign: 0                 Assign: 0
    //Destruct: 2               Destruct: 2
    //Move Construct: 0         Move Construct: 0
    //Move Assign: 0            Move Assign: 0

    {
        std::cout << "Vector" << std::endl;
        simple_string a("Goober");
        vector<simple_string> vec;
        vec.push_back(a);
        vec.push_back(a);
        simple_string::initialize_counts();
        vec.clear();
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        arr.push_back(a);
        simple_string::initialize_counts();
        arr.clear();
        simple_string::print_counts();
    }
#endif

#ifdef TEST_POP_FRONT
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 0               Copy: 0
    //Assign: 0             Assign: 0
    //Destruct: 1           Destruct: 1
    //Move Construct: 0     Move Construct: 0
    //Move Assign: 2        Move Assign: 2

    {
        simple_string a("Goober");
        simple_string b("Gabber");
        simple_string c("Gupper");

        std::cout << "Vector" << std::endl;
        vector<simple_string> vec;
        vec.push_back(a);
        vec.push_back(b);
        vec.push_back(c);
        simple_string::initialize_counts();
        //note: std::vec does not have pop_front
        vec.erase(vec.begin());
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        arr.push_back(b);
        arr.push_back(c);
        simple_string::initialize_counts();
        arr.pop_front();
        simple_string::print_counts();
    }

#endif

#ifdef TEST_POP_BACK
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 0               Copy: 0
    //Assign: 0             Assign: 0
    //Destruct: 1           Destruct: 1
    //Move Construct: 0     Move Construct: 0
    //Move Assign: 0        Move Assign: 0
    {
        simple_string a("Goober");


        std::cout << "Vector" << std::endl;
        vector<simple_string> vec;
        vec.push_back(a);
        simple_string::initialize_counts();
        vec.pop_back();
        simple_string::print_counts();


        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        simple_string::initialize_counts();
        arr.pop_back();
        simple_string::print_counts();
    }
#endif

#ifdef TEST_PUSH_FRONT_WITH_RESERVE
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 1               Copy: 1
    //Assign: 0             Assign: 0
    //Destruct: 0           Destruct: 0
    //Move Construct: 0     Move Construct: 0
    //Move Assign: 0        Move Assign: 0

    {
        simple_string a("Goober");

        simple_string::initialize_counts();
        std::cout << "Vector" << std::endl;
        vector<simple_string> vec;
        vec.reserve(2);
        vec.insert(vec.begin(), a);
        simple_string::print_counts();

        simple_string::initialize_counts();
        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.reserve(2);
        arr.push_front(a);
        simple_string::print_counts();
    }
#endif

#ifdef TEST_PUSH_FRONT_VEC
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 1               Copy: 1
    //Assign: 0             Assign: 0
    //Destruct: 2           Destruct: 2
    //Move Construct: 2     Move Construct: 2
    //Move Assign: 0        Move Assign: 0

    {
        simple_string a;
        simple_string b("Foob");
        std::cout << "Vector" << std::endl;

        vector<simple_string> vec;
        vec.push_back(a);
        vec.push_back(a);
        simple_string::initialize_counts();
        //note std::vector doesn't have a push_front
        vec.insert(vec.begin(), a);
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        array<simple_string> arr;
        arr.push_back(a);
        arr.push_back(a);
        simple_string::initialize_counts();
        arr.push_front(b);
        simple_string::print_counts();
    }

#endif

#ifdef TEST_PUSH_BACK_NEW_VEC

    //Push back new vec with no reserve
    //
    //Vector                Array
    //Default: 0            Default: 0
    //Create: 0             Create: 0
    //Copy: 1               Copy: 1
    //Assign: 0             Assign: 0
    //Destruct: 0           Destruct: 0
    //Move Construct: 0     Move Construct: 0
    //Move Assign: 0        Move Assign: 0
    {
        simple_string a;

        std::cout << "Vector" << std::endl;
        simple_string::initialize_counts();
        vector<simple_string> vec;
        vec.push_back(a);
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        simple_string::initialize_counts();
        array<simple_string> arr;
        arr.push_back(a);
        simple_string::print_counts();
    }
#endif

#ifdef TEST_INITIALIZER_LIST

    // Test initializer list

    // Vector                  Array
    // Default: 0              Default: 0
    // Create: 0               Create: 0
    // Copy: 4                 Copy: 4
    // Assign: 0               Assign: 0
    // Destruct: 2             Destruct: 2
    // Move Construct: 0       Move Construct: 0
    // Move Assign: 0          Move Assign: 0

    {
        simple_string a;
        simple_string b;
        simple_string c;

        std::cout << "Vector" << std::endl;
        simple_string::initialize_counts();
        vector<simple_string> vec({a, b, c});
        simple_string::print_counts();

        std::cout << "Array" << std::endl;
        simple_string::initialize_counts();
        array<simple_string> arr({a, b, c});
        simple_string::print_counts();
    }
#endif


    return 0;
}