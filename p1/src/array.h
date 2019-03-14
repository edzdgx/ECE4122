#pragma once

#include <iostream>

//students should not change this

using size_t = std::size_t;

template<typename T> class array_iterator;

template<typename T>
class array {
public:
    //default constructor
    array() {
        m_size = 0;
        m_reserved_size = 0;
        m_elements = nullptr;
    }

    //assign the value of list to your array respectively. Besides you have to
    //take into consideration the memory allocation.
    //initialize array with elements in initializer
    array(std::initializer_list<T> L) {
        //std::initializer_list<T>::iterator<T> it;
        //L::iterator it;

        m_size = L.size();
        m_reserved_size = m_size;
        m_elements = (T*) malloc(sizeof(T) * m_size);

        // for (auto it=L.begin(); it != L.end(); it++) {
        //     new ()
        // }
        for (int i = 0; i < m_size; i++) {
            new (m_elements + i) T(*(L.begin()+i));
        }
    }

    //copy constructor
    array(const array& rhs) {
        m_size = rhs.m_size;
        m_reserved_size = rhs.m_reserved_size;
        if (m_size > 0) {
            m_elements = (T*) malloc(sizeof(T) * m_size);
            for (int i = 0; i < m_size; i++) {
                new (m_elements+i) T(rhs.m_elements[i]);
            }
        }
    }

    //move constructor
    array(array&& rhs) : m_elements(nullptr), m_size(0), m_reserved_size(0)
    {
        m_size = rhs.m_size;
        m_reserved_size = rhs.m_reserved_size;
        m_elements = (T*) malloc(sizeof(T) * m_size);
        for (int i = 0; i < m_size; i++) {
            new (m_elements+i) T(std::move(rhs.m_elements[i]));
            //m_elements[i] = rhs[i];
        }
//
        rhs.m_elements = nullptr;
        rhs.m_size = 0;
        rhs.m_reserved_size = 0;
    }


    //construct with initial "reserved" size
    array(size_t s) {
        m_size = 0;
        m_reserved_size = s;
        m_elements = nullptr;
    }

    //construct with n copies of t
    array(size_t n, const T& t) {
        m_size = n;
        m_reserved_size = m_size;
        m_elements = (T*) malloc(sizeof(T) * m_size);
        if(n > 0) {
            for (int i = 0; i < n; i++) {
                new (&m_elements[i]) T(t);
            }
        }
    }

    //destructor
    ~array()
    {
        if(m_elements) {
            for (int i = 0; i < m_size; i++) {
                m_elements[i].~T();
            }
            free(m_elements);
        }
        m_size = 0;
        m_reserved_size = 0;
    }

    //ensure enough memory for n elements
    void reserve(size_t n) {
        this -> m_reserved_size = n;
        m_elements = (T*) malloc(sizeof(T) * m_reserved_size);
    }


    //add to end of vector
    void push_back(const T& rhs) {

        if(m_size == m_reserved_size) { // no more reserved space
            m_size++;
            m_reserved_size++;  // increment
            //std::cout <<"m_size = " << m_size << " m_reserved_size = "<<m_reserved_size << std::endl;
            T* t = (T*) malloc(sizeof(T) * m_reserved_size); // allocate mem
            for(int i = 0; i < m_size - 1; i++) {
                new (&t[i]) T((T&&)m_elements[i]); // in-place new
                m_elements[i].~T();
            }
            free(m_elements);
            new (&t[m_size - 1]) T(rhs);
            m_elements = t; // copying
        } else {    // if have reserved space
            m_size++;
            new (&m_elements[m_size - 1]) T(rhs);
        }
    }

    //add to front of vector
    void push_front(const T& rhs) {
        if(m_size == m_reserved_size) { // no more reserved space

            m_size++;
            m_reserved_size++;  // increment
            T* t = (T*) malloc(sizeof(T) * m_reserved_size); // allocate mem

            for(int i = 0; i < m_size; i++) {
                new (&t[i]) T((T&&)m_elements[i - 1]); // in-place new move
                m_elements[i - 1].~T();
            }

            free(m_elements);
            new (t) T(rhs); // copy
            m_elements = t;

        } else {

            for (int i = m_size; i > 1; i--) { // do a reverse for loop
                new (&m_elements[i]) T((T&&)m_elements[i - 1]); // move back every element
                m_elements[i - 1].~T(); // call destructor on previous pos
            }

            m_size++; // increment
            new (m_elements) T(rhs); // copy
        }

    }

    //remove last element
    void pop_back() {
        if (m_size > 0) {
            m_elements[m_size - 1].~T();
            m_size--;
        }
    }

    //remove first element
    void pop_front() {
        if (m_size > 0) {
            m_elements[0].~T(); // remove first ele
        }
        // move all ele to its prev pos
        for (int i = 0; i < m_size - 1; i++) {
            new (&m_elements[i]) T((T&&)m_elements[i + 1]);
            m_elements[i + 1].~T();
        }
        m_size--;
    }

    //return reference to first element
    T& front() const {
        if(m_elements != nullptr) {
            return m_elements[0];
        }
    }

    //return reference to last element
    T& back() const {
        if(m_elements != nullptr) {
            return m_elements[m_size - 1];
        }
    }

    //return reference to specified element
    const T& operator[](size_t i) const {
        return m_elements[i];
    }

    //return reference to specified element
    T& operator[](size_t i) {
        return m_elements[i];
    }

    //return number of elements
    size_t length() const {
        return m_size;
    }

    //returns true if empty
    bool empty() const {
        return m_size == 0;
    }

    //remove all elements
    void clear() {
        for (int i = 0; i < m_size; i++) {
            m_elements[i].~T(); // call destructor
        }
        m_size = 0; // reset it to 0
    }

    //obtain iterator to first element
    array_iterator<T> begin() const {
        return array_iterator<T>(m_elements);
    }

    //obtain iterator to one beyond element
    array_iterator<T> end() const {
        return array_iterator<T>(m_elements + m_size);
    }

    //remove specified element
    void erase(const array_iterator<T>& iter) {
        array_iterator<T> arr;
        int idx = 0;
        // find position of element to be erased
        m_size--; // decrement m_size

        for (arr = m_elements; arr != m_elements + m_size - 1; arr++) {
            idx++;
            if (arr == iter) {
                break; // if find the position, break out of the loop
            }
        }

        m_elements[idx - 1].~T(); // the actual deletion

        // move left by 1 using move
        for (int i = idx - 1; i < m_size; i++) {
            new (&m_elements[i]) T((T&&)m_elements[i + 1]); // in-place new with move
            m_elements[i + 1].~T(); // destr
        }

    }

    //insert element right before itr
    void insert(const T& rhs, const array_iterator<T>& iter) {
        m_size++;
        T* t = (T*) malloc(sizeof(T) * m_size);

        // elements before iter position
        array_iterator<T> arr; // temp iter to loop
        int idx = 0; // the index to insert
        for(arr = m_elements; arr != m_elements + m_size - 1; arr++) {
            idx++;
            if(arr==iter) {
                break;
            }
        }

        // actial insert
        for(int i = 0; i < idx - 1; i++) {
            new (&t[i]) T((T&&)m_elements[i]); // in place new
            m_elements[i].~T();
        }

        new (&t[idx - 1]) T(rhs); // insert the element rhs into idx - 1

        // append rest of the elements
        for(int i = 0; i < m_size; i++) {
            new (&t[i]) T((T&&)m_elements[i - 1]);
            m_elements[i - 1].~T();
        }
        free(m_elements);
        m_elements = t;
    }


private:
    T* m_elements;              //points to actual elements
    size_t m_size;              //number of elements
    size_t m_reserved_size;     //number of reserved elements
};






//************************************************************
template<typename T>
class array_iterator {
public:
    array_iterator() {
        m_current = nullptr;
    }

    array_iterator(T* c) {
        m_current = c;
    }

    array_iterator(const array_iterator<T>& rhs) {
        m_current = rhs.m_current;
    }

    T& operator*() const {
        return *m_current;
    }

    array_iterator<T> operator++() {
        m_current++;
        return *this;
    }

    array_iterator<T> operator++(int __unused) {
        array_iterator<T> c(m_current);
        m_current++;
        return c;
    }
    bool operator != (const array_iterator<T>& rhs) const {
        return m_current != rhs.m_current;
    }

    bool operator == (const array_iterator<T>& rhs) const {
        return m_current == rhs.m_current;
    }

private:
    T* m_current;
    // I want my array class to be able to access m_current even though it's private
    // 'friend' allows this to happen
    friend class array<T>;
};