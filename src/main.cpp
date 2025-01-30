#include <iostream>

#include "tensor.h"

int main() {
    std::cout << "DL Framework Implementation in Progress...\n" << std::endl;

    // Test 1: Create tensors
    Tensor A(3, 3, 2.0); // A 3x3 tensor initialized with 2.0
    Tensor B(3, 3, 3.0); // B 3x3 tensor initialized with 3.0

    // Print initial tensors
    std::cout << "Tensor A:" << std::endl;
    A.print();
    std::cout << "Tensor B:" << std::endl;
    B.print();

    // Test 2: Tensor addition
    Tensor sum = A + B;
    std::cout << "A + B:" << std::endl;
    sum.print();

    // Test 3: Tensor subtraction
    Tensor diff = A - B;
    std::cout << "A - B:" << std::endl;
    diff.print();

    // Test 4: Tensor multiplication
    Tensor prod = A * B;
    std::cout << "A * B:" << std::endl;
    prod.print();

    // Test 5: Tensor division
    Tensor div = A / B;
    std::cout << "A / B:" << std::endl;
    div.print();

    // Test 6: Scalar operations
    double scalar = 5.0;

    // Tensor + scalar
    Tensor sum_scalar = A + scalar;
    std::cout << "A + " << scalar << ":" << std::endl;
    sum_scalar.print();

    // Tensor - scalar
    Tensor diff_scalar = A - scalar;
    std::cout << "A - " << scalar << ":" << std::endl;
    diff_scalar.print();

    // Tensor * scalar
    Tensor prod_scalar = A * scalar;
    std::cout << "A * " << scalar << ":" << std::endl;
    prod_scalar.print();

    // Tensor / scalar
    Tensor div_scalar = A / scalar;
    std::cout << "A / " << scalar << ":" << std::endl;
    div_scalar.print();

    // Test 7: In-place scalar operations
    A += scalar;
    std::cout << "A after A += " << scalar << ":" << std::endl;
    A.print();

    A -= scalar;
    std::cout << "A after A -= " << scalar << ":" << std::endl;
    A.print();

    A *= scalar;
    std::cout << "A after A *= " << scalar << ":" << std::endl;
    A.print();

    A /= scalar;
    std::cout << "A after A /= " << scalar << ":" << std::endl;
    A.print();

    // Test 8: Matrix multiplication (A ^ B)
    Tensor matmul = A ^ B; // Matrix multiplication (A * B)
    std::cout << "A ^ B (Matrix multiplication):" << std::endl;
    matmul.print();

    return 0;

    return 0;
}