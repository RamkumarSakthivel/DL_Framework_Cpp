#pragma once

#include <vector>

class Tensor {
private:
    size_t rows, cols;
    std::vector<double> data;
public:
    //constructors
    Tensor(const size_t& rows, const size_t& cols, const double& initval);
    //operators
    double& operator()(const size_t& rows, const size_t& cols);
    const double& operator()(const size_t& rows, const size_t& cols) const;
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
};