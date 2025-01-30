#pragma once

#include <vector>
#include <cassert>

class Tensor {
private:
    size_t rows, cols;
    std::vector<double> data;
public:
    //constructors

    Tensor(const size_t& rows, const size_t& cols, const double& initval=0.0);

    //operators

    double& operator()(const size_t& i, const size_t& j);
    const double& operator()(const size_t& i, const size_t& j) const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    Tensor operator+(const double& scalar) const;
    Tensor operator-(const double& scalar) const;
    Tensor operator*(const double& scalar) const;
    Tensor operator/(const double& scalar) const;

    Tensor& operator+=(const double& scalar);
    Tensor& operator-=(const double& scalar);
    Tensor& operator*=(const double& scalar);
    Tensor& operator/=(const double& scalar);

    Tensor operator^(const Tensor& other) const;

    //member functions

    void print();
    
};