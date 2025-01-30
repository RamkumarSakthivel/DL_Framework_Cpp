#include <iostream>

#include "tensor.h"

Tensor::Tensor(const size_t& rows, const size_t& cols, const double& initval)
    :rows(rows) , cols(cols) , data(rows*cols,initval){
}

double& Tensor::operator()(const size_t& i, const size_t& j) {
    assert((i < rows&& j < cols) && "Index/Indices out of bounds");
    return data[i * cols + j];
}

const double& Tensor::operator()(const size_t& i, const size_t& j) const {
    assert((i < rows&& j < cols) && "Index/Indices out of bounds");
    return data[i * cols + j];
}

Tensor Tensor::operator+(const Tensor& other) const {
    assert((rows == other.rows && cols == other.cols) && "Tensors size mismatch... cannot perform element wise addition");
    Tensor ret_tensor(rows, cols);
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            ret_tensor(i,j) = (*this)(i,j) + other(i,j);
        }
    }
    return ret_tensor;
}

Tensor Tensor::operator-(const Tensor& other) const {
    assert((rows == other.rows && cols == other.cols) && "Tensors size mismatch... cannot perform element wise subtraction");
    Tensor ret_tensor(rows, cols);
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            ret_tensor(i, j) = (*this)(i, j) - other(i, j);
        }
    }
    return ret_tensor;
}

Tensor Tensor::operator*(const Tensor& other) const {
    assert((rows == other.rows && cols == other.cols) && "Tensors size mismatch... cannot perform element wise multiplication");
    Tensor ret_tensor(rows, cols);
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            ret_tensor(i, j) = (*this)(i, j) * other(i, j);
        }
    }
    return ret_tensor;
}

Tensor Tensor::operator/(const Tensor& other) const {
    assert((rows == other.rows && cols == other.cols) && "Tensors size mismatch... cannot perform element wise division");
    Tensor ret_tensor(rows, cols);
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            ret_tensor(i, j) = (*this)(i, j) / other(i, j);
        }
    }
    return ret_tensor;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    assert((rows == other.rows && cols == other.cols) && "Tensors size mismatch... cannot perform inplace element wise addition");
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            (*this)(i, j) += other(i, j);
        }
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    assert((rows == other.rows && cols == other.cols) && "Tensors size mismatch... cannot perform inplace element wise subtraction");
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
           (*this)(i, j) -= other(i, j);
        }
    }
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    assert((rows == other.rows && cols == other.cols) && "Tensors size mismatch... cannot perform inplace element wise multiplication");
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            (*this)(i, j) *= other(i, j);
        }
    }
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    assert((rows == other.rows && cols == other.cols) && "Tensors size mismatch... cannot perform inplace element wise division");
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            (*this)(i, j) /= other(i, j);
        }
    }
    return *this;
}

Tensor Tensor::operator+(const double& scalar) const {
    Tensor ret_tensor(rows, cols);
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            ret_tensor(i, j) = (*this)(i, j) + scalar;
        }
    }
    return ret_tensor;
}

Tensor Tensor::operator-(const double& scalar) const {
    Tensor ret_tensor(rows, cols);
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            ret_tensor(i, j) = (*this)(i, j) - scalar;
        }
    }
    return ret_tensor;
}

Tensor Tensor::operator*(const double& scalar) const {
    Tensor ret_tensor(rows, cols);
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            ret_tensor(i, j) = (*this)(i, j) * scalar;
        }
    }
    return ret_tensor;
}

Tensor Tensor::operator/(const double& scalar) const {
    Tensor ret_tensor(rows, cols);
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            ret_tensor(i, j) = (*this)(i, j) / scalar;
        }
    }
    return ret_tensor;
}

Tensor& Tensor::operator+=(const double& scalar) {
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            (*this)(i, j) += scalar;
        }
    }
    return *this;
}

Tensor& Tensor::operator-=(const double& scalar) {
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            (*this)(i, j) -= scalar;
        }
    }
    return *this;
}

Tensor& Tensor::operator*=(const double& scalar) {
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            (*this)(i, j) *= scalar;
        }
    }
    return *this;
}

Tensor& Tensor::operator/=(const double& scalar) {
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            (*this)(i, j) /= scalar;
        }
    }
    return *this;
}

Tensor Tensor::operator^(const Tensor& other) const {
    assert((cols == other.rows) && "Tensors size mismatch... cannot perform matrix multiplication");
    Tensor ret_tensor(rows, other.cols);
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < other.cols; ++j) {
            for (auto k = 0; k < cols; ++k) {
                ret_tensor(i, j) += (*this)(i, k) * other(k, j);
            }
        }
    }
    return ret_tensor;
}


void Tensor::print() {
    for (auto i = 0; i < rows; ++i) {
        for (auto j = 0; j < cols; ++j) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}