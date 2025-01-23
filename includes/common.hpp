#ifndef COMMON_HPP
#define COMMON_HPP
#include <hiprand_kernel.h>
#include <hiprand_kernel_rocm.h>
#pragma once

#include <hip/hip_runtime.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <hiprand.hpp>
#include <rocrand.hpp>


// C++ Std Usings

using std::make_shared;
using std::shared_ptr;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

__device__ inline double random_double(hiprandStateXORWOW_t* state){
    return hiprand_uniform<hiprandStateXORWOW_t>(state);

}

__device__ inline double random_double(hiprandStateXORWOW_t* state,double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double(state);
}
//Common Headers
//#include "interval.hpp"


#endif