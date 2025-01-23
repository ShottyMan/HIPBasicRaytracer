#ifndef VEC3_HPP
#define VEC3_HPP

#include "common.hpp"
#include <hiprand_kernel.h>
#include <hiprand_kernel_rocm.h>
#pragma once

class vec3{

public: 
    double x;
    double y;
    double z;
    __host__ __device__ vec3(double x, double y, double z)
    {

        this->x = x;
        this->y = y;
        this->z = z;

    }

    __host__ __device__ vec3()
    {
        this->x = 0;
        this->y = 0;
        this->z = 0;

    }

    __host__ __device__ vec3& operator+=(const vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    __host__ __device__ inline double length_squared() const {
        return x*x + y*y + z*z;
    }

    __host__ __device__ vec3 operator-() const { return vec3(-x, -y, -z); }

    __device__ static vec3 random(hiprandStateXORWOW_t *state) {
        return vec3(random_double(state), random_double(state), random_double(state));

    }

    __device__ static vec3 random(hiprandStateXORWOW_t *state, double min, double max) {
        return vec3(random_double(state, min, max), random_double(state, min, max), random_double(state, min, max));

    }



};


__host__ __device__ inline double dot(const vec3& u, const vec3& v) {
    return u.x * v.x
         + u.y * v.y
         + u.z * v.z;
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.y * v.z - u.z * v.y,
                u.z * v.x - u.x * v.z,
                u.x * v.y - u.y * v.x);
}

__host__ __device__ inline double vec_len(vec3 vec)
{

    return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);

}
__host__ __device__ inline vec3 operator*(vec3 v, double db)
{

    return vec3(v.x * db, v.y*db, v.z*db);

}

__host__ __device__ inline vec3 operator*(double db, vec3 v)
{

    return v * db;

}
__host__ __device__ inline vec3 operator/(vec3 v, double db)
{

    return v * (1/db);

}


__device__ __host__ inline vec3 unit_vector(vec3 vec)
{

    return vec / vec_len(vec);

}
__device__ __host__ inline vec3 operator+(vec3 vec, vec3 vec2)
{

    return vec3(vec.x + vec2.x, vec.y + vec2.y, vec.z + vec2.z);

}
__device__ __host__ inline vec3 operator-(vec3 vec, vec3 vec2)
{

    return vec3(vec.x - vec2.x, vec.y - vec2.y, vec.z - vec2.z);

}



__host__ __device__ inline double dot(const dim3& u, const dim3& v) {
    return u.x * v.x
         + u.y * v.y
         + u.z * v.z;
}

__host__ __device__ inline dim3 cross(const dim3& u, const dim3& v) {
    return dim3(u.y * v.z - u.z * v.y,
                u.z * v.x - u.x * v.z,
                u.x * v.y - u.y * v.x);
}

inline double vec_len(dim3 vec)
{

    return sqrt((vec.x * vec.x) + (vec.y * vec.y) + (vec.z * vec.z));

}
__host__ __device__ inline dim3 operator*(dim3 v, double db)
{

    return dim3(v.x * db, v.y*db, v.z*db);

}

__host__ __device__ inline dim3 operator*(double db, dim3 v)
{

    return v * db;

}
__host__ __device__ inline dim3 operator/(dim3 v, double db)
{

    return v * (1/db);

}


__device__ __host__ inline dim3 unit_vector(dim3 vec)
{

    return vec / vec_len(vec);

}
__device__ __host__ inline dim3 operator+(dim3 vec, dim3 vec2)
{

    return dim3(vec.x + vec2.x, vec.y + vec2.y, vec.z + vec2.z);

}
__device__ __host__ inline dim3 operator-(dim3 vec, dim3 vec2)
{

    return dim3(vec.x - vec2.x, vec.y - vec2.y, vec.z - vec2.z);

}

__device__ inline vec3 random_unit_vector(hiprandStateXORWOW_t* state) {
    while (true) {
        auto p = vec3::random(state,-1,1);
        auto lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
}
__device__ inline vec3 random_on_hemisphere(const vec3& normal, hiprandStateXORWOW_t *state) {
    vec3 on_unit_sphere = random_unit_vector(state);
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

#endif