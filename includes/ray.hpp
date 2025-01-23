#ifndef RAY_HPP
#define RAY_HPP

#include "common.hpp"
#include "vec3.hpp"

class ray {
public:
    __device__ __host__ ray() {}

    __device__ __host__ ray(const vec3 origin, const vec3& direction) : orig(origin), dir(direction) {}

    __device__ __host__ const vec3& origin() const {return orig;}
    __device__ __host__ const vec3& direction() const {return dir;}

    __device__ __host__ vec3 at(double t) const {
        return orig + t*dir;
    }

private: 
    vec3 orig;
    vec3 dir;


};

#endif