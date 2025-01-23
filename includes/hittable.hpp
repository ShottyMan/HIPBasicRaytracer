#ifndef HITTABLE_HPP
#define HITTABLE_HPP


#include "vec3.hpp"
#include "ray.hpp"
#include "interval.hpp"
class hit_record {

public:
    vec3 point;
    vec3 normal;
    bool hit = false;
    double t;

    bool front_face;

    __device__ __host__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }

};

class hittable {

public:
    virtual ~hittable() = default;

    __device__ __host__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;

};

#endif