#ifndef COLOR_HPP
#define COLOR_HPP

#include "common.hpp"
#include "hittable_list.hpp"
#include "hittable.hpp"
#include "sphere.hpp"
#include <hiprand_kernel.h>

using color = vec3;

__host__ __device__ inline double hit_sphere(const vec3& center, double radius, const ray& r) {
    vec3 oc = center - r.origin();
    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius*radius;
    auto discriminant = h*h - a*c;

    if (discriminant < 0) {
        return -1.0;
    } else {
        return (h - sqrt(discriminant)) / a;
    }
}

inline double linear_to_gamma(double linear_component)
{
    if (linear_component > 0)
        return std::sqrt(linear_component);

    return 0;
}

__host__ __device__ inline color ray_color(const ray& r, const hittable_list& world, hiprandStateXORWOW_t *state, int depth) {
    hit_record rec;
    bool result = world.hit(r, interval(0,infinity),rec);
    if (depth <= 0)
        return vec3(0,0,0);
    if (result) {
        vec3 direction = random_on_hemisphere(rec.normal, state);
        return 0.5 * ray_color(ray(rec.point, direction), world, state, depth-1);
        //return 0.5 * (rec.normal + color(1,1,1));
        //return 0.5 * vec3(1,1,1);
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

__host__ __device__ inline color ray_color(const ray& r, const hittable_list& world, hit_record& rec) {
    bool result = world.hit(r, interval(0,infinity),rec);
    if (result) {
        //vec3 direction = random_on_hemisphere(rec.normal, state);
        //return 0.5 * ray_color(ray(rec.point, direction), world, state);
        //return 0.5 * (rec.normal + color(1,1,1));
        return vec3(.5,.5,.5);
        rec.hit = true;
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y + 1.0);
    rec.hit = false;
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}
__host__ __device__ inline color ray_color(const ray& r, const hittable_list& world, hit_record& rec, bool* hit) {
    bool result = world.hit(r, interval(0.000000000000001,infinity),rec);
    if (result) {
        //vec3 direction = random_on_hemisphere(rec.normal, state);
        //return 0.5 * ray_color(ray(rec.point, direction), world, state);
        //return 0.5 * (rec.normal + color(1,1,1));
        *hit = true;
        return vec3(.5,.5,.5);
        
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y + 1.0);
    *hit = false;
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

__host__ __device__ inline color ray_color(const ray& r, const hittable_list& world, hit_record& rec, bool* hit, int idx, int idy) {
    bool result = world.hit(r, interval(0,infinity),rec);
    if (result) {
        //vec3 direction = random_on_hemisphere(rec.normal, state);
        //return 0.5 * ray_color(ray(rec.point, direction), world, state);
        //return 0.5 * (rec.normal + color(1,1,1));
        return vec3(.5,.5,.5);
        *hit = true;
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5*(unit_direction.y + 1.0);
    *hit = false;
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}

inline vec3 dim3tovec3(dim3 int_vector)
{

    return vec3 {(double)int_vector.x, (double)int_vector.y, (double)int_vector.z};

}

#endif