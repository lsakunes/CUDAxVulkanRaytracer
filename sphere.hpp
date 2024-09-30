#ifndef SPHEREHPP
#define SPHEREHPP

#include "hitable.hpp"

class sphere : public hitable {
public:
	__device__ sphere() {};
	__device__ sphere(vec3 cen, float r, material* m) : center0(cen), center1(cen), radius(r), mat_ptr(m) {};
	__device__ sphere(vec3 cen0, vec3 cen1, float r, material* m) : center0(cen0), center1(cen1), radius(r), mat_ptr(m) {};
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
	__device__ vec3 center(float time) const;
	vec3 center0, center1;
	float radius;
	material* mat_ptr;
};

__device__ vec3 sphere::center(float time) const {
	return center0 + time * (center0 - center1);
}

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	vec3 oc = r.origin() - center(r.time());
	auto a = dot(r.direction(), r.direction());
	auto b = dot(oc, r.direction());
	auto c = dot(oc, oc) - radius * radius;
	auto discriminant = b * b - a * c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(b * b - a * c)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.at(rec.t);
			rec.normal = (rec.p - center(r.time())) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
		temp = (-b + sqrt(b * b - a * c)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.at(rec.t);
			rec.normal = (rec.p - center(r.time())) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}

__device__ bool sphere::bounding_box(float t0, float t1, aabb& box) const {
	aabb box0 = aabb(center(t0) - vec3(radius, radius, radius), center(t0) + vec3(radius, radius, radius));
	aabb box1 = aabb(center(t1) - vec3(radius, radius, radius), center(t1) + vec3(radius, radius, radius));
	box = surrounding_box(box0, box1);
	return true;
}
#endif