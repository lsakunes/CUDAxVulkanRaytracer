#ifndef TEXTUREHPP
#define TEXTUREHPP

#include <cstdlib>
#include "vec3.hpp"
#include "perlin.hpp"

class _texture {
public:
	__device__ _texture() {}
	__device__ virtual vec3 value(float u, float v, const vec3& p) const = 0;
};

class constant_texture : public _texture {
public:
	__device__ constant_texture() {}
	__device__ constant_texture(vec3 c) : color(c) {}
	__device__ virtual vec3 value(float u, float v, const vec3& p) const {
		return color;
	}
	color color;
};

class checker_texture : public _texture {
public:
	__device__ checker_texture() {}
	__device__ checker_texture(color t0, color t1) : even(t0), odd(t1) {	}
	__device__ virtual vec3 value(float u, float v, const vec3& p) const {
		float sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
		if (sines < 0)
			return odd;
		else 
			return even;
	}

	color odd;
	color even;
};

class noise_texture : public _texture {
public:
	__device__ noise_texture(vec3* r, int* x, int* y, int* z, float s) : ranvec(r), perm_x(x), perm_y(y), perm_z(z), scale(s) {}

	__device__ virtual vec3 value(float u, float v, const vec3& p) const {
		vec3 retVal = vec3(0.8, 0.8, 0.8) * noise.smooth_noise(scale*p, ranvec, perm_x, perm_y, perm_z);
		retVal += vec3(0.2, 0.2, 0.2);
		return retVal;
	}

	perlin noise;
	float scale;
	vec3* ranvec;
	int* perm_x;
	int* perm_y;
	int* perm_z;
};

class marble_texture : public _texture {
public:
	__device__ float turb(const vec3& p, int depth = 7) const {
		float accum = 0;
		vec3 temp_p = p;
		float weight = 1.0;
		for (int i = 0; i < depth; i++) {
			accum += weight * noise.smooth_noise(temp_p, ranvec, perm_x, perm_y, perm_z);
			weight *= 0.5;
			temp_p *= 2;
		}
		return accum > 0 ? accum : -accum;
	}

	__device__ marble_texture(vec3* r, int* x, int* y, int* z, float s) : ranvec(r), perm_x(x), perm_y(y), perm_z(z), scale(s) {}

	__device__ virtual vec3 value(float u, float v, const vec3& p) const {
		vec3 retVal = vec3(1, 1, 1)*0.5 * (1 + sin(scale * p.z() + 10 * turb(p)));;
		return retVal;
	}

	perlin noise;
	float scale;
	vec3* ranvec;
	int* perm_x;
	int* perm_y;
	int* perm_z;
};

// --------------
//	   BROKEN 
// --------------
//class cool_noise_texture : public _texture {
//public:
//	__device__ cool_noise_texture(vec3* r, int* x, int* y, int* z, float s) : ranvec(r), perm_x(x), perm_y(y), perm_z(z), scale(s) {}
//	
//	__device__ float turb(const vec3& p, int depth = 7) const {
//		float accum = 0;
//		vec3 temp_p = p;
//		printf("%d\n", temp_p.length_squared());
//		float weight = 1.0;
//		for (int i = 0; i < depth; i++) {
//			float temp = noise.smooth_noise(temp_p, ranvec, perm_x, perm_y, perm_z);
//			accum += weight*temp;
//			weight *= 0.5;
//			temp_p *= 2;
//		}
//		return accum>0? accum:-accum;
//	}
//	__device__ virtual vec3 value(float u, float v, const vec3& p) const {
//		vec3 retVal = vec3(1, 0.1, 0.1) * turb(scale*p);
//		return retVal;
//	}
//
//	perlin noise;
//	float scale;
//	vec3* ranvec;
//	int* perm_x;
//	int* perm_y;
//	int* perm_z;
//};


#endif