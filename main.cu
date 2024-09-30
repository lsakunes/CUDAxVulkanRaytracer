#include <iostream>                                                   
#include <fstream>    
#include <time.h>
#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"

#include "vec3.hpp"
#include "color.hpp"
#include "ray.hpp"
#include "hitable_list.hpp"
#include "sphere.hpp"
#include "sphere.hpp"
#include "camera.hpp"
#include "material.hpp"
#include "stats.hpp"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

__device__ float MAXFLOAT = 999;



int numSpheres = 5;

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result != cudaSuccess) {
		std::cerr << "CUDA error = " << cudaGetErrorString(result) << " at " << file << ":" << line << " '" << func << " " << "' \n";
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

//DO NOT TOUCH
void bar(int j,int ny){;;;;;;;;;;
;;;;std::cout<<"\r";;;;;;;;;;;;;;
;;;;std::cout<<"[";;;;;;;;;;;;;;;
;;;;float ratio=(float)j/(ny-1);;
;;;;int space=10*ratio;;;;;;;;;;;
;;;;int equals=10-space;;;;;;;;;;
;;;;for(int i=equals;i>0;i--){;;;
;;;;;;;;std::cout<<"=";;;;;;;;;;;
;;;;};;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;for(int i=space;i>0;i--){;;;;
;;;;;;;;std::cout<<" ";;;;;;;;;;;
;;;;};;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;std::cout<<"]";;;;;;;;;;;;;;;
};;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

__device__ color ray_color(const ray& r, hitable **world, curandState* local_rand_state) {
	vec3 bgColor(0.9, 0.9, 1);
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		int result = (*world)->hit(cur_ray, 0.001f, MAXFLOAT, rec);
		if (result == 1) {
			ray scattered;
			vec3 attenuation;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
		}
		else if (result == 2) {
			return rec.mat_ptr->albedo->value(0,0,rec.p);
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * bgColor;
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0);
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state, int seed) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curand_init(1984 * seed, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(color* fb, int max_x, int max_y, int samples, camera **d_cam, hitable** world, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	for (int s = 0; s < samples; s++) {
		auto u = double(i + curand_uniform(&local_rand_state)) / (max_x - 1);
		auto v = double(j + curand_uniform(&local_rand_state)) / (max_y - 1);
		ray r = (*d_cam)->get_ray(u, v, &local_rand_state);
		col += ray_color(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	fb[pixel_index] = col / float(samples);
}

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_cam, int numSpheres, int nx, int ny, vec3* ranvec, int* perm_x, int* perm_y, int* perm_z, curandState* localState) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		ranvec = perlin_generate(localState[0]);
		perm_x = perlin_generate_perm(localState[1]);
		perm_y = perlin_generate_perm(localState[2]);
		perm_z = perlin_generate_perm(localState[3]);

		//BASE
		d_list[0] = new sphere(vec3(0, -100, -1), 100,
			new lambertian(new marble_texture(ranvec, perm_x, perm_y, perm_z, 30)));
		//d_list[1] = new sphere(vec3(0, -100, -1), 100, 
		//	new lambertian(new checker_texture(vec3(1, 0.2, 0.2), vec3(0.2, 0.2, 1))));

		d_list[1] = new sphere(vec3(0, 0.5, -1), 0.5,
			new metal(new constant_texture(vec3(0.8, 0.2, 0.3)), 0.3));
		d_list[2] = new sphere(vec3(1, 0.22, -1), 0.25,
			new Emit(new constant_texture(vec3(5, 5, 15))));
		d_list[3] = new sphere(vec3(-1, 0.3, -1), vec3(-1, 0.1, -1), 0.1,
			new lambertian(new constant_texture(vec3(0.3, 0.9, 0.4))));
		d_list[4] = new sphere(vec3(-2, 0.4, -1), 0.4,
			new dielectric(1.5));
		*d_world = new hitable_list(d_list, numSpheres);
		float R = cos(PI / 4);

		vec3 lookfrom(-5, 2, 3);
		vec3 lookat(0, 0.5, -1);
		float dist_to_focus = (lookfrom - lookat).length();
		float aperture = 0.05;
		*d_cam = new camera(lookfrom, lookat, vec3(0,1,0), 20, float(nx)/float(ny), aperture, dist_to_focus, 0, 1);
	}
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_cam, int numSpheres, vec3* ranvec, int* perm_x, int* perm_y, int* perm_z) {
	for (int i = 0; i < numSpheres; i++) {
		delete ((sphere*)d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete ranvec;
	delete perm_x;
	delete perm_y;
	delete perm_z;
	delete* d_world;
	delete* d_cam;
}

int main() {
	//hmmmm
	cudaDeviceSetLimit(cudaLimitStackSize, 4096);
	//hmmmmmmmm

	int nx = 1200;
	int ny = 600;
	int samples = 500;

	int tx = 8;
	int ty = 8;

	std::cerr << "Rendering a " << nx << "x" << ny << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	clock_t start, stop;
	start = clock();

	int num_pixels = nx * ny;
	size_t fb_size = num_pixels * sizeof(color);

	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init << <blocks, threads >> > (nx, ny, d_rand_state, 0);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	vec3* ranvec;
	checkCudaErrors(cudaMalloc((void**)&ranvec, sizeof(vec3*)));
	int* perm_x;
	checkCudaErrors(cudaMalloc((void**)&perm_x, sizeof(int*)));
	int* perm_y;
	checkCudaErrors(cudaMalloc((void**)&perm_y, sizeof(int*)));
	int* perm_z;
	checkCudaErrors(cudaMalloc((void**)&perm_z, sizeof(int*)));

	camera** d_cam;
	checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(camera*)));
	hitable** d_list;
	checkCudaErrors(cudaMalloc((void**)& d_list, numSpheres * sizeof(hitable*)));
	hitable** d_world;
	checkCudaErrors(cudaMalloc((void**)& d_world, sizeof(hitable*)));
	create_world<<<1,1>>>(d_list, d_world, d_cam, numSpheres, nx, ny, ranvec, perm_x, perm_y, perm_z, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	color* fb;
	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

	color* finalFb;
	checkCudaErrors(cudaMallocManaged((void**)&finalFb, fb_size));

	render << <blocks, threads >> > (fb, nx, ny, samples, d_cam, d_world, d_rand_state);
	/*checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());*/
	cudaDeviceSynchronize();

	stop = clock();
	float timer_seconds = ((float)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";

	std::ofstream file_streamRaw;
	file_streamRaw.open("fileRaw.ppm");

	file_streamRaw << "P3\n" << nx << ' ' << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; j--) {
		for (int i = 0; i < nx; i++) {
			size_t pixel_index = j * nx + i;
			write_color(file_streamRaw, fb[pixel_index]);
		}
		if (j % 100 == 0)
			bar(j, ny);
	}

	//denoise << <blocks, threads >> > (fb, finalFb, nx, ny, 7);
	//checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaDeviceSynchronize());

	//std::ofstream file_stream;
	//file_stream.open("file.ppm");

	//file_stream << "P3\n" << nx << ' ' << ny << "\n255\n";

	//for (int j = ny - 1; j >= 0; j--) {
	//	for (int i = 0; i < nx; i++) {
	//		size_t pixel_index = j * nx + i;
	//		write_color(file_stream, finalFb[pixel_index]);
	//	}
	//	if (j%100 == 0)
	//		bar(j, ny);
	//}
	checkCudaErrors(cudaDeviceSynchronize());
	free_world << <1, 1 >> > (d_list, d_world, d_cam, numSpheres, ranvec, perm_x, perm_y, perm_z);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_cam));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();
}



