#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <xmmintrin.h>

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
	int ncpus = CPU_COUNT(&cpu_set);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */ // using calloc to initialze to 0
    int* image = (int*)calloc(width * height, sizeof(int));
    assert(image);

    // MPI
    int size, rank;
    MPI_Init(&argc, &argv);  
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    __m128d two = _mm_set_pd1(2);
    double row_step = ((upper - lower) / height);
    double col_step = ((right - left) / width);
    int offset;

#pragma omp parallel for schedule(dynamic)
    for (int j = rank; j < height; j += size) {
        // vectorization
        offset = j * width;
        int index[2] = {0, 1}, cur = 2;
        int repeats[2] = {0, 0};
        __m128d y0_v = _mm_set_pd(j * row_step + lower, j * row_step + lower);
        double x0[2] = {index[0] * col_step + left, index[1] * col_step + left};
        __m128d x0_v = _mm_load_pd(x0);
        __m128d x_v = _mm_set_pd(0, 0);
        __m128d y_v = _mm_set_pd(0, 0);
        __m128d length_squared_v = _mm_set_pd(0, 0);
        __m128d x2_v = _mm_mul_pd(x_v, x_v);
        __m128d y2_v = _mm_mul_pd(y_v, y_v);

        while (index[0] < width && index[1] < width) {
            y_v = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(two, x_v), y_v), y0_v); // y = 2 * x * y + y0
            x_v = _mm_add_pd(_mm_sub_pd(x2_v, y2_v), x0_v); // x = x * x - y * y + x0
            x2_v = _mm_mul_pd(x_v, x_v);
            y2_v = _mm_mul_pd(y_v, y_v);
            length_squared_v = _mm_add_pd(x2_v, y2_v); // length_squared = x * x + y * y
            repeats[0]++;
            repeats[1]++;

            if (repeats[0] == iters || length_squared_v[0] >= 4) { // index[0] done
                image[offset + index[0]] = repeats[0];
                index[0] = cur++;
                repeats[0] = 0;
                x0_v[0] = index[0] * col_step + left;
                x_v[0] = y_v[0] = length_squared_v[0] = x2_v[0] = y2_v[0] = 0;
            }
            if (repeats[1] == iters || length_squared_v[1] >= 4) { // index[1] done
                image[offset + index[1]] = repeats[1];
                index[1] = cur++;
                repeats[1] = 0;
                x0_v[1] = index[1] * col_step + left;
                x_v[1] = y_v[1] = length_squared_v[1] = x2_v[1] = y2_v[1] = 0;
            }
        }
        // do the remaining part
        if (index[0] < width) {
            while (repeats[0] < iters && length_squared_v[0] < 4) {
                y_v[0] = 2 * x_v[0] * y_v[0] + y0_v[0];
                x_v[0] = x2_v[0] - y2_v[0] + x0_v[0];
                x2_v[0] = x_v[0] * x_v[0];
                y2_v[0] = y_v[0] * y_v[0];
                length_squared_v[0] = x2_v[0] + y2_v[0];
                repeats[0]++;
            }
            image[offset + index[0]] = repeats[0];
        }
        else if (index[1] < width) {
            while (repeats[1] < iters && length_squared_v[1] < 4) {
                y_v[1] = 2 * x_v[1] * y_v[1] + y0_v[1];
                x_v[1] = x2_v[1] - y2_v[1] + x0_v[1];
                x2_v[1] = x_v[1] * x_v[1];
                y2_v[1] = y_v[1] * y_v[1];
                length_squared_v[1] = x2_v[1] + y2_v[1];
                repeats[1]++;
            }
            image[offset + index[1]] = repeats[1];
        }
    }

    int* finalImage = (int*)calloc(width * height, sizeof(int));
    MPI_Reduce(image, finalImage, width * height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /* draw and cleanup */
    if (rank == 0)
        write_png(filename, iters, width, height, finalImage);
    free(image);

    MPI_Finalize();
}
