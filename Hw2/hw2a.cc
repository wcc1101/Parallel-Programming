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
#include <pthread.h>
#include <xmmintrin.h>

int iters, width, height, *image, row;
double left, right, lower, upper, row_step, col_step;
pthread_mutex_t mutex;
__m128d two = _mm_set_pd1(2);

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

void *Cal(void* arg) {
    int row_local, offset;

    while (true) {
        pthread_mutex_lock(&mutex);
        if (row == height) { // end
            pthread_mutex_unlock(&mutex);
            break;
        }
        else // next row
            row_local = row++;
        pthread_mutex_unlock(&mutex);

        // vectorization
        offset = row_local * width;
        int index[2] = {0, 1}, cur = 2;
        int repeats[2] = {0, 0};
        __m128d y0_v = _mm_set_pd(row_local * row_step + lower, row_local * row_step + lower);
        double x0[2] = {index[0] * col_step + left, index[1] * col_step + left};
        __m128d x0_v = _mm_load_pd(x0);
        __m128d x_v = _mm_set_pd(0, 0);
        __m128d y_v = _mm_set_pd(0, 0);
        __m128d length_squared_v = _mm_set_pd(0, 0);
        __m128d x2_v = _mm_mul_pd(x_v, x_v);
        __m128d y2_v = _mm_mul_pd(y_v, y_v);

        while (index[0] < width || index[1] < width) {
            y_v = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(two, x_v), y_v), y0_v); // y = 2 * x * y + y0
            x_v = _mm_add_pd(_mm_sub_pd(x2_v, y2_v), x0_v); // x = x * x - y * y + x0
            x2_v = _mm_mul_pd(x_v, x_v);
            y2_v = _mm_mul_pd(y_v, y_v);
            length_squared_v = _mm_add_pd(x2_v, y2_v); // length_squared = x * x + y * y
            repeats[0]++;
            repeats[1]++;

            if (index[0] < width && (repeats[0] == iters || length_squared_v[0] >= 4)) { // index[0] done
                image[offset + index[0]] = repeats[0];
                index[0] = cur++;
                repeats[0] = 0;
                x0_v[0] = index[0] * col_step + left;
                x_v[0] = y_v[0] = length_squared_v[0] = x2_v[0] = y2_v[0] = 0;
            }
            if (index[1] < width && (repeats[1] == iters || length_squared_v[1] >= 4)) { // index[1] done
                image[offset + index[1]] = repeats[1];
                index[1] = cur++;
                repeats[1] = 0;
                x0_v[1] = index[1] * col_step + left;
                x_v[1] = y_v[1] = length_squared_v[1] = x2_v[1] = y2_v[1] = 0;
            }
        }
    }
    pthread_exit(NULL);
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
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    row_step = ((upper - lower) / height);
    col_step = ((right - left) / width);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    // pthread
    pthread_t threads[ncpus];
    row = 0;
    pthread_mutex_init(&mutex, NULL);
    for (int t = 0; t < ncpus; t++)
        pthread_create(&threads[t], NULL, Cal, NULL);
    for (int t = 0; t < ncpus; t++)
        pthread_join(threads[t], NULL);
    pthread_mutex_destroy(&mutex);

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
