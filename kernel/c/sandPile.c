#include "easypap.h"

#include <omp.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <unistd.h>

typedef unsigned int TYPE;

static TYPE* restrict TABLE = NULL;

// array of changes for each tile (0 if no change, 1 if change)
static bool* CHANGETABLE = NULL;

// asynchronous table
static inline TYPE* atable_cell(TYPE* restrict i, int y, int x) {
    return i + y * DIM + x;
}

#define atable(y, x) (*atable_cell(TABLE, (y), (x)))

// asynchronous change table
static inline bool* achangetable_cell(bool* restrict i, int y, int x) {
    int id = (x / TILE_W) + (y / TILE_H) * NB_TILES_X;
    return i + id;
}

#define achangetable(y, x) (*achangetable_cell(CHANGETABLE, (y), (x)))

// synchronous table
static inline TYPE* table_cell(TYPE* restrict i, int step, int y, int x) {
    return DIM * DIM * step + i + y * DIM + x;
}

#define table(step, y, x) (*table_cell(TABLE, (step), (y), (x)))

// synchronous change table
static inline bool* changetable_cell(bool* restrict i, int step, int y, int x) {
    int id = (x / TILE_W) + (y / TILE_H) * NB_TILES_X;
    return NB_TILES_X * NB_TILES_Y * step + i + id;
}

#define changetable(step, y, x) (*changetable_cell(CHANGETABLE, (step), (y), (x)))

static int in = 0;
static int out = 1;

static inline void swap_tables() {
    int tmp = in;
    in = out;
    out = tmp;
}

#define RGB(r, g, b) rgba(r, g, b, 0xFF)

static TYPE max_grains;

void asandPile_refresh_img() {
    unsigned long int max = 0;
    for (int i = 1; i < DIM - 1; i++)
        for (int j = 1; j < DIM - 1; j++) {
            int g = table(in, i, j);
            int r, v, b;
            r = v = b = 0;
            if (g == 1)
                v = 255;
            else if (g == 2)
                b = 255;
            else if (g == 3)
                r = 255;
            else if (g == 4)
                r = v = b = 255;
            else if (g > 4)
                r = b = 255 - (240 * ((double)g) / (double)max_grains);

            cur_img(i, j) = RGB(r, v, b);
            if (g > max)
                max = g;
        }
    max_grains = max;
}

/////////////////////////////  Initial Configurations

static inline void set_cell(int y, int x, unsigned v) {
    atable(y, x) = v;
    if (gpu_used)
        cur_img(y, x) = v;
}

void asandPile_draw_4partout(void);

void asandPile_draw(char* param) {
    // Call function ${kernel}_draw_${param}, or default function (second
    // parameter) if symbol not found
    hooks_draw_helper(param, asandPile_draw_4partout);
}

void ssandPile_draw(char* param) {
    hooks_draw_helper(param, asandPile_draw_4partout);
}

void asandPile_draw_4partout(void) {
    max_grains = 8;
    for (int i = 1; i < DIM - 1; i++)
        for (int j = 1; j < DIM - 1; j++)
            set_cell(i, j, 4);
}

void asandPile_draw_DIM(void) {
    max_grains = DIM;
    for (int i = DIM / 4; i < DIM - 1; i += DIM / 4)
        for (int j = DIM / 4; j < DIM - 1; j += DIM / 4)
            set_cell(i, j, i * j / 4);
}

void asandPile_draw_alea(void) {
    max_grains = 5000;
    for (int i = 0; i < DIM >> 3; i++) {
        set_cell(1 + random() % (DIM - 2), 1 + random() % (DIM - 2), 1000 + (random() % (4000)));
    }
}

void asandPile_draw_big(void) {
    const int i = DIM / 2;
    set_cell(i, i, 100000);
}

void asandPile_draw_little(void){
    const int i = DIM / 2;
    set_cell(i, i, 4000);
}

static void one_spiral(int x, int y, int step, int turns) {
    int i = x, j = y, t;

    for (t = 1; t <= turns; t++) {
        for (; i < x + t * step; i++)
            set_cell(i, j, 3);
        for (; j < y + t * step + 1; j++)
            set_cell(i, j, 3);
        for (; i > x - t * step - 1; i--)
            set_cell(i, j, 3);
        for (; j > y - t * step - 1; j--)
            set_cell(i, j, 3);
    }
    set_cell(i, j, 4);

    for (int i = -2; i < 3; i++)
        for (int j = -2; j < 3; j++)
            set_cell(i + x, j + y, 3);
}

static void many_spirals(int xdebut, int xfin, int ydebut, int yfin, int step, int turns) {
    int i, j;
    int size = turns * step + 2;

    for (i = xdebut + size; i < xfin - size; i += 2 * size)
        for (j = ydebut + size; j < yfin - size; j += 2 * size)
            one_spiral(i, j, step, turns);
}

static void spiral(unsigned twists) {
    many_spirals(1, DIM - 2, 1, DIM - 2, 2, twists);
}

void asandPile_draw_spirals(void) {
    spiral(DIM / 32);
}

// shared functions

#define ALIAS(fun)           \
    void ssandPile_##fun() { \
        asandPile_##fun();   \
    }

ALIAS(refresh_img);
ALIAS(draw_4partout);
ALIAS(draw_DIM);
ALIAS(draw_alea);
ALIAS(draw_big);
ALIAS(draw_spirals);

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Synchronous Kernel /////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void ssandPile_init() {
    TABLE = calloc(2 * DIM * DIM, sizeof(TYPE));
    CHANGETABLE = calloc(2 * NB_TILES_X * NB_TILES_Y, sizeof(bool));

    for (size_t i = 0; i < NB_TILES_X * NB_TILES_Y; i++) {
        CHANGETABLE[i] = 1;
    }

    for (size_t i = 0; i < NB_TILES_X * NB_TILES_Y; i++) {
        CHANGETABLE[i + NB_TILES_X * NB_TILES_Y] = 0;
    }
}

void ssandPile_finalize() {
    free(TABLE);
    free(CHANGETABLE);
}

int ssandPile_do_tile_default(int x, int y, int width, int height) {
    int diff = 0;

    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++) {
            table(out, i, j) = table(in, i, j) % 4;
            table(out, i, j) += table(in, i + 1, j) / 4;
            table(out, i, j) += table(in, i - 1, j) / 4;
            table(out, i, j) += table(in, i, j + 1) / 4;
            table(out, i, j) += table(in, i, j - 1) / 4;
            if (table(out, i, j) >= 4)
                diff = 1;
        }

    return diff;
}

// version opti MAXIMUM 🏎️💨
int ssandPile_do_tile_opt(int x, int y, int width, int height) {
    int diff = 0;

    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++) {
            TYPE tmp = table(in, i, j) % 4 + table(in, i + 1, j) / 4 + table(in, i - 1, j) / 4 + table(in, i, j + 1) / 4 + table(in, i, j - 1) / 4;
            table(out, i, j) = tmp;
            if (tmp >= 4)
                diff = 1;
        }

    return diff;
}

int ssandPile_do_tile_lazy_check(int x, int y, int width, int height) {
    int diff = 0;

    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++) {
            TYPE tmp = table(in, i, j) % 4 + table(in, i + 1, j) / 4 + table(in, i - 1, j) / 4 + table(in, i, j + 1) / 4 + table(in, i, j - 1) / 4;
            table(out, i, j) = tmp;
            if ((tmp >= 4) || (table(in, i, j) != table(out, i, j)))
                diff = 1;
        }

    return diff;
}

#ifdef ENABLE_VECTO
#include <immintrin.h>

#if __AVX2__ == 1

int ssandPile_do_tile_avx(int x, int y, int width, int height) {
    // border tiles have a width = TILE_W - 1
    // for example, with -s 32 and -tw 8, we have the following tile sizes for a line
    // X - 7 - 8 - 8 - 7 - X
    // (X is one pixel wide and is not computed)
    bool border_tile = (x == 1) || (x >= DIM - TILE_W);
    bool tile_too_small = (TILE_W < 8);

    // left and right border tiles have to be ignored, since they can't fit nicely inside an AVX register
    // TOD0 : if border tile is wide enough (e.g 31), subdivide it into chunks of width 8
    // for now we just ignore it
    if (border_tile || tile_too_small) {
        return ssandPile_do_tile_lazy_check(x, y, width, height);
    }

    // normal tiles are guaranteed to have lines divisible into chunks of 8 since -tw must be a power of 2
    int diff = 0;
    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j += AVX_VEC_SIZE_INT) {
            // get the current row and its neighbors
            __m256i current = _mm256_loadu_si256((__m256i*)&table(in, i, j));
            __m256i up = _mm256_loadu_si256((__m256i*)&table(in, i - 1, j));
            __m256i down = _mm256_loadu_si256((__m256i*)&table(in, i + 1, j));
            __m256i left = _mm256_loadu_si256((__m256i*)&table(in, i, j - 1));
            __m256i right = _mm256_loadu_si256((__m256i*)&table(in, i, j + 1));

            // modulo 4 current
            // exemple: 1111 & 0011 = 0011 | 15 & 3 = 3
            // exemple: 1100 & 0011 = 0000 | 12 & 3 = 0
            current = _mm256_and_si256(current, _mm256_set1_epi32(3));

            // divide neighbors by 4
            // exemple: 1100 >> 2 = 0011 | 12 >> 2 = 3
            // exemple: 1111 >> 2 = 0011 | 15 >> 2 = 3
            // exemple: 0101 >> 2 = 0001 | 5 >> 2 = 1
            up = _mm256_srli_epi32(up, 2);
            down = _mm256_srli_epi32(down, 2);
            left = _mm256_srli_epi32(left, 2);
            right = _mm256_srli_epi32(right, 2);

            // sum them all
            __m256i sum = _mm256_add_epi32(current, up);
            sum = _mm256_add_epi32(sum, down);
            sum = _mm256_add_epi32(sum, left);
            sum = _mm256_add_epi32(sum, right);

            // store the result
            _mm256_storeu_si256((__m256i*)&table(out, i, j), sum);

            // make the mask for the conditional: if any of the elements is greater than 3 (same as >= 4)
            // sets the corresponding 32bits of mask to 0xFFFFFFFF else 0
            __m256i mask = _mm256_cmpgt_epi32(sum, _mm256_set1_epi32(3));

            // if any bit of mask is true testz will return 0 so we negate.
            diff |= (!_mm256_testz_si256(mask, mask));

            // get the in and out lines
            __m256i in_vec = _mm256_loadu_si256((__m256i*)&table(in, i, j));
            __m256i out_vec = _mm256_loadu_si256((__m256i*)&table(out, i, j));

            // compare them : 0xFFFFFFFF if identical
            __m256i diff_mask = _mm256_cmpeq_epi32(in_vec, out_vec);

            // Create a vector with all bits set to 1
            __m256i ones = _mm256_set1_epi32(-1);
            // flip the diff mask
            __m256i inverted_diff_mask = _mm256_xor_si256(diff_mask, ones);

            // returns 1 if vector is full of zeros (if in = out) so we need to negate
            diff |= (!_mm256_testz_si256(inverted_diff_mask, inverted_diff_mask));
        }
    }

    return diff;
}

int ssandPile_do_tile_avx_facto(int x, int y, int width, int height) {

    // border tiles have a width = TILE_W - 1
    // for example, with -s 32 and -tw 8, we have the following tile sizes for a line
    // X - 7 - 8 - 8 - 7 - X
    // (X is one pixel wide and is not computed)
    bool border_tile = (x == 1)  || (x >= DIM - TILE_W);
    bool tile_too_small = (TILE_W < 8);

    // left and right border tiles have to be ignored, since they can't fit nicely inside an AVX register
    // TOD0 : if border tile is wide enough (e.g 31), subdivide it into chunks of width 8
    // for now we just ignore it
    if (border_tile || tile_too_small) {
        return ssandPile_do_tile_lazy_check(x, y, width, height);
    }

    int diff = 0;

    for (int j = x; j < x + width; j += AVX_VEC_SIZE_INT) {

        // init for first iteration
        __m256i current_in = _mm256_loadu_si256((__m256i*)&table(in, y, j));
        __m256i up_in = _mm256_loadu_si256((__m256i*)&table(in, y - 1, j));

        for (int i = y; i < y + height; i++) {    
            
            // get the current row and its neighbors
            __m256i down_in = _mm256_loadu_si256((__m256i*)&table(in, i + 1, j));
            __m256i left_in = _mm256_loadu_si256((__m256i*)&table(in, i, j - 1));
            __m256i right_in = _mm256_loadu_si256((__m256i*)&table(in, i, j + 1));

            // divide neighbors by 4
            // exemple: 1100 >> 2 = 0011 | 12 >> 2 = 3
            // exemple: 1111 >> 2 = 0011 | 15 >> 2 = 3
            // exemple: 0101 >> 2 = 0001 | 5 >> 2 = 1
            __m256i up_out = _mm256_srli_epi32(up_in, 2);
            __m256i down_out = _mm256_srli_epi32(down_in, 2);
            __m256i left_out = _mm256_srli_epi32(left_in, 2);
            __m256i right_out = _mm256_srli_epi32(right_in, 2);

            // modulo 4 current
            // exemple: 1111 & 0011 = 0011 | 15 & 3 = 3
            // exemple: 1100 & 0011 = 0000 | 12 & 3 = 0
            __m256i current_out = _mm256_and_si256(current_in, _mm256_set1_epi32(3));

            // sum them all
            __m256i sum = _mm256_add_epi32(current_out, up_out);
            sum = _mm256_add_epi32(sum, down_out);
            sum = _mm256_add_epi32(sum, left_out);
            sum = _mm256_add_epi32(sum, right_out);

            // store the result
            _mm256_storeu_si256((__m256i*)&table(out, i, j), sum);

            // make the mask for the conditional: if any of the elements is greater than 3 (same as >= 4) 
            // sets the corresponding 32bits of mask to 0xFFFFFFFF else 0
            __m256i mask = _mm256_cmpgt_epi32(sum, _mm256_set1_epi32(3));

            // if any bit of mask is true testz will return 0 so we negate.
            diff |= (!_mm256_testz_si256(mask, mask));

            // compare the in and out lines : 0xFFFFFFFF if identical
            __m256i diff_mask = _mm256_cmpeq_epi32(current_in, current_out);

            // Create a vector with all bits set to 1
            __m256i ones = _mm256_set1_epi32(-1);
            // flip the diff mask
            __m256i inverted_diff_mask = _mm256_xor_si256(diff_mask, ones);

            // returns 1 if vector is full of zeros (if in = out) so we need to negate
            diff |= (!_mm256_testz_si256(inverted_diff_mask, inverted_diff_mask));

            // update the registers
            up_in = current_in;
            current_in = down_in;
        }
    }

    return diff;
}

int ssandPile_do_tile_avx_full(int x, int y, int width, int height) {

    bool tile_too_small = (width < 8);
    // left and right border tiles have to be ignored, since they can't fit nicely inside an AVX register

    if (tile_too_small) {
        return ssandPile_do_tile_lazy_check(x, y, width, height);
    }

    int diff = 0;

    // border tiles have a width = TILE_W - 1
    // for example, with -s 32 and -tw 8, we have the following tile sizes for a line
    // X - 7 - 8 - 8 - 7 - X
    // (X is one pixel wide and is not computed)
    bool border_tile = (x == 1)  || (x >= DIM - TILE_W);

    int start = x;
    // if we are on the border tile, we update the part that can't be vectorized then proceed by updating the start position
    if (border_tile){
        start = x + (width % AVX_VEC_SIZE_INT);
        diff |= ssandPile_do_tile_lazy_check(x,y, width % AVX_VEC_SIZE_INT, height);
    }

    // normal tiles are guaranteed to have lines divisible into chunks of 8 since -tw must be a power of 2
    for (int i = y; i < y + height; i++) {
        for (int j = start; j < x + width; j += AVX_VEC_SIZE_INT) {    
            
            // get the current row and its neighbors
            __m256i current = _mm256_loadu_si256((__m256i*)&table(in, i, j));
            __m256i up = _mm256_loadu_si256((__m256i*)&table(in, i - 1, j));
            __m256i down = _mm256_loadu_si256((__m256i*)&table(in, i + 1, j));
            __m256i left = _mm256_loadu_si256((__m256i*)&table(in, i, j - 1));
            __m256i right = _mm256_loadu_si256((__m256i*)&table(in, i, j + 1));

            // divide neigbors by 4
            // exemple: 1100 >> 2 = 0011 | 12 >> 2 = 3
            // exemple: 1111 >> 2 = 0011 | 15 >> 2 = 3
            // exemple: 0101 >> 2 = 0001 | 5 >> 2 = 1
            up = _mm256_srli_epi32(up, 2);
            down = _mm256_srli_epi32(down, 2);
            left = _mm256_srli_epi32(left, 2);
            right = _mm256_srli_epi32(right, 2);

            // modulo 4 current
            // exemple: 1111 & 0011 = 0011 | 15 & 3 = 3
            // exemple: 1100 & 0011 = 0000 | 12 & 3 = 0
            current = _mm256_and_si256(current, _mm256_set1_epi32(3));

            // sum them all
            __m256i sum = _mm256_add_epi32(current, up);
            sum = _mm256_add_epi32(sum, down);
            sum = _mm256_add_epi32(sum, left);
            sum = _mm256_add_epi32(sum, right);

            // store the result
            _mm256_storeu_si256((__m256i*)&table(out, i, j), sum);

            // make the mask for the if: if any of the elements is greater than 3 (same as >= 4) 
            // sets the corresponding 32bits of mask to 0xFFFFFFFF else 0
            __m256i mask = _mm256_cmpgt_epi32(sum, _mm256_set1_epi32(3));

            // if any 32bit of mask is true testz will return 0 so we negate.
            diff |= (!_mm256_testz_si256(mask, mask));

            // get the in and out lines            
            __m256i in_vec = _mm256_loadu_si256((__m256i*)&table(in, i, j));
            __m256i out_vec = _mm256_loadu_si256((__m256i*)&table(out, i, j));

            // compare them : 0xFFFFFFFF if identical
            __m256i diff_mask = _mm256_cmpeq_epi32(in_vec, out_vec);

            // Create a vector with all bits set to 1
            __m256i ones = _mm256_set1_epi32(-1);
            // flip the diff mask
            __m256i inverted_diff_mask = _mm256_xor_si256(diff_mask, ones);

            // returns 1 if vector is full of zeros (if in = out) so we need to negate
            diff |= (!_mm256_testz_si256(inverted_diff_mask, inverted_diff_mask));
        }
    }

    return diff;
}

#endif
#endif

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned ssandPile_compute_seq(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = do_tile(1, 1, DIM - 2, DIM - 2);
        swap_tables();
        if (change == 0)
            return it;
    }
    return 0;
}

unsigned ssandPile_compute_tiled(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

        for (int y = 0; y < DIM; y += TILE_H)
            for (int x = 0; x < DIM; x += TILE_W)
                change |= do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));
        swap_tables();
        if (change == 0)
            return it;
    }

    return 0;
}

// EDIT : Namyst m'a répondu
/*
Bonjour

Ca veut juste dire qu’il s’agit de paralléliser une version qui parcourt l’image de manière non tuilée,
en utilisant une simple double-boucle sur les pixels :

for (i = 1; i < DIM-1; i++)
  for (j = 1; j < DIM-1; j++)
    table(out, i, j) = …

Dans ce cas, les pixels ne sont pas forcément distribués aux threads sous forme de tuiles.

*/
// Donc j'ai corrigé du coup
unsigned ssandPile_compute_omp(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

#pragma omp parallel for schedule(runtime) shared(change)
        for (int i = 1; i < DIM - 1; i++) {
            for (int j = 1; j < DIM - 1; j++) {
                TYPE tmp = table(in, i, j) % 4 + table(in, i + 1, j) / 4 + table(in, i - 1, j) / 4 + table(in, i, j + 1) / 4 + table(in, i, j - 1) / 4;
                table(out, i, j) = tmp;
                if (tmp >= 4)
                    change = 1;
            }
        }
        swap_tables();
        if (change == 0)
            return it;
    }
    return 0;
}

// pareil qu'avant mais avec des tuiles
unsigned ssandPile_compute_omp_tiled(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

#pragma omp parallel for collapse(2) schedule(runtime) reduction(| : change)
        for (int y = 0; y < DIM; y += TILE_H) {
            for (int x = 0; x < DIM; x += TILE_W) {
                change |= do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));
            }
        }

        swap_tables();
        if (change == 0)
            return it;
    }

    return 0;
}

// idem, mais tâches openMP
unsigned ssandPile_compute_omp_taskloop(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

#pragma omp parallel
#pragma omp single
        {
#pragma omp taskloop collapse(2) reduction(| : change)
            for (int y = 0; y < DIM; y += TILE_H) {
                for (int x = 0; x < DIM; x += TILE_W) {
                    change |= do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));
                }
            }
        }

        swap_tables();
        if (change == 0)
            return it;
    }

    return 0;
}

#ifdef ENABLE_OPENCL

// Only called when --dump or --thumbnails is used
void ssandPile_refresh_img_ocl() {
    cl_int err;

    err = clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0, sizeof(unsigned) * DIM * DIM, TABLE, 0, NULL, NULL);
    check(err, "Failed to read buffer from GPU");

    ssandPile_refresh_img();
}

#endif

////////////////////////////////////////////////////////////////////////////////
// LAZY
////////////////////////////////////////////////////////////////////////////////

unsigned ssandPile_compute_lazy(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

        for (int y = 0; y < DIM; y += TILE_H) {
            for (int x = 0; x < DIM; x += TILE_W) {
                // décider si on doit mettre à jour : si la tuile ou un de ses voisin est marquée dans changetable
                bool should_change = changetable(in, y, x);
                if (x + TILE_W < DIM)
                    should_change |= changetable(in, y, x + TILE_W);
                if (x - (int)TILE_W >= 0)
                    should_change |= changetable(in, y, x - TILE_W);
                if (y + TILE_H < DIM)
                    should_change |= changetable(in, y + TILE_H, x);
                if (y - (int)TILE_H >= 0)
                    should_change |= changetable(in, y - TILE_H, x);

                // si on doit mettre à jour
                if (should_change) {
                    changetable(out, y, x) =
                        do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));

                    change |= changetable(out, y, x);
                } else {
                    changetable(out, y, x) = false;
                }
            }
        }

        swap_tables();

        if (change == 0)
            return it;
    }

    return 0;
}

unsigned ssandPile_compute_omp_lazy(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

#pragma omp parallel for collapse(2) schedule(runtime) reduction(| : change)
        for (int y = 0; y < DIM; y += TILE_H) {
            for (int x = 0; x < DIM; x += TILE_W) {
                // décider si on doit mettre à jour : si la tuile ou un de ses voisin est marquée dans changetable
                bool should_change = changetable(in, y, x);
                if (x + TILE_W < DIM)
                    should_change |= changetable(in, y, x + TILE_W);
                if (x - (int)TILE_W >= 0)
                    should_change |= changetable(in, y, x - TILE_W);
                if (y + TILE_H < DIM)
                    should_change |= changetable(in, y + TILE_H, x);
                if (y - (int)TILE_H >= 0)
                    should_change |= changetable(in, y - TILE_H, x);

                // si on doit mettre à jour
                if (should_change) {
                    changetable(out, y, x) =
                        do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));

                    change |= changetable(out, y, x);
                } else {
                    changetable(out, y, x) = false;
                }
            }
        }

        swap_tables();

        if (change == 0)
            return it;
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
///////////////////////////// Asynchronous Kernel ////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

void asandPile_init() {
    in = out = 0;
    if (TABLE == NULL) {
        const unsigned size = DIM * DIM * sizeof(TYPE);

        PRINT_DEBUG('u', "Memory footprint = 2 x %d bytes\n", size);

        TABLE = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }

    CHANGETABLE = malloc(sizeof(bool) * NB_TILES_X * NB_TILES_Y);
    for (int i = 0; i < NB_TILES_X * NB_TILES_Y; i++) {
        CHANGETABLE[i] = true;
    }
}

void asandPile_finalize() {
    const unsigned size = DIM * DIM * sizeof(TYPE);

    munmap(TABLE, size);

    free(CHANGETABLE);
}

///////////////////////////// Version séquentielle simple (seq)
// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0

int asandPile_do_tile_default(int x, int y, int width, int height) {
    int change = 0;

    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++)
            if (atable(i, j) >= 4) {
                atable(i, j - 1) += atable(i, j) / 4;
                atable(i, j + 1) += atable(i, j) / 4;
                atable(i - 1, j) += atable(i, j) / 4;
                atable(i + 1, j) += atable(i, j) / 4;
                atable(i, j) %= 4;
                change = 1;
            }
    return change;
}

int asandPile_do_tile_avx (int x, int y, int width, int height){
    bool border_tile = (x == 1)  || (x >= DIM - TILE_W);
    bool tile_too_small = (TILE_W < 8);

    if(border_tile || tile_too_small)
        return asandPile_do_tile_default(x, y, width, height);

    int change = 0;

    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j += AVX_VEC_SIZE_INT){
        
        // check if any of the 8 pixels in the row is greater than 3
        __m256i current = _mm256_load_epi32(&atable(i, j));
        __m256i mask = _mm256_cmpgt_epi32(current, _mm256_set1_epi32(3));

        // if none of the 8 pixels are greater than 3, we can skip
        if (_mm256_testz_si256(mask, mask))
            continue;

        // algo du pdf 
        __m256i up = _mm256_load_epi32(&atable(i - 1, j));
        __m256i down = _mm256_load_epi32(&atable(i + 1, j));

        __m256i D = _mm256_srli_epi32(current, 2);
        __m256i D_right = _mm256_alignr_epi32(_mm256_set1_epi32(0), D, 1);
        __m256i D_left = _mm256_alignr_epi32(D, _mm256_set1_epi32(0), 7); 

        __m256i current_mod4 = _mm256_and_si256(current, _mm256_set1_epi32(3));

        current = _mm256_add_epi32(current_mod4, _mm256_add_epi32(D_left, D_right));
        up = _mm256_add_epi32(up, D);
        down = _mm256_add_epi32(down, D);

        _mm256_store_epi32(&atable(i - 1, j), up);
        _mm256_store_epi32(&atable(i, j), current);
        _mm256_store_epi32(&atable(i + 1, j), down);

        atable(i, j - 1) += _mm256_extract_epi32(D, 0);
        atable(i, j + 8) += _mm256_extract_epi32(D, 7);

        change = 1;
    }

    return change;
}

int asandPile_do_tile_avx_iter (int x, int y, int width, int height){
    bool border_tile = (x == 1)  || (x >= DIM - TILE_W);
    bool tile_too_small = (TILE_W < 8);

    if (border_tile || tile_too_small)
        return asandPile_do_tile_default(x, y, width, height);

    int change = 0;

    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j += AVX_VEC_SIZE_INT) {
            // check if any of the 8 pixels in the row is greater than 3
            __m256i current = _mm256_load_epi32(&atable(i, j));
            __m256i mask = _mm256_cmpgt_epi32(current, _mm256_set1_epi32(3));

            // if none of the 8 pixels are greater than 3, we can skip
            if (!_mm256_testz_si256(mask, mask)) {
                // algo du pdf
                __m256i up = _mm256_load_epi32(&atable(i - 1, j));
                __m256i down = _mm256_load_epi32(&atable(i + 1, j));

                __m256i D;
                for (int iter = 0; iter < 4; iter++) {
                    D = _mm256_srli_epi32(current, 2);
                    __m256i D_right = _mm256_alignr_epi32(_mm256_set1_epi32(0), D, 1);
                    __m256i D_left = _mm256_alignr_epi32(D, _mm256_set1_epi32(0), 7);

                    __m256i current_mod4 = _mm256_and_si256(current, _mm256_set1_epi32(3));

                    current = _mm256_add_epi32(current_mod4, _mm256_add_epi32(D_left, D_right));
                    up = _mm256_add_epi32(up, D);
                    down = _mm256_add_epi32(down, D);

                    atable(i, j - 1) += _mm256_extract_epi32(D, 0);
                    atable(i, j + 8) += _mm256_extract_epi32(D, 7);
                }

                _mm256_store_epi32(&atable(i - 1, j), up);
                _mm256_store_epi32(&atable(i, j), current);
                _mm256_store_epi32(&atable(i + 1, j), down);

                change = 1;
            }
        }
    }


    return change;
}

int asandPile_do_tile_avx_facto (int x, int y, int width, int height){
    bool border_tile = (x == 1)  || (x >= DIM - TILE_W);
    bool tile_too_small = (TILE_W < 8);

    if(border_tile || tile_too_small)
        return asandPile_do_tile_default(x, y, width, height);

    int change = 0;

    for (int j = x; j < x + width; j += AVX_VEC_SIZE_INT){
        __m256i current = _mm256_load_epi32(&atable(y, j));
        __m256i up = _mm256_load_epi32(&atable(y - 1, j));
        for (int i = y; i < y + height; i++){
            __m256i down = _mm256_load_epi32(&atable(i + 1, j));

            // check if any of the 8 pixels in the row is greater than 3
            __m256i mask = _mm256_cmpgt_epi32(current, _mm256_set1_epi32(3));
            // if none of the 8 pixels are greater than 3, we can skip
            if (!_mm256_testz_si256(mask, mask)){
                // algo du pdf 

                __m256i D = _mm256_srli_epi32(current, 2);
                __m256i D_right = _mm256_alignr_epi32(_mm256_set1_epi32(0), D, 1);
                __m256i D_left = _mm256_alignr_epi32(D, _mm256_set1_epi32(0), 7); 

                __m256i current_mod4 = _mm256_and_si256(current, _mm256_set1_epi32(3));

                current = _mm256_add_epi32(current_mod4, _mm256_add_epi32(D_left, D_right));
                up = _mm256_add_epi32(up, D);
                down = _mm256_add_epi32(down, D);

                _mm256_store_epi32(&atable(i - 1, j), up);
                _mm256_store_epi32(&atable(i, j), current);
                _mm256_store_epi32(&atable(i + 1, j), down);

                atable(i, j - 1) += _mm256_extract_epi32(D, 0);
                atable(i, j + 8) += _mm256_extract_epi32(D, 7);

                change = 1;
            }

            up = current;
            current = down;
        }

    }

    return change;
}

unsigned asandPile_compute_seq(unsigned nb_iter) {
    int change = 0;
    for (unsigned it = 1; it <= nb_iter; it++) {
        // On traite toute l'image en un coup (oui, c'est une grosse tuile)
        change = do_tile(1, 1, DIM - 2, DIM - 2);

        if (change == 0)
            return it;
    }
    return 0;
}

unsigned asandPile_compute_tiled(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

        for (int y = 0; y < DIM; y += TILE_H)
            for (int x = 0; x < DIM; x += TILE_W)
                change |= do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));
        if (change == 0)
            return it;
    }

    return 0;
}

unsigned asandPile_compute_omp(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

/*
    we split the grid in 4 different "colors" so that no two neighboring tiles have the same color
    for example :

    a b a b
    c d c d
    a b a b
    c d c d

    we can then do each color in parallel
*/

// a tiles
#pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
        for (int y = 0; y < DIM; y += 2 * TILE_H)
            for (int x = 0; x < DIM; x += 2 * TILE_W)
                change |= do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));
// b tiles
#pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
        for (int y = 0; y < DIM; y += 2 * TILE_H)
            for (int x = TILE_W; x < DIM; x += 2 * TILE_W)
                change |= do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));
// c tiles
#pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
        for (int y = TILE_H; y < DIM; y += 2 * TILE_H)
            for (int x = 0; x < DIM; x += 2 * TILE_W)
                change |= do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));
// d tiles
#pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
        for (int y = TILE_H; y < DIM; y += 2 * TILE_H)
            for (int x = TILE_W; x < DIM; x += 2 * TILE_W)
                change |= do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));

        if (change == 0) {
            return it;
        }
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// LAZY
////////////////////////////////////////////////////////////////////////////////

static inline bool asand_do_tile_lazy_wrapper(int x, int y) {
    bool should_be_computed = achangetable(y, x);

    if (x + TILE_W < DIM)
        should_be_computed |= achangetable(y, x + TILE_W);
    if (x - (int)TILE_W >= 0)
        should_be_computed |= achangetable(y, x - TILE_W);
    if (y + TILE_H < DIM)
        should_be_computed |= achangetable(y + TILE_H, x);
    if (y - (int)TILE_H >= 0)
        should_be_computed |= achangetable(y - TILE_H, x);

    if (!should_be_computed)
        return false;

    achangetable(y, x) = do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));

    return achangetable(y, x);
}

unsigned asandPile_compute_lazy(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

        for (int y = 0; y < DIM; y += TILE_H) {
            for (int x = 0; x < DIM; x += TILE_W) {
                change |= asand_do_tile_lazy_wrapper(x, y);
            }
        }

        if (change == 0)
            return it;
    }
    return 0;
}

unsigned asandPile_compute_omp_lazy(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it++) {
        int change = 0;

// a tiles
#pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
        for (int y = 0; y < DIM; y += 2 * TILE_H)
            for (int x = 0; x < DIM; x += 2 * TILE_W)
                change |= asand_do_tile_lazy_wrapper(x, y);
// b tiles
#pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
        for (int y = 0; y < DIM; y += 2 * TILE_H)
            for (int x = TILE_W; x < DIM; x += 2 * TILE_W)
                change |= asand_do_tile_lazy_wrapper(x, y);
// c tiles
#pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
        for (int y = TILE_H; y < DIM; y += 2 * TILE_H)
            for (int x = 0; x < DIM; x += 2 * TILE_W)
                change |= asand_do_tile_lazy_wrapper(x, y);
// d tiles
#pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
        for (int y = TILE_H; y < DIM; y += 2 * TILE_H)
            for (int x = TILE_W; x < DIM; x += 2 * TILE_W)
                change |= asand_do_tile_lazy_wrapper(x, y);

        if (change == 0)
            return it;
    }
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// MPI
////////////////////////////////////////////////////////////////////////////////

#ifdef ENABLE_MPI
#include <mpi.h>

static int rank, size, OVERFLOW_WIDTH;

static int rankTop(int rank) {
    return (DIM / size) * rank;
}

static int rankSize(int rank) {
    return (DIM / size);
}

void asandPile_init_mpi() {
    easypap_check_mpi();  // check if MPI was correctly configured

    /* récupérer son propre numéro */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* récupérer le nombre de processus lancés */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* récupérer OVERFLOW_WIDTH à partir de l'environnement */
    char* env_var = getenv("MPI_OVERFLOW_WIDTH");
    int env_overflow_width = env_var != NULL ? atoi(env_var) : - 1;
    OVERFLOW_WIDTH = env_overflow_width > 0 ? env_overflow_width : 4;

    /* vérifier différentes contraintes */
    if(OVERFLOW_WIDTH >= (rankSize(0))){
        printf("MPI_OVERFLOW_WIDTH > DIM / MPI_SIZE !!\n");
        exit(1);
    }

    if(TILE_H > (rankSize(0))){
        printf("TILE_H > DIM / MPI_SIZE !!\n");
        exit(1);
    }

    asandPile_init();
}

void exchangeBorderZone() {
    MPI_Status status;
    TYPE buffer[DIM * OVERFLOW_WIDTH];

    // PART DU PRINCIPE QUE LE NOMBRE DE TRAVAILLEURS TOTAUX EST PAIR

    if (rank % 2 == 0) {
        // send bottom to next
        MPI_Send(&atable(rankTop(rank) + rankSize(rank), 0), DIM * OVERFLOW_WIDTH, MPI_UNSIGNED, rank + 1, 0, MPI_COMM_WORLD);
        memset(&atable(rankTop(rank) + rankSize(rank), 0),  0, DIM * OVERFLOW_WIDTH * sizeof(TYPE)); // clear
        if(rank != 0){
            // receive bottom of previous
            MPI_Recv(buffer, DIM * OVERFLOW_WIDTH, MPI_UNSIGNED, rank - 1, 0, MPI_COMM_WORLD, &status);
            for(int i = 0; i < DIM * OVERFLOW_WIDTH; i++){
                (&atable(rankTop(rank), 0))[i] += buffer[i];
            }
        }

        // receive top of next
        MPI_Recv(buffer, DIM * OVERFLOW_WIDTH, MPI_UNSIGNED, rank + 1, 0, MPI_COMM_WORLD, &status);
        for(int i = 0; i < DIM * OVERFLOW_WIDTH; i++){
            (&atable(rankTop(rank) + rankSize(rank) - OVERFLOW_WIDTH, 0))[i] += buffer[i];
        }
        if(rank != 0){
            // send top to previous
            MPI_Send(&atable(rankTop(rank) - OVERFLOW_WIDTH, 0), DIM * OVERFLOW_WIDTH, MPI_UNSIGNED, rank - 1, 0, MPI_COMM_WORLD);
            memset(&atable(rankTop(rank) - OVERFLOW_WIDTH, 0), 0, DIM * OVERFLOW_WIDTH * sizeof(TYPE)); // clear
        }

    } else if (rank % 2 == 1) {
        // receive bottom of previous
        MPI_Recv(buffer, DIM * OVERFLOW_WIDTH, MPI_UNSIGNED, rank - 1, 0, MPI_COMM_WORLD, &status);
        for(int i = 0; i < DIM * OVERFLOW_WIDTH; i++){
            (&atable(rankTop(rank), 0))[i] += buffer[i];
        }
        if(rank != (size - 1)){
            // send bottom to next
            MPI_Send(&atable(rankTop(rank) + rankSize(rank), 0), DIM * OVERFLOW_WIDTH, MPI_UNSIGNED, rank + 1, 0, MPI_COMM_WORLD);
            memset(&atable(rankTop(rank) + rankSize(rank), 0), 0, DIM * OVERFLOW_WIDTH * sizeof(TYPE)); // clear
        } 

        // send top to previous
        MPI_Send(&atable(rankTop(rank) - OVERFLOW_WIDTH, 0), DIM * OVERFLOW_WIDTH, MPI_UNSIGNED, rank - 1, 0, MPI_COMM_WORLD);
        memset(&atable(rankTop(rank) - OVERFLOW_WIDTH, 0), 0, DIM * OVERFLOW_WIDTH * sizeof(TYPE)); // clear
        if(rank != (size - 1)){
            // receive top of next
            MPI_Recv(buffer, DIM * OVERFLOW_WIDTH, MPI_UNSIGNED, rank + 1, 0, MPI_COMM_WORLD, &status);
            for(int i = 0; i < DIM * OVERFLOW_WIDTH; i++){
                (&atable(rankTop(rank) + rankSize(rank) - OVERFLOW_WIDTH, 0))[i] += buffer[i];
            }
        }
    }
}

void checkTermination(int* change) {

    int new_change = *change;

    for(int i = 0; i < size; i++){
        if(i == rank){
            // on envoie
            MPI_Bcast(change, 1, MPI_INT, i, MPI_COMM_WORLD);
        } else {
            // on reçoit
            int local_change;
            MPI_Bcast(&local_change, 1, MPI_INT, i, MPI_COMM_WORLD);
            new_change |= local_change;
        }
    }

    *change = new_change;
}

void fetchResults() {
    for(int i = 0; i < size; i++){
        MPI_Bcast(&atable(rankTop(i), 0), DIM * rankSize(i), MPI_UNSIGNED, i, MPI_COMM_WORLD);
    }
}

void clearOthers(){
    for(int i = 0; i < size; i++){
        if(i == rank)
            continue;

        memset(&atable(rankTop(i), 0), 0, DIM * rankSize(i) * sizeof(TYPE));
    }
}

static int total_iter = 0;
unsigned asandPile_compute_mpi(unsigned nb_iter) {

    for (unsigned it = 1; it <= nb_iter; it++) {

        int change = 0;

        // the overflow zones have to be empty before computation starts, but it's easier and faster to just clear everything
        if(total_iter == 0)
            clearOthers();

        if(rank != 0){
            // top overlow zone
            change |= do_tile(1, rankTop(rank) - (total_iter % OVERFLOW_WIDTH), DIM - 2, (total_iter % OVERFLOW_WIDTH));
        }

        if(rank != size - 1){
            // bottom overflow zone
            change |= do_tile(1, rankTop(rank) + rankSize(rank), DIM - 2, (total_iter % OVERFLOW_WIDTH));
        }

        // a tiles
        #pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
        for (int y = rankTop(rank); y < rankTop(rank) + rankSize(rank); y += 2 * TILE_H)
            for (int x = 0; x < DIM; x += 2 * TILE_W)
                change |= do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));
        // b tiles
        #pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
        for (int y = rankTop(rank); y < rankTop(rank) + rankSize(rank); y += 2 * TILE_H)
            for (int x = TILE_W; x < DIM; x += 2 * TILE_W)
                change |= do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));
        // c tiles
        #pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
        for (int y = rankTop(rank) + TILE_H; y < rankTop(rank) + rankSize(rank); y += 2 * TILE_H)
            for (int x = 0; x < DIM; x += 2 * TILE_W)
                change |= do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));
        // d tiles
        #pragma omp parallel for schedule(runtime) collapse(2) reduction(| : change)
        for (int y = rankTop(rank) + TILE_H; y < rankTop(rank) + rankSize(rank); y += 2 * TILE_H)
            for (int x = TILE_W; x < DIM; x += 2 * TILE_W)
                change |= do_tile(x + (x == 0), y + (y == 0), TILE_W - ((x + TILE_W == DIM) + (x == 0)), TILE_H - ((y + TILE_H == DIM) + (y == 0)));

        if((OVERFLOW_WIDTH == 1) || ((total_iter % (OVERFLOW_WIDTH - 1)) == 0)) {
            exchangeBorderZone();
        } 

        checkTermination(&change);

        if (change == 0) {
            fetchResults();
            return it;
        }

        total_iter++;
    }

    return 0;
}

#endif