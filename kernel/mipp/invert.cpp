#include <omp.h>
#include "cppdefs.h"
#include "global.h"
#include "img_data.h"
#include "mipp.h"

#define INV_MASK ((unsigned)0xFFFFFF00)

#ifdef ENABLE_VECTO
///////////////////////////// Vectorized version
// ./run -l images/shibuya.png -k invert -v tiled -wt mipp -i 100 -n
//
EXTERN int invert_do_tile_mipp(int x, int y, int width, int height) {
	mipp::Reg<int32_t> r_inv_mask, r_cur_img, r_res;
	r_inv_mask = INV_MASK;

	int nb_elem_per_reg = mipp::N<int32_t>();

	for (int i = y; i < y + height; i++)
		for (int j = x; j < x + width; j += nb_elem_per_reg) {
			r_cur_img.load((const int*)&cur_img(i, j));
			r_res = r_inv_mask ^ r_cur_img;
			r_res.store((int*)&cur_img(i, j));
		}

	return 0;
}
#endif
