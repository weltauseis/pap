
#ifndef GLOBAL_IS_DEF
#define GLOBAL_IS_DEF

// Images are DIM * DIM arrays of pixels
// Tiles have a size of CPU_TILE_H * CPU_TILE_W
// An image contains CPU_NBTILES_Y * CPU_NBTILES_X

extern unsigned DIM;

extern unsigned TILE_W;
extern unsigned TILE_H;
extern unsigned NB_TILES_X;
extern unsigned NB_TILES_Y;

extern unsigned GPU_SIZE_X;
extern unsigned GPU_SIZE_Y;

extern unsigned do_display;
extern unsigned vsync;
extern unsigned soft_rendering;
extern unsigned refresh_rate;
extern unsigned do_first_touch;
extern int max_iter;
extern char *easypap_image_file;
extern char *draw_param;

extern unsigned gpu_used;
extern unsigned easypap_mpirun;

extern char *kernel_name, *variant_name, *tile_name;

#endif
