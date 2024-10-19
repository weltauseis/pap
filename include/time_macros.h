#ifndef TIME_MACROS_IS_DEF
#define TIME_MACROS_IS_DEF

#include <sys/time.h>
#ifdef ENABLE_FUT
#define CONFIG_FUT
#include <fut.h>
#endif

#define TIME2USEC(t) ((uint64_t)(t).tv_sec * 1000000ULL + (t).tv_usec)
#define TIMESPEC2USEC(t) ((uint64_t)(t).tv_sec * 1000000ULL + (t).tv_nsec / 1000)

// Returns duration in µsecs
#define TIME_DIFF(t1, t2) (TIME2USEC (t2) - TIME2USEC (t1))

static inline uint64_t what_time_is_it (void)
{
#ifdef ENABLE_FUT
  return fut_getstamp () / 1000;
#else
#if 0
  struct timeval tv_now;

  gettimeofday (&tv_now, NULL);

  return TIME2USEC (tv_now);
#else
  struct timespec tp;
  clock_gettime(CLOCK_MONOTONIC, &tp);

	return TIMESPEC2USEC(tp);

#endif
#endif
}

#endif
