
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "error.h"
#include "debug.h"
#include "global.h"
#include "scheduler.h"
#include "ez_pthread.h"

static int nbWorkers = -1;

volatile static int nbTask = 0;
static pthread_mutex_t mutex      = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond        = PTHREAD_COND_INITIALIZER;

static unsigned nb_cores;

#define WORK_QUEUE 1024

struct task
{
  task_func_t fun;
  void *p;
};

static struct worker
{
  int id;
  pthread_t tid;
  pthread_cond_t cond;
  pthread_mutex_t mutex;
  int fin, todo;
  struct task tasks[WORK_QUEUE];
  unsigned d, f;
} * workers;

void scheduler_task_wait ()
{
  pthread_mutex_lock (&mutex);
  while (nbTask > 0)
    pthread_cond_wait (&cond, &mutex);
  pthread_mutex_unlock (&mutex);
}

static void one_more_task ()
{
  pthread_mutex_lock (&mutex);
  nbTask++;
  pthread_mutex_unlock (&mutex);
}

static void one_less_task ()
{
  pthread_mutex_lock (&mutex);
  nbTask--;
  if (nbTask == 0)
    pthread_cond_signal (&cond);
  pthread_mutex_unlock (&mutex);
}

static void add_task (struct task todo, int w)
{
  one_more_task ();
  pthread_mutex_lock (&workers[w].mutex);
  workers[w].tasks[workers[w].f] = todo;
  workers[w].f                   = (workers[w].f + 1) % WORK_QUEUE;
  workers[w].todo++;
  if (workers[w].todo > WORK_QUEUE)
    exit_with_error ("Task queue overflow (#tasks > %d). Try to increase tile size.", WORK_QUEUE);
  pthread_cond_signal (&workers[w].cond);
  pthread_mutex_unlock (&workers[w].mutex);
}

static void no_more_task (int w)
{
  pthread_mutex_lock (&workers[w].mutex);
  workers[w].fin = 1;
  pthread_cond_signal (&workers[w].cond);
  pthread_mutex_unlock (&workers[w].mutex);
}

void scheduler_create_task (task_func_t task, void *param, unsigned cpu)
{
  struct task todo;

  todo.p   = param;
  todo.fun = task;

  if (cpu == -1) {
    static int cyclic = 0;
    // We quickly go through the list of workers to find an idle one, or the
    // least busy one
    PRINT_DEBUG ('s', "Dynamic task scheduling is not yet implemented\n");
    cpu    = cyclic;
    cyclic = (cyclic + 1) % nbWorkers;
  }
  add_task (todo, cpu);
}

static void *worker_main (void *p)
{
  struct worker *me = (struct worker *)p;
  struct task todo  = {NULL, NULL};
  unsigned tasks    = 0;

  PRINT_DEBUG ('s', "Hey, I'm worker %d\n", me->id);

  while (1) {

    pthread_mutex_lock (&me->mutex);

    if (me->d == me->f && me->fin == 0)
      pthread_cond_wait (&me->cond, &me->mutex);

    if (me->d != me->f) {
      todo  = me->tasks[me->d];
      me->d = (me->d + 1) % WORK_QUEUE;
      me->todo--;
    } else if (me->fin == 1) {
      me->fin = -1;
    }

    pthread_mutex_unlock (&me->mutex);

    if (me->fin == -1) {
      PRINT_DEBUG ('s', "Worker %d has computed %d tasks\n", me->id, tasks);
      return NULL;
    }

    tasks++;
    todo.fun (todo.p, me->id);
    one_less_task ();
  }
}

unsigned scheduler_init (unsigned default_P)
{
  int i;

  nb_cores = easypap_number_of_cores ();

  if (default_P != -1)
    nbWorkers = default_P;
  else
    nbWorkers =  easypap_requested_number_of_threads ();

  ez_pthread_init (nbWorkers);

  PRINT_DEBUG ('s', "[Starting %d workers]\n", nbWorkers);

  workers = malloc (nbWorkers * sizeof (struct worker));

  for (i = 0; i < nbWorkers; i++) {
    workers[i].id   = i;
    workers[i].fin  = 0;
    workers[i].todo = 0;
    workers[i].d    = 0;
    workers[i].f    = 0;
    pthread_cond_init (&workers[i].cond, NULL);
    pthread_mutex_init (&workers[i].mutex, NULL);

    ez_pthread_create (&workers[i].tid, NULL, worker_main,
                       &workers[i]);
  }

  return nbWorkers;
}

void scheduler_finalize (void)
{
  int i;

  for (i = 0; i < nbWorkers; i++)
    no_more_task (i);

  for (i = 0; i < nbWorkers; i++)
    ez_pthread_join (workers[i].tid, NULL);

  ez_pthread_finalize ();
  
  free (workers);

  PRINT_DEBUG ('s', "[Workers stopped]\n");
}
