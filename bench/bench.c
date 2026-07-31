/* Echo Edge benchmark harness.
 *
 * Reports the two numbers that matter for the Arm64 comparison:
 *   - feature extraction cost per beat (the signal-processing stage)
 *   - classification cost per inference (the forest traversal)
 * plus sustained throughput, and a parity check against the fixtures the
 * Python trainer emitted.
 *
 * Usage: ./bench [iterations]
 */
#define _POSIX_C_SOURCE 199309L
#include "../core/echo.h"
#include "testvectors.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NFIX ((int)(sizeof(ECHO_FIXTURES) / sizeof(ECHO_FIXTURES[0])))

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int parity(void)
{
    int fail = 0;
    echo_result_t r;
    for (int i = 0; i < NFIX; i++) {
        echo_classify(ECHO_FIXTURES[i].f, &r);
        if ((int)r.state != ECHO_FIXTURES[i].expect) {
            if (fail < 5)
                fprintf(stderr, "  parity miss @%d: C=%s expected=%d\n",
                        i, echo_state_name(r.state), ECHO_FIXTURES[i].expect);
            fail++;
        }
    }
    printf("parity: %d/%d fixtures match sklearn\n", NFIX - fail, NFIX);
    return fail;
}

int main(int argc, char **argv)
{
    const long iters = (argc > 1) ? atol(argv[1]) : 200000L;

    printf("Echo Edge bench — model %s\n", echo_model_id());
#if defined(__aarch64__)
    printf("arch: aarch64\n");
#elif defined(__x86_64__)
    printf("arch: x86_64\n");
#else
    printf("arch: unknown\n");
#endif

    const int fail = parity();

    /* Warm a realistic window so feature extraction has real work to do. */
    echo_window_t w;
    echo_window_init(&w, 60000.0f);
    unsigned seed = 1u;
    for (int i = 0; i < 90; i++) {
        seed = seed * 1103515245u + 12345u;
        echo_window_push(&w, 800.0f + (float)((seed >> 16) % 120) - 60.0f);
    }

    float feats[ECHO_N_FEATURES];
    echo_result_t r;
    volatile int sink = 0;

    double t0 = now_ns();
    for (long i = 0; i < iters; i++) {
        echo_features(&w, feats);
        sink += (int)feats[0];
    }
    double t_feat = (now_ns() - t0) / (double)iters;

    t0 = now_ns();
    for (long i = 0; i < iters; i++) {
        echo_classify(ECHO_FIXTURES[i % NFIX].f, &r);
        sink += (int)r.state;
    }
    double t_cls = (now_ns() - t0) / (double)iters;

    t0 = now_ns();
    for (long i = 0; i < iters; i++) {
        echo_step(&w, 780.0f + (float)(i % 90), &r);
        sink += (int)r.state;
    }
    double t_end = (now_ns() - t0) / (double)iters;

    printf("window beats:        %u\n", w.count);
    printf("feature extract:     %8.1f ns/beat\n", t_feat);
    printf("classify:            %8.1f ns/inference\n", t_cls);
    printf("end-to-end step:     %8.1f ns/beat\n", t_end);
    printf("throughput:          %8.0f inferences/sec/core\n", 1e9 / t_end);
    printf("concurrent operators at 1 Hz: %.0f/core\n", 1e9 / t_end);
    (void)sink;
    return fail ? 1 : 0;
}
