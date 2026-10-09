/* Verification harness for the gap-spanning successive-difference fix.
 * Not part of the build; compiled directly. Prints features for a series so
 * the Python twin can be compared on identical input. */
#include <stdio.h>
#include <stdlib.h>
#include "echo.h"

static void dump(const char *label, const float *rr, int n, float window_ms)
{
    echo_window_t w;
    echo_window_init(&w, window_ms);
    int pushed = 0;
    for (int i = 0; i < n; i++) pushed += echo_window_push(&w, rr[i]);

    float f[ECHO_N_FEATURES];
    if (!echo_features(&w, f)) {
        printf("%-34s UNDERFILLED (held %u)\n", label, w.count);
        return;
    }
    printf("%-34s held=%3u acc=%3u rej=%3u | mean_rr %8.3f  sdnn %7.3f  "
           "rmssd %8.4f  pnn50 %6.4f  slope %8.4f  cov %5.3f\n",
           label, w.count, w.n_accepted, w.n_rejected,
           f[ECHO_F_MEAN_RR], f[ECHO_F_SDNN], f[ECHO_F_RMSSD],
           f[ECHO_F_PNN50], f[ECHO_F_RR_SLOPE], f[ECHO_F_COVERAGE]);
    (void)pushed;
}

int main(void)
{
    float a[41], b[41], c[60], d[60];
    int i;

    /* 1. Constant series, no artifact. RMSSD must be exactly 0. */
    for (i = 0; i < 41; i++) a[i] = 1000.0f;
    dump("1 constant, no artifact", a, 41, 60000.0f);

    /* 2. Step across a rejected beat. 20 x 1000, one 3000 (out of range),
     *    20 x 1200. The 1000 -> 1200 step spans the rejection, so it must be
     *    suppressed and RMSSD must stay 0. Unsuppressed it would be
     *    sqrt(200^2 / 39) = 32.025. */
    for (i = 0; i < 20; i++) b[i] = 1000.0f;
    b[20] = 3000.0f;
    for (i = 21; i < 41; i++) b[i] = 1200.0f;
    dump("2 step across rejected beat", b, 41, 60000.0f);

    /* 3. Same step with NO rejected beat between. Here the difference is real
     *    and must be counted: RMSSD = sqrt(200^2 / 39) = 32.025. */
    for (i = 0; i < 20; i++) c[i] = 1000.0f;
    for (i = 20; i < 40; i++) c[i] = 1200.0f;
    dump("3 same step, no rejection", c, 40, 60000.0f);

    /* 4. pnn50 denominator. 30 beats alternating 1000/1100 (100 ms diffs,
     *    all > 50), one out-of-range beat inserted. pnn50 must be 1.0: every
     *    difference actually taken exceeds 50 ms. An n-1 denominator would
     *    report less than 1.0. */
    int k = 0;
    for (i = 0; i < 15; i++) { d[k++] = 1000.0f; d[k++] = 1100.0f; }
    d[k++] = 3000.0f;
    for (i = 0; i < 15; i++) { d[k++] = 1000.0f; d[k++] = 1100.0f; }
    dump("4 alternating + rejection", d, k, 120000.0f);

    /* 5. The relative jump filter. 1000 -> 1300 is a 30% step, above
     *    ECHO_RR_JUMP = 0.25, so C rejects it. The Python twin applies only
     *    absolute bounds and keeps it. */
    float e[41];
    for (i = 0; i < 20; i++) e[i] = 1000.0f;
    for (i = 20; i < 41; i++) e[i] = 1300.0f;
    dump("5 30% step (relative filter)", e, 41, 60000.0f);

    return 0;
}
