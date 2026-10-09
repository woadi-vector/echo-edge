/* Parity harness: feed RR series in, get the C feature vectors out.
 *
 * One series per input line, comma or space separated milliseconds. Prints
 * one line per series: either eight comma-separated floats, or UNDERFILLED.
 * train/parity.py drives this and compares against features.extract, so the
 * two sides are never asserted to agree, they are measured.
 *
 * Driving it from Python rather than generating the series on both sides
 * avoids the usual trap, where the two generators drift and the parity check
 * passes while comparing different inputs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "echo.h"

#define MAX_LINE (1 << 20)
#define MAX_VALS 20000

int main(int argc, char **argv)
{
    const float window_ms = (argc > 1) ? (float)atof(argv[1]) : 60000.0f;
    char *line = malloc(MAX_LINE);
    static float vals[MAX_VALS];
    if (!line) return 1;

    while (fgets(line, MAX_LINE, stdin)) {
        int n = 0;
        for (char *p = strtok(line, ", \t\r\n"); p && n < MAX_VALS;
             p = strtok(NULL, ", \t\r\n")) {
            vals[n++] = (float)atof(p);
        }
        if (n == 0) continue;

        echo_window_t w;
        echo_window_init(&w, window_ms);
        for (int i = 0; i < n; i++) echo_window_push(&w, vals[i]);

        float f[ECHO_N_FEATURES];
        if (!echo_features(&w, f)) {
            printf("UNDERFILLED\n");
        } else {
            for (int i = 0; i < ECHO_N_FEATURES; i++)
                printf("%.9g%s", f[i], i + 1 < ECHO_N_FEATURES ? "," : "\n");
        }
        fflush(stdout);
    }
    free(line);
    return 0;
}
