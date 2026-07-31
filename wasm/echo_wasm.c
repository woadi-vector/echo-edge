/* Flat, scalar-only surface for the browser. Keeps the JS side free of
 * struct layout knowledge so core/echo.h can change without touching app.js. */
#include "echo.h"

static echo_window_t g_win;
static echo_result_t g_res;

void echo_wasm_init(float window_ms)
{
    echo_window_init(&g_win, window_ms);
    g_res.valid = 0;
}

/* Push one RR interval (ms) and reclassify. Returns 1 if the result is valid. */
int echo_wasm_push(float rr_ms)
{
    echo_step(&g_win, rr_ms, &g_res);
    return g_res.valid;
}

int   echo_wasm_valid(void)          { return g_res.valid; }
int   echo_wasm_state(void)          { return (int)g_res.state; }
float echo_wasm_confidence(void)     { return g_res.confidence; }
float echo_wasm_feature(int i)
{
    if (i < 0 || i >= ECHO_N_FEATURES) return 0.0f;
    return g_res.features[i];
}
const char *echo_wasm_model_id(void) { return echo_model_id(); }
