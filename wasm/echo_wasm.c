/* Flat, scalar-only surface for the browser. Keeps the JS side free of
 * struct layout knowledge so core/echo.h can change without touching app.js.
 *
 * Mirrors the operator API: enrollment first, classification after. */
#include "echo.h"
#include "echo_model.h"

static echo_operator_t g_op;
static echo_result_t   g_res;
static float           g_pending[ECHO_N_FEATURES];

void echo_wasm_init(float window_ms, float enroll_ms)
{
    echo_operator_init(&g_op, window_ms, enroll_ms);
    g_res.valid = 0;
    g_res.enrolling = 0;
}

/* Push one RR interval (ms) and reclassify. Returns 1 if a state is available. */
int echo_wasm_push(float rr_ms)
{
    echo_operator_step(&g_op, rr_ms, &g_res);
    return g_res.valid;
}

int   echo_wasm_valid(void)      { return g_res.valid; }
int   echo_wasm_enrolling(void)  { return g_res.enrolling; }
int   echo_wasm_baselined(void)  { return ECHO_BASELINED; }
int   echo_wasm_ready(void)      { return g_op.ready; }
int   echo_wasm_state(void)      { return (int)g_res.state; }
float echo_wasm_confidence(void) { return g_res.confidence; }

/* Fraction of the enrollment period completed, 0..1. */
float echo_wasm_enroll_progress(void)
{
    if (g_op.ready) return 1.0f;
    if (g_op.enroll_ms <= 0.0f) return 1.0f;
    float p = g_op.elapsed_ms / g_op.enroll_ms;
    return p > 1.0f ? 1.0f : p;
}

float echo_wasm_feature(int i)
{
    if (i < 0 || i >= ECHO_N_FEATURES) return 0.0f;
    return g_res.features[i];
}

/* --- baseline persistence: read out after enrolling, restore next session --- */

float echo_wasm_baseline(int i)
{
    const float *b = echo_operator_baseline(&g_op);
    if (!b || i < 0 || i >= ECHO_N_FEATURES) return 0.0f;
    return b[i];
}

void echo_wasm_stage_baseline(int i, float v)
{
    if (i >= 0 && i < ECHO_N_FEATURES) g_pending[i] = v;
}

void echo_wasm_commit_baseline(void)
{
    echo_operator_set_baseline(&g_op, g_pending);
}

const char *echo_wasm_model_id(void) { return echo_model_id(); }
