#include "echo.h"
#include "echo_model.h"
#include <math.h>
#include <string.h>

#define ECHO_MIN_BEATS 20
#define ECHO_RR_MIN 250.0f
#define ECHO_RR_MAX 2500.0f

void echo_window_init(echo_window_t *w, float window_ms)
{
    memset(w, 0, sizeof(*w));
    w->window_ms = window_ms > 0.0f ? window_ms : 60000.0f;
}

static float win_at(const echo_window_t *w, uint16_t i)
{
    /* i = 0 is the oldest held beat. */
    uint16_t start = (uint16_t)((w->head + ECHO_MAX_BEATS - w->count) % ECHO_MAX_BEATS);
    return w->rr[(uint16_t)((start + i) % ECHO_MAX_BEATS)];
}

int echo_window_push(echo_window_t *w, float rr_ms)
{
    if (!(rr_ms >= ECHO_RR_MIN && rr_ms <= ECHO_RR_MAX)) return 0;

    if (w->count == ECHO_MAX_BEATS) {
        w->span_ms -= win_at(w, 0);
        w->count--;
    }
    w->rr[w->head] = rr_ms;
    w->head = (uint16_t)((w->head + 1) % ECHO_MAX_BEATS);
    w->count++;
    w->span_ms += rr_ms;

    while (w->count > ECHO_MIN_BEATS && w->span_ms - win_at(w, 0) > w->window_ms) {
        w->span_ms -= win_at(w, 0);
        w->count--;
    }
    return 1;
}

int echo_features(const echo_window_t *w, float *out)
{
    if (w->count < ECHO_MIN_BEATS) return 0;

    const uint16_t n = w->count;
    float sum = 0.0f, sumsq = 0.0f;
    float diff_sq = 0.0f;
    uint16_t nn50 = 0;
    float sum_i = 0.0f, sum_i2 = 0.0f, sum_ix = 0.0f;
    float prev = win_at(w, 0);

    for (uint16_t i = 0; i < n; i++) {
        const float x = win_at(w, i);
        const float fi = (float)i;
        sum   += x;
        sumsq += x * x;
        sum_i  += fi;
        sum_i2 += fi * fi;
        sum_ix += fi * x;
        if (i > 0) {
            const float d = x - prev;
            diff_sq += d * d;
            if (d > 50.0f || d < -50.0f) nn50++;
        }
        prev = x;
    }

    const float fn      = (float)n;
    const float mean_rr = sum / fn;
    float var = sumsq / fn - mean_rr * mean_rr;
    if (var < 0.0f) var = 0.0f;
    const float sdnn  = sqrtf(var);
    const float rmssd = (n > 1) ? sqrtf(diff_sq / (float)(n - 1)) : 0.0f;

    const float denom = fn * sum_i2 - sum_i * sum_i;
    const float slope = (denom != 0.0f) ? (fn * sum_ix - sum_i * sum) / denom : 0.0f;

    float coverage = w->span_ms / w->window_ms;
    if (coverage > 1.0f) coverage = 1.0f;

    out[ECHO_F_MEAN_RR]  = mean_rr;
    out[ECHO_F_MEAN_HR]  = 60000.0f / mean_rr;
    out[ECHO_F_SDNN]     = sdnn;
    out[ECHO_F_RMSSD]    = rmssd;
    out[ECHO_F_PNN50]    = (n > 1) ? (float)nn50 / (float)(n - 1) : 0.0f;
    out[ECHO_F_RR_SLOPE] = slope;
    out[ECHO_F_HR_CV]    = (mean_rr > 0.0f) ? sdnn / mean_rr : 0.0f;
    out[ECHO_F_COVERAGE] = coverage;
    return 1;
}

void echo_classify(const float *features, echo_result_t *out)
{
    float z[ECHO_N_FEATURES];
    for (int i = 0; i < ECHO_N_FEATURES; i++)
        z[i] = (features[i] - ECHO_SCALER_MEAN[i]) * ECHO_SCALER_INV_SCALE[i];

    int tally[3] = {0, 0, 0};
    for (int t = 0; t < ECHO_N_TREES; t++) {
        int32_t node = ECHO_TREE_ROOT[t];
        while (ECHO_FEATURE[node] >= 0) {
            node = (z[ECHO_FEATURE[node]] <= ECHO_THRESHOLD[node])
                 ? ECHO_LEFT[node] : ECHO_RIGHT[node];
        }
        tally[ECHO_LEAF_CLASS[node]]++;
    }

    int best = 0;
    for (int c = 1; c < 3; c++) if (tally[c] > tally[best]) best = c;

    for (int c = 0; c < 3; c++) out->votes[c] = (float)tally[c] / (float)ECHO_N_TREES;
    out->state      = (echo_state_t)best;
    out->confidence = out->votes[best];
    if (features != out->features)
        memcpy(out->features, features, sizeof(float) * ECHO_N_FEATURES);
    out->valid = 1;
}

void echo_step(echo_window_t *w, float rr_ms, echo_result_t *out)
{
    echo_window_push(w, rr_ms);
    if (!echo_features(w, out->features)) {
        out->valid = 0;
        out->state = ECHO_GREEN;
        out->confidence = 0.0f;
        return;
    }
    echo_classify(out->features, out);
}

const char *echo_state_name(echo_state_t s)
{
    switch (s) {
        case ECHO_GREEN: return "GREEN";
        case ECHO_AMBER: return "AMBER";
        case ECHO_RED:   return "RED";
        default:         return "UNKNOWN";
    }
}

const char *echo_model_id(void) { return ECHO_MODEL_ID; }
