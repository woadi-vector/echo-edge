#include "echo.h"
#include "echo_model.h"
#include <math.h>
#include <string.h>
#include <stddef.h>

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
        const echo_node_t *n = &ECHO_NODES[ECHO_TREE_ROOT[t]];
        while (n->feature >= 0)
            n = (z[n->feature] <= n->threshold) ? n + 1 : n + n->right_off;
        tally[n->klass]++;
    }

    int best = 0;
    for (int c = 1; c < 3; c++) if (tally[c] > tally[best]) best = c;

    for (int c = 0; c < 3; c++) out->votes[c] = (float)tally[c] / (float)ECHO_N_TREES;
    out->state      = (echo_state_t)best;
    out->confidence = out->votes[best];
    if (features != out->features && features != out->relative)
        memcpy(out->features, features, sizeof(float) * ECHO_N_FEATURES);
    out->valid = 1;
}

int echo_classify_batch(const float *features, int n,
                        echo_state_t *states, float *confidence)
{
    if (n <= 0) return 0;
    if (n > ECHO_BATCH_MAX) n = ECHO_BATCH_MAX;

    float z[ECHO_BATCH_MAX * ECHO_N_FEATURES];
    uint16_t tally[ECHO_BATCH_MAX * 3];

    for (int s = 0; s < n; s++) {
        const float *src = features + (size_t)s * ECHO_N_FEATURES;
        float *dst = z + (size_t)s * ECHO_N_FEATURES;
        for (int i = 0; i < ECHO_N_FEATURES; i++)
            dst[i] = (src[i] - ECHO_SCALER_MEAN[i]) * ECHO_SCALER_INV_SCALE[i];
    }
    memset(tally, 0, sizeof(uint16_t) * (size_t)n * 3);

    /* Tree-major: the tree stays resident while the batch streams past it. */
    for (int t = 0; t < ECHO_N_TREES; t++) {
        const echo_node_t *root = &ECHO_NODES[ECHO_TREE_ROOT[t]];
        for (int s = 0; s < n; s++) {
            const float *zi = z + (size_t)s * ECHO_N_FEATURES;
            const echo_node_t *nd = root;
            while (nd->feature >= 0)
                nd = (zi[nd->feature] <= nd->threshold) ? nd + 1 : nd + nd->right_off;
            tally[s * 3 + nd->klass]++;
        }
    }

    for (int s = 0; s < n; s++) {
        const uint16_t *v = &tally[s * 3];
        int best = 0;
        if (v[1] > v[best]) best = 1;
        if (v[2] > v[best]) best = 2;
        states[s] = (echo_state_t)best;
        if (confidence) confidence[s] = (float)v[best] / (float)ECHO_N_TREES;
    }
    return n;
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

void echo_relativize(const float *raw, const float *baseline, float *out)
{
#if ECHO_BASELINED
    for (int i = 0; i < ECHO_N_FEATURES; i++)
        out[i] = (raw[i] - baseline[i]) * ECHO_POP_INV_SCALE[i];
#else
    (void)baseline;
    memcpy(out, raw, sizeof(float) * ECHO_N_FEATURES);
#endif
}

void echo_operator_init(echo_operator_t *op, float window_ms, float enroll_ms)
{
    memset(op, 0, sizeof(*op));
    echo_window_init(&op->win, window_ms);
    op->enroll_ms = (enroll_ms > 0.0f) ? enroll_ms : 180000.0f;
#if !ECHO_BASELINED
    op->ready = 1;              /* model needs no baseline */
#endif
}

void echo_operator_set_baseline(echo_operator_t *op, const float *baseline)
{
    memcpy(op->baseline, baseline, sizeof(float) * ECHO_N_FEATURES);
    op->ready = 1;
}

const float *echo_operator_baseline(const echo_operator_t *op)
{
    return op->ready ? op->baseline : (const float *)0;
}

void echo_operator_step(echo_operator_t *op, float rr_ms, echo_result_t *out)
{
    out->enrolling = 0;
    if (!echo_window_push(&op->win, rr_ms)) { out->valid = 0; return; }
    if (!echo_features(&op->win, out->features)) { out->valid = 0; return; }

    if (!op->ready) {
        /* Accumulate resting feature vectors until the quiet period is met. */
        for (int i = 0; i < ECHO_N_FEATURES; i++) op->accum[i] += out->features[i];
        op->accum_n++;
        op->elapsed_ms += rr_ms;

        if (op->elapsed_ms >= op->enroll_ms && op->accum_n > 0) {
            for (int i = 0; i < ECHO_N_FEATURES; i++)
                op->baseline[i] = op->accum[i] / (float)op->accum_n;
            op->ready = 1;
        } else {
            out->valid = 0;
            out->enrolling = 1;
            return;
        }
    }

    echo_relativize(out->features, op->baseline, out->relative);
    echo_classify(out->relative, out);
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
