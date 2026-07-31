/* Echo Edge — portable readiness inference core.
 * C99, no dynamic allocation, no platform assumptions.
 * Targets: Arm64 (Neoverse), Arm Cortex-M, WebAssembly, x86_64.
 */
#ifndef ECHO_H
#define ECHO_H

#include <stdint.h>

#define ECHO_N_FEATURES 8
#define ECHO_MAX_BEATS  512

#ifdef __cplusplus
extern "C" {
#endif

/* Packed decision node — 8 bytes, so eight nodes share a 64-byte cache line.
 * The left child is always the next node in memory (trees are emitted in
 * depth-first preorder), so only the right child needs an index, stored as a
 * relative offset. This replaces five parallel arrays totalling 20 bytes per
 * node with one contiguous array at 8. */
typedef struct {
    float   threshold;  /* split point, or unused on a leaf */
    int16_t right_off;  /* offset to right child from this node */
    int8_t  feature;    /* feature index, or -1 to mark a leaf */
    uint8_t klass;      /* predicted class on a leaf */
} echo_node_t;

typedef enum {
    ECHO_GREEN = 0,
    ECHO_AMBER = 1,
    ECHO_RED   = 2
} echo_state_t;

/* Feature vector layout. The Python trainer must emit these in this order.
 * Any change here is a breaking change to the model artifact. */
enum {
    ECHO_F_MEAN_RR   = 0,  /* ms   */
    ECHO_F_MEAN_HR   = 1,  /* bpm  */
    ECHO_F_SDNN      = 2,  /* ms   */
    ECHO_F_RMSSD     = 3,  /* ms   */
    ECHO_F_PNN50     = 4,  /* 0..1 */
    ECHO_F_RR_SLOPE  = 5,  /* ms per beat, drift velocity proxy */
    ECHO_F_HR_CV     = 6,  /* sdnn / mean_rr */
    ECHO_F_COVERAGE  = 7   /* 0..1, fraction of window filled */
};

/* Rolling beat-to-beat window. Fixed capacity; oldest beats are evicted
 * once the window duration is exceeded. */
typedef struct {
    float    rr[ECHO_MAX_BEATS];   /* RR intervals, ms */
    uint16_t head;                 /* next write index */
    uint16_t count;                /* beats currently held */
    float    span_ms;              /* sum of held RR intervals */
    float    window_ms;            /* target window duration */
} echo_window_t;

/* Per-operator enrollment state.
 *
 * The model is trained on deviation from an operator's own resting baseline,
 * not on absolute physiology, so a baseline vector must exist before any
 * classification is meaningful. Establish it once per operator from a quiet
 * enrollment period, then store it and reuse it forever.
 *
 * A model built with ECHO_BASELINED == 0 ignores all of this. */
typedef struct {
    echo_window_t win;
    float    baseline[ECHO_N_FEATURES];
    float    accum[ECHO_N_FEATURES];
    uint32_t accum_n;
    float    enroll_ms;        /* target quiet duration */
    float    elapsed_ms;       /* quiet time observed so far */
    int      ready;            /* baseline established */
} echo_operator_t;

typedef struct {
    echo_state_t state;
    float        confidence;                 /* winning class vote share */
    float        votes[3];                   /* GREEN / AMBER / RED */
    float        features[ECHO_N_FEATURES];  /* raw, pre-scaling */
    float        relative[ECHO_N_FEATURES];  /* deviation from the operator's baseline */
    int          valid;                      /* 0 until the window fills */
    int          enrolling;                  /* 1 while the baseline is still forming */
} echo_result_t;

void echo_window_init(echo_window_t *w, float window_ms);

/* Push one beat-to-beat interval in milliseconds. Rejects implausible
 * intervals (<250ms / >2500ms) as ectopic or dropped-beat artifacts. */
int  echo_window_push(echo_window_t *w, float rr_ms);

/* Compute the feature vector. Returns 1 when the window holds enough
 * beats to be meaningful, 0 otherwise. */
int  echo_features(const echo_window_t *w, float *out);

/* Scale + classify a feature vector. */
void echo_classify(const float *features, echo_result_t *out);

/* Batched, tree-major classification for fleet scoring.
 *
 * The single-vector path walks all trees per operator, pulling the whole
 * forest through cache to produce one answer. This inverts the loops: each
 * tree is loaded once and every operator in the batch is pushed through it
 * before moving on. The forest is streamed once per batch instead of once
 * per operator, and the working set becomes the batch rather than the model.
 *
 * `features` is n contiguous vectors of ECHO_N_FEATURES floats.
 * `states` receives n results; `confidence` may be NULL. */
#define ECHO_BATCH_MAX 64
int echo_classify_batch(const float *features, int n,
                        echo_state_t *states, float *confidence);

/* Push a beat and classify in one call. */
void echo_step(echo_window_t *w, float rr_ms, echo_result_t *out);

/* --- per-operator API: use this one for anything live --- */

/* enroll_ms is the quiet period used to establish the baseline. Around
 * 180000 (three minutes) matches how the shipped model was trained. */
void echo_operator_init(echo_operator_t *op, float window_ms, float enroll_ms);

/* Restore a baseline saved from a previous session, skipping enrollment. */
void echo_operator_set_baseline(echo_operator_t *op, const float *baseline);

/* Read the established baseline back out for storage. NULL until ready. */
const float *echo_operator_baseline(const echo_operator_t *op);

/* Push one RR interval. While enrolling, out->enrolling is 1 and out->valid
 * is 0 — no state is reported, because none can be. */
void echo_operator_step(echo_operator_t *op, float rr_ms, echo_result_t *out);

/* Express raw features as deviation from a baseline, in population units. */
void echo_relativize(const float *raw, const float *baseline, float *out);

const char *echo_state_name(echo_state_t s);

/* Model provenance, emitted by the trainer. */
const char *echo_model_id(void);

#ifdef __cplusplus
}
#endif
#endif /* ECHO_H */
