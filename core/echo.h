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

typedef struct {
    echo_state_t state;
    float        confidence;                 /* winning class vote share */
    float        votes[3];                   /* GREEN / AMBER / RED */
    float        features[ECHO_N_FEATURES];  /* raw, pre-scaling */
    int          valid;                      /* 0 until the window fills */
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

/* Push a beat and classify in one call. */
void echo_step(echo_window_t *w, float rr_ms, echo_result_t *out);

const char *echo_state_name(echo_state_t s);

/* Model provenance, emitted by the trainer. */
const char *echo_model_id(void);

#ifdef __cplusplus
}
#endif
#endif /* ECHO_H */
