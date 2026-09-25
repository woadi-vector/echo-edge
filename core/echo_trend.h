/* Echo Edge — readiness trend / graded-warning engine.
 *
 * The classifier answers "where is the operator now." This answers "which way,
 * and how fast." Rothman-style: combine rate-of-change over two windows with
 * absolute floors, so a downward trajectory is flagged before the discrete
 * GREEN/AMBER/RED state flips.
 *
 * C99, no dynamic allocation — a fixed ring of recent readiness samples, sized
 * to cover the long window at ~1 sample/second. Portable to the same targets as
 * the core (Arm64, Cortex-M, WebAssembly, x86_64).
 */
#ifndef ECHO_TREND_H
#define ECHO_TREND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ECHO_TREND_NONE     = 0,
    ECHO_TREND_WATCH    = 1,  /* gradual decline over the long window */
    ECHO_TREND_ELEVATED = 2,  /* fast decline, or readiness below the mid floor */
    ECHO_TREND_HIGH     = 3   /* steep decline, or readiness below the low floor */
} echo_trend_level_t;

/* ~1 sample/s; 512 covers >8 min, comfortably past the long window. */
#define ECHO_TREND_CAP 512

/* Windows (ms) over which drops are measured. Tunable at compile time. */
#ifndef ECHO_TREND_SHORT_MS
#define ECHO_TREND_SHORT_MS 120000.0f   /* 2 min */
#endif
#ifndef ECHO_TREND_LONG_MS
#define ECHO_TREND_LONG_MS  300000.0f   /* 5 min */
#endif

/* Thresholds in readiness points (0-100). A drop is a fall from the past value
 * to now; a floor is an absolute level. Both directions of the Rothman recipe. */
#ifndef ECHO_TREND_WATCH_DROP
#define ECHO_TREND_WATCH_DROP 12.0f     /* long-window fall -> WATCH */
#endif
#ifndef ECHO_TREND_ELEV_DROP
#define ECHO_TREND_ELEV_DROP  18.0f     /* short-window fall -> ELEVATED */
#endif
#ifndef ECHO_TREND_HIGH_DROP
#define ECHO_TREND_HIGH_DROP  28.0f     /* short-window fall -> HIGH */
#endif
#ifndef ECHO_TREND_ELEV_FLOOR
#define ECHO_TREND_ELEV_FLOOR 40.0f     /* absolute readiness -> ELEVATED */
#endif
#ifndef ECHO_TREND_HIGH_FLOOR
#define ECHO_TREND_HIGH_FLOOR 25.0f     /* absolute readiness -> HIGH */
#endif

typedef struct {
    float    readiness[ECHO_TREND_CAP];
    float    t_ms[ECHO_TREND_CAP];
    uint16_t head;    /* next write index */
    uint16_t count;   /* samples held */
    float    now_ms;  /* running clock, advanced by the caller's dt */
    echo_trend_level_t level;
} echo_trend_t;

void echo_trend_init(echo_trend_t *tr);

/* Advance the clock by dt_ms (e.g. the RR interval since the last sample),
 * record `readiness`, and return the graded level. Call once per valid
 * classification. */
echo_trend_level_t echo_trend_push(echo_trend_t *tr, float readiness, float dt_ms);

/* Readiness as of ~ago_ms in the past: the newest held sample no more recent
 * than that instant, or the oldest sample if the window is not yet filled. */
float echo_trend_at(const echo_trend_t *tr, float ago_ms);

const char *echo_trend_name(echo_trend_level_t l);

#ifdef __cplusplus
}
#endif
#endif /* ECHO_TREND_H */
