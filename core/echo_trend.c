#include "echo_trend.h"
#include <string.h>

void echo_trend_init(echo_trend_t *tr)
{
    memset(tr, 0, sizeof(*tr));
    tr->level = ECHO_TREND_NONE;
}

static float readiness_at_index(const echo_trend_t *tr, uint16_t i)
{
    /* i = 0 is the oldest held sample. */
    uint16_t start = (uint16_t)((tr->head + ECHO_TREND_CAP - tr->count) % ECHO_TREND_CAP);
    return tr->readiness[(uint16_t)((start + i) % ECHO_TREND_CAP)];
}

static float time_at_index(const echo_trend_t *tr, uint16_t i)
{
    uint16_t start = (uint16_t)((tr->head + ECHO_TREND_CAP - tr->count) % ECHO_TREND_CAP);
    return tr->t_ms[(uint16_t)((start + i) % ECHO_TREND_CAP)];
}

float echo_trend_at(const echo_trend_t *tr, float ago_ms)
{
    if (tr->count == 0) return 0.0f;
    const float target = tr->now_ms - ago_ms;
    /* Newest sample at or before the target instant. Walk from newest back. */
    for (int i = (int)tr->count - 1; i >= 0; i--) {
        if (time_at_index(tr, (uint16_t)i) <= target)
            return readiness_at_index(tr, (uint16_t)i);
    }
    /* Window not yet filled — fall back to the oldest sample we hold. */
    return readiness_at_index(tr, 0);
}

echo_trend_level_t echo_trend_push(echo_trend_t *tr, float readiness, float dt_ms)
{
    if (dt_ms > 0.0f) tr->now_ms += dt_ms;

    tr->readiness[tr->head] = readiness;
    tr->t_ms[tr->head]      = tr->now_ms;
    tr->head = (uint16_t)((tr->head + 1) % ECHO_TREND_CAP);
    if (tr->count < ECHO_TREND_CAP) tr->count++;

    const float r_now   = readiness;
    const float r_short = echo_trend_at(tr, ECHO_TREND_SHORT_MS);
    const float r_long  = echo_trend_at(tr, ECHO_TREND_LONG_MS);
    const float drop_short = r_short - r_now;   /* positive = readiness fell */
    const float drop_long  = r_long  - r_now;

    echo_trend_level_t lvl;
    if (r_now < ECHO_TREND_HIGH_FLOOR || drop_short >= ECHO_TREND_HIGH_DROP)
        lvl = ECHO_TREND_HIGH;
    else if (r_now < ECHO_TREND_ELEV_FLOOR || drop_short >= ECHO_TREND_ELEV_DROP)
        lvl = ECHO_TREND_ELEVATED;
    else if (drop_long >= ECHO_TREND_WATCH_DROP)
        lvl = ECHO_TREND_WATCH;
    else
        lvl = ECHO_TREND_NONE;

    tr->level = lvl;
    return lvl;
}

const char *echo_trend_name(echo_trend_level_t l)
{
    switch (l) {
        case ECHO_TREND_NONE:     return "NONE";
        case ECHO_TREND_WATCH:    return "WATCH";
        case ECHO_TREND_ELEVATED: return "ELEVATED";
        case ECHO_TREND_HIGH:     return "HIGH";
        default:                  return "UNKNOWN";
    }
}
