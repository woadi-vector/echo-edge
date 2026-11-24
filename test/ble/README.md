# BLE link regression harness

Drives `docs/` in headless Chromium against a scriptable fake strap
(`fakebt.js`) to check that the link recovers from the failures seen on
Windows Chrome: a reconnect that hangs, a link that stays "connected" but
stops notifying, a service lookup that fails, a manual re-pair, and a
dismissed chooser.

    cd docs && python3 -m http.server 8765 &
    node test/ble/run.js hang        # or zombie, failService, repair, cancel, contact

Each scenario prints the link state every 5 s for 30 s. Pass = the status
ends on `streaming` with packets still climbing and `maxInFlight` at 1.
