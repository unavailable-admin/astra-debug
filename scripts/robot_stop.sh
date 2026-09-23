#!/usr/bin/env bash
# Minimal standard-library client: no NumPy/SDK imports and no sudo needed.
set -euo pipefail
exec "${ASTRABOT_PYTHON:-python3}" - "$@" <<'PY'
import argparse, json, os, socket, sys, time
p = argparse.ArgumentParser(description='Pause Astra executor and invalidate all queued work.')
p.add_argument('--socket', default=f'/tmp/astra-robot-{os.getuid()}/executor.sock')
a = p.parse_args()
start = time.monotonic()
try:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(.5)
        s.connect(a.socket)
        s.sendall(b'{"op":"pause"}\n')
        with s.makefile('rb') as f:
            result = json.loads(f.readline(65537))
    if not result.get('ok'):
        raise RuntimeError(result.get('error'))
    state = result['status']
    print(f"STOP ACK: {state['state']}; generation={state['generation']}; "
          f"receipt={(time.monotonic()-start)*1000:.1f}ms; error={state['error']}")
    print('This acknowledges command cancellation; it does not measure physical stopping time.')
except Exception as e:
    print(f'NO STOP ACK: {e}. Use the robot hardware stop if needed.', file=sys.stderr)
    sys.exit(1)
PY
