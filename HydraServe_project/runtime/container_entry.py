"""Small stdlib-only resource gate; exec native entry only after host permit.

No torch import before the original app: preserves its load/import overlap.
HYDRASERVE_EXEC_JSON is a JSON argv, never interpreted by a shell.
"""
import json
import os
from pathlib import Path
import time


def main():
    output = Path('/hydra-output')
    start_ticks = int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    temporary = output/'entry.tmp'
    with temporary.open('x') as handle:
        json.dump(dict(pid=os.getpid(),start_ticks=start_ticks,
            affinity=sorted(os.sched_getaffinity(0)),
            name=os.environ['HYDRASERVE_CONTAINER_NAME']), handle)
    temporary.rename(output/'entry.json')
    deadline = time.monotonic()+60
    while not (output/'permit.json').exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('no actual resource acknowledgement')
        time.sleep(.01)
    permit = json.loads((output/'permit.json').read_text())
    if (permit['name'] != os.environ['HYDRASERVE_CONTAINER_NAME']
            or permit['pid_start_ticks'] != start_ticks):
        raise RuntimeError('resource permit belongs to a different process')
    argv = json.loads(os.environ['HYDRASERVE_EXEC_JSON'])
    if not isinstance(argv,list) or not argv or any(not isinstance(x,str) for x in argv):
        raise ValueError('native entry requires an explicit argv')
    os.execvpe(argv[0],argv,os.environ)


if __name__ == '__main__':
    main()
