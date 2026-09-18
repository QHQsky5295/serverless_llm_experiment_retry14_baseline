#!/usr/bin/env python3
"""Show pinned project recovery commands, or clone them into a new directory."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True, help='Published recovery-projects.json')
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Clone and detach at recorded commits; default only prints a plan')
    args = parser.parse_args()
    rows = json.loads(args.map.read_text())
    jobs = []
    for row in rows:
        name = row['project']
        if '/' in name or name in {'', '.', '..'}:
            raise SystemExit('Invalid project directory')
        dest = args.destination / name
        if dest.exists() or dest.is_symlink():
            raise SystemExit('Destination already exists: ' + str(dest))
        if not row['remote'].startswith('https://github.com/'):
            raise SystemExit('Unexpected remote')
        jobs.append((row, dest))
    for row, dest in jobs:
        command = ['git', 'clone', '--single-branch', '--branch', row['branch'], row['remote'], str(dest)]
        print(json.dumps({'project': row['project'], 'clone': command, 'commit': row['commit']}, ensure_ascii=False), flush=True)
        if args.apply:
            subprocess.run(command, check=True)
            subprocess.run(['git', '-C', str(dest), 'checkout', '--detach', row['commit']], check=True)
    if args.apply:
        print('Cloned pinned sources. Follow RECOVERY_GUIDE.md for supplemental files, environments, external assets and validation.')


if __name__ == '__main__':
    main()
