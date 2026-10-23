"""Check BWA's intentional no-argument exit 1; not a successful mapping run."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    binary, output, raw_output = sys.argv[1:]
    result = subprocess.run([binary], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    data = result.stdout
    expected = b'Version: 0.7.17-r1188\n'
    passed = (result.returncode == 1 and len(data) <= 65536 and data.count(expected) == 1
              and b'Program: bwa (alignment via Burrows-Wheeler transformation)\n' in data
              and b'Usage:   bwa <command> [options]\n' in data)
    Path(raw_output).write_bytes(data)
    Path(output).write_text(json.dumps(dict(schema='fudan-bwa-version-probe-v1',
        status='expected_usage_exit_verified' if passed else 'version_probe_failed',
        actual_BWA_exit=result.returncode, expected_usage_exit=1,
        expected_version='0.7.17-r1188', output_sha256=hashlib.sha256(data).hexdigest(),
        mapping_performed=False), indent=2) + '\n')
    raise SystemExit(0 if passed else 2)


if __name__ == '__main__':
    main()
