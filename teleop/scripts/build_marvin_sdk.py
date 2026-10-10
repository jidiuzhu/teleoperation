"""Build pinned 100341 control + kinematics libraries; never connect hardware."""
import argparse
from pathlib import Path
import subprocess
import platform
import json

ROOT = Path(__file__).resolve().parents[2] / 'third_party/marvin'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.sdk_root.resolve()
    output = root / 'SDK_PYTHON'
    for directory, name, flags in [('contrlSDK','libMarvinSDK.so',['-DCMPL_LIN']),
                                    ('kinematicsSDK','libKine.so',[])]:
        sources = sorted((root/directory).glob('*.cpp'))
        if not sources:
            raise FileNotFoundError(f'Missing SDK sources: {directory}')
        tmp = output / (name+'.building')
        subprocess.run(['g++','-std=c++11','-O2','-fPIC','-shared',*flags,
                        *map(str,sources),'-lpthread','-lrt','-o',str(tmp)], check=True)
        tmp.replace(output/name)
        print(f'Built {output/name}', flush=True)
    (output/'build-info.json').write_text(json.dumps({'machine':platform.machine(),
        'control_sdk':'contrlSDK (100341)', 'source_commit':'4ac74e392b658c23bdea2c24ccf37c77fa1ef19e'}, indent=2))


if __name__ == '__main__':
    main()
