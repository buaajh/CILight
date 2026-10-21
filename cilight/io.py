"""Artifact IO and read-only scenario validation (no simulator/model imports)."""
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NETWORKS = ('grid16', 'ingolstadt21', 'nanshan55', 'newyork196')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def spec(network):
    if network not in NETWORKS:
        raise ValueError(network)
    return read(ROOT / 'configs' / (network + '.json'))


def verify_code():
    for item in read(ROOT / 'provenance/sources.json')['files']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Reference source/final weights changed: ' + item['path'])


def verify_inputs(settings, data_root):
    cfg = Path(data_root).resolve() / settings['sumocfg']
    for item in settings['inputs']:
        path = cfg.parent / item['path']
        if not path.is_file() or sha(path) != item['sha256']:
            raise ValueError('Missing or changed reference input: ' + str(path))
    tree = ET.parse(cfg)
    net = cfg.parent / tree.find('input/net-file').get('value')
    signals = {n.get('id') for n in ET.parse(net).getroot().findall('tlLogic')}
    if len(signals) != settings['signal_count']:
        raise ValueError('Signal count mismatch')
    return cfg


def isolated_config(source, destination):
    """Preserve inputs; redirect detector files into a fresh output directory."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    tree = ET.parse(source)
    groups = {}
    for node in tree.getroot().find('input'):
        paths = [(source.parent / x.strip()).resolve() for x in node.get('value').split(',')]
        if node.tag == 'additional-files':
            derived = []
            for index, path in enumerate(paths):
                extra = ET.parse(path)
                for child in extra.getroot().iter():
                    if child.get('file') is None:
                        continue
                    if child.tag not in ('e1Detector', 'e2Detector', 'inductionLoop', 'laneAreaDetector'):
                        raise ValueError('Unknown additional-file output semantics')
                    original = str((path.parent / child.get('file')).resolve())
                    if original not in groups:
                        groups[original] = destination / f'detector_group_{len(groups)}.xml'
                    child.set('file', str(groups[original]))
                target = destination / f'additional_{index}.xml'
                extra.write(target, encoding='utf-8', xml_declaration=True)
                derived.append(target)
            paths = derived
        elif node.tag not in ('net-file', 'route-files'):
            raise ValueError('Unknown scenario input: ' + node.tag)
        node.set('value', ','.join(str(p) for p in paths))
    for node in tree.getroot().iter('gui-settings-file'):
        node.set('value', str((source.parent / node.get('value')).resolve()))
    target = destination / 'scenario.sumocfg'
    tree.write(target, encoding='utf-8', xml_declaration=True)
    return target


def verify_result(directory):
    directory = Path(directory)
    marker = read(directory / 'complete.json')
    for name, expected in marker['sha256'].items():
        if sha(directory / name) != expected:
            raise ValueError('Result artifact changed: ' + str(directory / name))
    return read(directory / 'metrics.json')
