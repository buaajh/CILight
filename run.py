"""Use --help. Importing this module never starts a simulator."""
import argparse
import os
from pathlib import Path

from cilight.io import ROOT, NETWORKS, save, spec, verify_code, verify_inputs


def parser():
    p = argparse.ArgumentParser(description='CILight / CI-MPC four-network reference implementation')
    commands = p.add_subparsers(dest='command', required=True)
    for name in ('check', 'evaluate', 'train'):
        sub = commands.add_parser(name)
        sub.add_argument('--network', choices=NETWORKS, required=True)
        sub.add_argument('--data-root', type=Path, default=Path(os.environ.get('CILIGHT_DATA_ROOT', ROOT/'data')))
        if name != 'check':
            sub.add_argument('--output', type=Path, required=True)
        if name == 'evaluate':
            sub.add_argument('--method', choices=('ci-mpc', 'cilight', 'both'), default='both')
            sub.add_argument('--seeds', nargs='+', type=int, help='Default: the four original test seeds')
            sub.add_argument('--seconds', type=int, default=3600, help='3600 for paper protocol; shorter runs are smoke tests')
            sub.add_argument('--checkpoint', type=Path, help='Trusted custom CILight checkpoint (e.g. training output selected.pt)')
            sub.add_argument('--checkpoint-sha256', help='Required digest for a custom checkpoint')
    summary = commands.add_parser('summarize')
    summary.add_argument('--input', type=Path, required=True)
    summary.add_argument('--output', type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'summarize':
        from cilight.summary import summarize
        summarize(args.input, args.output)
        return
    settings = spec(args.network)
    verify_code()
    source = verify_inputs(settings, args.data_root)
    if args.command == 'check':
        print(f"PASS: {settings['name']}; {settings['signal_count']} signals; inputs/code/checkpoint SHA verified. No simulation run.")
        return
    if args.output.exists():
        raise FileExistsError('Use a new output directory; existing runs are never overwritten.')
    if args.command == 'evaluate':
        if args.seconds <= 0 or args.seconds % settings['interval']:
            raise ValueError('Seconds must be a positive multiple of the control interval.')
        seeds = settings['test_seeds'] if args.seeds is None else args.seeds
        if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
            raise ValueError('Seeds must be distinct nonnegative integers.')
        if bool(args.checkpoint) != bool(args.checkpoint_sha256):
            raise ValueError('Supply both --checkpoint and --checkpoint-sha256, or neither.')
        if args.checkpoint and args.method == 'ci-mpc':
            raise ValueError('CI-MPC does not use a checkpoint.')
    from cilight.runtime import configure, load_critic, run_episode
    configure(settings)
    args.output.mkdir(parents=True)
    save(args.output/'protocol.json', settings)
    if args.command == 'train':
        from cilight.training import train
        train(settings, source, args.output)
        return
    methods = ('ci-mpc', 'cilight') if args.method == 'both' else (args.method,)
    checkpoint = args.checkpoint or ROOT/settings['checkpoint']
    checkpoint_sha = args.checkpoint_sha256 or settings['checkpoint_sha256']
    critic = load_critic(checkpoint, checkpoint_sha) if 'cilight' in methods else None
    save(args.output/'evaluation_request.json', dict(seeds=seeds, seconds=args.seconds, methods=methods,
                                                   checkpoint_sha256=checkpoint_sha if critic else None))
    for seed in seeds:
        for method in methods:
            run_episode(settings, source, args.output/method/f'seed_{seed}', seed,
                        critic if method == 'cilight' else None, settings['evaluation_beta'] if method == 'cilight' else 0.,
                        args.seconds, args.output/'STOP')
    from cilight.summary import summarize
    summarize(args.output, args.output/'summary.csv')
    save(args.output/'evaluation_complete.json', dict(network=args.network, seeds=seeds, methods=methods,
                                                    seconds=args.seconds, paper_horizon=args.seconds == 3600))


if __name__ == '__main__':
    main()
