"""One entry point for running, resetting and benchmarking Scene11."""

import argparse
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {"run": "controller", "reset": "reset", "benchmark": "benchmark"}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=commands)
    if not args or args[0] not in commands:
        parser.parse_args(args)
        return
    from importlib import import_module

    import_module("." + commands[args[0]], __package__).main(args[1:])


if __name__ == "__main__":
    main()
