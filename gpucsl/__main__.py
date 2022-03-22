import sys
from gpucsl.cli.cli import gpucsl_cli
from gpucsl.cli.cli_util import GpucslError


if __name__ == "__main__":
    try:
        gpucsl_cli(sys.argv[1:])
    except GpucslError:
        # do not show a stack trace to end users
        exit(1)
