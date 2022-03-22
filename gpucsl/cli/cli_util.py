import colorama
import logging


class GpucslError(Exception):
    pass


def error(error_message: str):
    logging.error(colorama.Fore.RED + error_message + colorama.Style.RESET_ALL)
    raise GpucslError()


def warning(warning_message: str):
    logging.warning(colorama.Fore.YELLOW + warning_message + colorama.Style.RESET_ALL)
