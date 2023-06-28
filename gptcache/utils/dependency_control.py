import subprocess

from gptcache.utils.error import PipInstallError
from gptcache.utils.log import gptcache_log


def prompt_install(package: str, warn: bool = False):  # pragma: no cover
    """
    Function used to prompt user to install a package.
    """
    cmd = f"pip install -q {package}"
    try:
        if warn and input(f"Install {package}? Y/n: ") != "Y":
            raise ModuleNotFoundError(f"No module named {package}")
        print(f"start to install package: {package}")
        subprocess.check_call(cmd, shell=True)
        print(f"successfully installed package: {package}")
        gptcache_log.info("%s installed successfully!", package)
    except subprocess.CalledProcessError as e:
        raise PipInstallError(package) from e
