import subprocess


def prompt_install(package): # pragma: no cover
    """
    Function used to prompt user to install a package.
    """
    try:
        cmd = f"pip install {package}"
        subprocess.check_call(cmd, shell=True)
        print(f'{package} installed successfully!')
    except subprocess.CalledProcessError:
        print(f'Ran into error installing {package}.')