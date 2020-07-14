import os

try:
    from gencad import _native_component as torch_extensions
except ImportError:
    torch_extensions = None


def use_native_extension():
    if torch_extensions is None:
        if use_native_extension.print_message:
            print("Not using native torch extensions. Extensions not found.")
            use_native_extension.print_message = False
        return False

    if int(os.environ.get('GENCAD_NO_USE_TORCH_EXTENSION', 0)):
        if use_native_extension.print_message:
            print("Not using native torch extensions. Disabled by environment variable")
            use_native_extension.print_message = False
        return False

    return True

use_native_extension.print_message = True
