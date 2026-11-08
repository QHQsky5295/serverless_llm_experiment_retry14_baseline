"""Linux x86-64 pidfd UAPI for the qualified Conda runtime.

The runtime lacks Python's pidfd wrappers. These are kernel ABI bindings, not
PID-based fallbacks. Numbers are from Linux 6.8 syscall_64.tbl and the installed
asm/unistd_64.h; unsupported platforms and syscall errors remain explicit.
"""
import ctypes
import os


def _syscall(number, *args):
    if (os.uname().sysname != 'Linux' or os.uname().machine != 'x86_64'
            or ctypes.sizeof(ctypes.c_void_p) != 8):
        raise ValueError('pidfd binding requires Linux x86-64')
    syscall = ctypes.CDLL(None, use_errno=True).syscall
    syscall.restype = ctypes.c_long
    result = syscall(ctypes.c_long(number), *args)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def open_pidfd(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError('pidfd binding requires a positive PID')
    return _syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def send_pidfd_signal(fd, sig):
    if type(fd) is not int or fd < 0:
        raise ValueError('pidfd signal requires a valid descriptor')
    # NULL siginfo and flags=0 address the process pinned by this pidfd.
    return _syscall(424, ctypes.c_int(fd), ctypes.c_int(sig),
                    ctypes.c_void_p(), ctypes.c_uint(0))
