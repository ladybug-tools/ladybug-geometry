# coding=utf-8
from functools import wraps


def immutable(cls):
    """A decorator for making an immutable class.

    If a class is immutable, attributes can only be assigned to the class under __init__
    The class will throw an AttributeError exception if one tries to assign
    attributes to the class.

    Usage:
        @immutable
        class FooImmutable(object):
            def __init__(self):
                self.bar = 10
        foo = FooImmutable()

        try:
            foo.bar = 24
        except AttributeError as e:
            print(e)

        > Failed to set bar to 24.  bar cannot be set on FooImmutable.
          Use `to_mutable()` method to get a mutable version of FooImmutable instance.
    """
    def immutablesetattr(self, key, value):
        if hasattr(self, key) and not key.startswith("_"):
            raise AttributeError(
                "Failed to set {1} to {2}. {1} cannot be set on {0}.\nUse "
                "`to_mutable()` method to get a mutable version of the {0} instance."
                .format(cls.__name__, key, value)
            )
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
        return wrapper

    cls.__setattr__ = immutablesetattr
    cls.__init__ = init_decorator(cls.__init__)
    return cls
