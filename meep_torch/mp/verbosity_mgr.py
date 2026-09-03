# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Faithful port of meep/python/verbosity_mgr.py: a singleton managing one or
more per-library ``verbosity`` flags (levels 0-3) behind a single global
level. meep-torch has no C library cvar, so flags land on the dummy-object
fallback meep itself uses when a cvar is absent; assignments still propagate
to every registered flag and the instance is wired as ``mp.verbosity``.
"""


class Verbosity:
    """Pythonic access to the global verbosity level (meep Verbosity).

    Levels: 0 minimal, 1 a little, 2 a lot, 3 debugging. Singleton: every
    construction returns the same instance, adding its ``cvar`` to the managed
    set; each managed flag is also exposed as a property named after it
    (``verbosity.meep = 2``). Calling the instance (``verbosity(2)``) or
    ``verbosity.set(2)`` sets every managed flag.
    """

    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._master_verbosity = -1
        self._cvars = {}

    @classmethod
    def reset(cls):
        """Drop the singleton so the next construction starts fresh (tests)."""
        if cls._instance is not None:
            cls._instance._init()
            cls._instance = None

    def __init__(self, cvar=None, name=None, initial_level=1):
        """See ``add_verbosity_var()``."""
        self.add_verbosity_var(cvar, name, initial_level)

    def add_verbosity_var(self, cvar=None, name=None, initial_level=1):
        """Manage a new flag. ``cvar`` is any object with a ``verbosity``
        attribute (a dummy is substituted when absent, meep-style)."""
        if cvar is None or not hasattr(cvar, "verbosity"):
            class _dummy:
                def __init__(self):
                    self.verbosity = 1
            cvar = _dummy()
        if name is None:
            name = f"cvar_{len(self._cvars)}"
        self._cvars[name] = cvar
        self.make_property(name)
        if self._master_verbosity == -1:
            self.set(initial_level)

    def get(self):
        """Current global verbosity level."""
        return self._master_verbosity

    def get_all(self):
        """Values of all managed flags (debugging aid)."""
        return [value.verbosity for value in self._cvars.values()]

    def set(self, level):
        """Validate the range, set every managed flag; returns the old level."""
        if level < 0 or level > 3:
            raise ValueError("Only verbosity levels 0-3 are supported")
        old = self._master_verbosity
        for cvar in self._cvars.values():
            cvar.verbosity = level
        self._master_verbosity = level
        return old

    def __call__(self, level):
        return self.set(level)

    def __int__(self):
        return self.get()

    def __repr__(self):
        return f"Verbosity: level={self.get()}"

    def __gt__(self, o):
        return self.get() > o

    def __lt__(self, o):
        return self.get() < o

    def __eq__(self, o):
        return self.get() == o

    def __ne__(self, o):
        return self.get() != o

    def __le__(self, o):
        return self.get() <= o

    def __ge__(self, o):
        return self.get() >= o

    def make_property(self, name):
        """Class property exposing the flag ``name`` (get/set, range-checked)."""
        def _getter(self, name=name):
            return self._cvars[name].verbosity

        def _setter(self, level, name=name):
            if level < 0 or level > 3:
                raise ValueError("Only verbosity levels 0-3 are supported")
            self._cvars[name].verbosity = level

        setattr(Verbosity, name, property(_getter, _setter))


verbosity = Verbosity(name="meep")
