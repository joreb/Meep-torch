# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port stub: meep/python/examples/cherenkov-radiation.py

BLOCKED: moving point-charge source.

meep-torch does not implement moving point-charge source. To port, meep-torch would need
a change_sources run-time callback driving a source position via sim.meep_time() (a moving charge), not a static current. See the meep original for the physics.

meep original: meep/python/examples/cherenkov-radiation.py
"""
