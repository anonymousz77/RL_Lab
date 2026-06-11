#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
register_doorkey8x8.py  —  Additive env registration; does not modify the base codebase.

Registers an 8x8 fixed-room SimpleDoorKey gym id that REUSES the base codebase's
env.doorkey:DoorKeyEnv class UNCHANGED — only the room-size kwargs differ from the
existing MiniGrid-SimpleDoorKey-Min5-Max10-View3 registration. This keeps the 8x8 env
inside the same HistoricalObsEnv family so the teacher can run the identical env later.

NOTE: the base class is `DoorKeyEnv` (the "SimpleDoorKey" env), not `SimpleDoorKeyEnv`.

With minRoomSize == maxRoomSize == 8, DoorKeyEnv._gen_grid forces sizeX = sizeY = 8
(fixed 8x8 room; door/key/agent still randomized inside), and HistoricalObsEnv yields an
"image" obs of shape (8, 8, 4).

Importing this module registers the id (guarded against double-registration).
"""
import gymnasium as gym
import env  # noqa: F401  ensures the base env ids + DoorKeyEnv class are importable

ENV_ID = "MiniGrid-SimpleDoorKey-Min8-Max8-View3"

if ENV_ID not in gym.registry:
    gym.register(
        id=ENV_ID,
        entry_point="env.doorkey:DoorKeyEnv",   # base class, unchanged
        kwargs={
            "minRoomSize": 8,
            "maxRoomSize": 8,
            "agent_view_size": 3,
            "max_steps": 150,
        },
    )
