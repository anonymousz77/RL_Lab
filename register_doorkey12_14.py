#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
register_doorkey12_14.py  —  Additive env registration; does not modify the base codebase.

Registers a LARGER SimpleDoorKey gym id (12-14 room grid) that REUSES the base codebase's
env.doorkey:DoorKeyEnv class UNCHANGED — only the room-size / view / max_steps kwargs differ
from the existing MiniGrid-SimpleDoorKey-Min5-Max10-View3 registration. This keeps the bigger
env inside the same HistoricalObsEnv family so the teacher can run the identical env later.

NOTE: the base class is `DoorKeyEnv` (the "SimpleDoorKey" env), not `SimpleDoorKeyEnv`.

With minRoomSize=12, maxRoomSize=14, DoorKeyEnv._gen_grid picks sizeX/sizeY in [12, 14] each
episode (door/key/agent randomized inside). Because DoorKeyEnv sets width=height=maxRoomSize,
HistoricalObsEnv yields an "image" obs of shape (14, 14, 4) — bigger than the (10, 10, 4) of
the Min5-Max10 runs. max_steps=196 (= maxRoomSize**2, the env's own default for max=14).

Importing this module registers the id (guarded against double-registration).
"""
import gymnasium as gym
import env  # noqa: F401  ensures the base env ids + DoorKeyEnv class are importable

ENV_ID = "MiniGrid-SimpleDoorKey-Min12-Max14-View3"

if ENV_ID not in gym.registry:
    gym.register(
        id=ENV_ID,
        entry_point="env.doorkey:DoorKeyEnv",   # base class, unchanged
        kwargs={
            "minRoomSize": 12,
            "maxRoomSize": 14,
            "agent_view_size": 3,
            "max_steps": 196,
        },
    )
