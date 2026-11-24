# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Horizon Robotics and Hobot Contributors. All Rights Reserved.
# Modified for Scale VLA RL.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from itertools import product

import alf
import torch

from src.environment.scenes.utils import get_simple_obj_name


@dataclass(frozen=True)
class PutTask:
    graspable: str
    placeable: str
    weight: float = 1.0

    @property
    def instr(self) -> str:
        graspable = self.graspable.replace("_", " ")
        placeable = self.placeable.replace("_", " ")
        return f"put the {graspable} on the {placeable}."

    def succeeded(self, contacts: dict[str, torch.Tensor]) -> torch.Tensor:
        placed = contacts[f"{self.graspable}|{self.placeable}"]
        released = ~contacts[f"gripper|{self.graspable}"]
        if self.placeable == "table":
            return placed & released
        return placed & released & ~contacts[f"{self.graspable}|table"]


@alf.configurable
@dataclass(frozen=True)
class MoveTask:
    placeable: str
    direction: str
    weight: float = 0.01

    @property
    def instr(self) -> str:
        return f"move {self.placeable.replace('_', ' ')} {self.direction}"

    def succeeded(self, contacts: dict[str, torch.Tensor]) -> torch.Tensor:
        # Move tasks run until timeout to vary the object positions.
        return torch.zeros_like(next(iter(contacts.values())))


@alf.configurable
def get_tasks(
        graspable_objects: list[str],
        placeable_objects: list[str],
        include_table: bool = False,
        include_move_tasks: bool = False) -> list[PutTask | MoveTask]:
    """Build tasks for a scene with a fixed set of objects."""
    graspables = [get_simple_obj_name(name) for name in graspable_objects]
    placeables = [get_simple_obj_name(name) for name in placeable_objects]
    if include_table:
        placeables.append("table")

    tasks: list[PutTask | MoveTask] = [
        PutTask(graspable, placeable)
        for graspable, placeable in product(graspables, placeables)
    ]
    if include_move_tasks:
        directions = ["left", "right", "up", "down", "to the middle"]
        tasks.extend(MoveTask(placeable, direction)
                     for placeable, direction in product(placeables, directions))
    return tasks
