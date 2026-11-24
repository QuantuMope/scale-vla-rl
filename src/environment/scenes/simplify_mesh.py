# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Horizon Robotics and Hobot Contributors. All Rights Reserved.
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

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import trimesh


def simplify_mesh(collision_file: Path):
    output = collision_file.with_name("collision_file_simplified.obj")
    if output.is_file():
        return

    mesh = trimesh.load_mesh(collision_file)
    simplified = mesh.simplify_quadric_decimation(
        face_count=max(1, len(mesh.faces) // 5))
    with TemporaryDirectory(dir=output.parent) as temporary:
        temporary_mesh = Path(temporary) / output.name
        simplified.export(temporary_mesh)
        temporary_mesh.replace(output)


def main():
    parser = argparse.ArgumentParser(
        description="Simplify collision meshes in downloaded EmbodiedGen scenes.")
    parser.add_argument("--base-dir", required=True, type=Path)
    args = parser.parse_args()
    collision_files = sorted(args.base_dir.glob(
        "task_*/asset3d/*/result/mesh/*_collision.obj"))
    if not collision_files:
        parser.error(f"No scene collision meshes found in {args.base_dir}")
    for collision_file in collision_files:
        simplify_mesh(collision_file)


if __name__ == "__main__":
    main()
