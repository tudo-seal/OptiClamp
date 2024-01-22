import os
import pathlib
import random
import sys
from functools import reduce
from typing import Tuple, List

import OCP
import cadquery as cq
from OCP import TopoDS
from OCP.STEPControl import STEPControl_Reader
from OCP.StepShape import StepShape_AdvancedFace
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from cadquery import Shape, Workplane, Face

from jinja2 import Environment, PackageLoader
from pathlib import Path

env = Environment(loader=PackageLoader("coam", "templates"))


def import_step_with_markers(file_name: str) -> tuple[Workplane, list[int]]:
    reader = STEPControl_Reader()
    tr = reader.WS().TransferReader()
    reader.ReadFile(file_name)
    reader.TransferRoots()
    shape = reader.OneShape()

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    no_intersect_face_hashes = []
    while exp.More():
        s: TopoDS.TopoDS_Shape = exp.Current()
        exp.Next()
        item: StepShape_AdvancedFace = tr.EntityFromShapeResult(s, 1)
        if item.Name().ToCString() == "NoIntersect":
            no_intersect_face_hashes.append(s.HashCode(1000000000))
    cq_shape = Shape.cast(shape)
    no_intersect_face_ids = [
        idx
        for idx, face, in enumerate(cq_shape.Faces())
        if face.wrapped.HashCode(1000000000) in no_intersect_face_hashes
    ]
    return cq.Workplane("XY").newObject([cq_shape]), no_intersect_face_ids


def get_faces_for_ids(face_ids: list[int], part: Workplane):
    return [part.val().Faces()[i] for i in face_ids]


def main():
    part, face_ids = import_step_with_markers("resources/effector.step")
    part = part.rotateAboutCenter((0, 0, 1), random.randint(0, 360))
    part = part.rotateAboutCenter((0, 1, 0), random.randint(0, 360))
    face_box = reduce(
        lambda a, b: a.add(b),
        [face.BoundingBox() for face in get_faces_for_ids(face_ids, part)],
    )
    jaw_actuated = (
        cq.Workplane("XY")
        .box(20, 80, face_box.zmax - face_box.zmin + 10)
        .translate((face_box.xmin - 10, 0, (face_box.zmax + face_box.zmin) / 2))
        - part
    )
    jaw_fixed = (
        cq.Workplane("XY")
        .box(20, 80, face_box.zmax - face_box.zmin + 10)
        .translate((face_box.xmax + 10, 0, (face_box.zmax + face_box.zmin) / 2))
        - part
    )

    Path("iterations/1").mkdir(parents=True, exist_ok=True)
    # shutil.copytree("resources", "iterations/1", dirs_exist_ok=True)
    cq.exporters.export(part, "iterations/1/part_geometry.step")
    cq.exporters.export(jaw_actuated, "iterations/1/jaw_actuated.step")
    cq.exporters.export(jaw_fixed, "iterations/1/jaw_fixed.step")
    data = {
        "cwd": pathlib.Path().resolve().as_posix(),
        "iteration": "1",
    }
    with open("iterations/1/clamp.py", "w") as fh:
        fh.write(env.get_template("clamp.py.jinja").render(data))
    os.chdir("iterations/1")
    os.system("abaqus cae script=clamp.py")


if __name__ == "__main__":
    main()
