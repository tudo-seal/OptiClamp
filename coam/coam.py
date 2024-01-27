import os
import pathlib
import shutil
from pathlib import Path

import cadquery as cq
import docker
import OCP.TopoDS
import pymeshlab
from cadquery import Shape, Workplane
from jinja2 import Environment, PackageLoader
from OCP import TopoDS
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCP.STEPControl import (
    STEPControl_ManifoldSolidBrep,
    STEPControl_Reader,
    STEPControl_Writer,
)
from OCP.StepShape import StepShape_AdvancedFace
from OCP.StlAPI import StlAPI
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer

env = Environment(loader=PackageLoader("coam", "templates"))


def convert_stl_to_step(filename: str):
    shape = OCP.TopoDS.TopoDS_Shape()
    StlAPI.Read_s(shape, filename)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    sew = BRepBuilderAPI_Sewing()
    while exp.More():
        sew.Add(exp.Current())
        exp.Next()
    sew.Perform()
    writer = STEPControl_Writer()
    writer.Transfer(sew.SewedShape(), STEPControl_ManifoldSolidBrep)
    writer.Write(f"{os.path.splitext(filename)[0]}.step")
    return
    cq_shape = Shape.cast(sew.SewedShape())
    cq_shape.exportStep(f"{os.path.splitext(filename)[0]}.step", write_pcurves=False)


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
    Path("iterations/1").mkdir(parents=True, exist_ok=True)
    shutil.copy("resources/part_geometry.stl", "iterations/1")
    remesh_simplify_part("iterations/1/part_geometry.stl")

    client = docker.from_env()
    client.images.pull("pymesh/pymesh")
    client.containers.run(
        "pymesh/pymesh",
        command=[
            "python",
            "/tmp/minkowski/minkowski.py",
            "part_geometry_simplified.stl",
            "10",
            "1",
        ],
        volumes={os.getcwd(): {"bind": "/tmp/", "mode": "rw"}},
        detach=False,
        auto_remove=True,
    )
    convert_stl_to_step("iterations/1/part_geometry_simplified.stl")
    convert_stl_to_step("iterations/1/part_geometry_minkowski.stl")

    part = cq.importers.importStep("iterations/1/part_geometry_minkowski.step")
    part_box = part.val().BoundingBox()
    jaw_actuated = (
        cq.Workplane("XY")
        .box(20, 80, part_box.zmax - part_box.zmin + 10)
        .translate((part_box.xmin, 0, (part_box.zmax + part_box.zmin) / 2))
    )
    jaw_fixed = (
        cq.Workplane("XY")
        .box(20, 80, part_box.zmax - part_box.zmin + 10)
        .translate((part_box.xmax, 0, (part_box.zmax + part_box.zmin) / 2))
    )
    cq.exporters.export(jaw_actuated, "iterations/1/jaw_actuated.stl")
    cq.exporters.export(jaw_fixed, "iterations/1/jaw_fixed.stl")

    jaw_actuated_stl = remesh_simplify_jaw_base("iterations/1/jaw_actuated.stl")
    jaw_actuated_stl.load_new_mesh("iterations/1/part_geometry_minkowski.stl")
    jaw_actuated_stl.generate_boolean_difference(first_mesh=0, second_mesh=1)
    jaw_actuated_stl.save_current_mesh("iterations/1/jaw_actuated_cut.stl")

    jaw_fixed_stl = remesh_simplify_jaw_base("iterations/1/jaw_fixed.stl")
    jaw_fixed_stl.load_new_mesh("iterations/1/part_geometry_minkowski.stl")
    jaw_fixed_stl.generate_boolean_difference(first_mesh=0, second_mesh=1)
    jaw_fixed_stl.save_current_mesh("iterations/1/jaw_fixed_cut.stl")

    convert_stl_to_step("iterations/1/jaw_actuated_cut.stl")
    convert_stl_to_step("iterations/1/jaw_fixed_cut.stl")
    jaw_actuated = cq.importers.importStep("iterations/1/jaw_actuated_cut.step")
    jaw_fixed = cq.importers.importStep("iterations/1/jaw_fixed_cut.step")
    jaw_actuated = jaw_actuated.translate((10, 0, 0))
    jaw_fixed = jaw_fixed.translate((-10, 0, 0))

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


def remesh_simplify_part(
    filename: str,
):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=10000,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        planarquadric=True,
    )
    ms.meshing_isotropic_explicit_remeshing()
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=5000,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        planarquadric=True,
    )
    ms.save_current_mesh(f"{os.path.splitext(filename)[0]}_simplified.stl")
    return ms


def remesh_simplify_jaw_base(
    filename: str,
):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    # ms.meshing_surface_subdivision_midpoint(iterations=4)
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PercentageValue(2.00000)
    )
    ms.save_current_mesh(f"{os.path.splitext(filename)[0]}_simplified.stl")
    return ms


if __name__ == "__main__":
    main()
