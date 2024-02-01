import datetime
import os
import pathlib
import shutil
from pathlib import Path

import cadquery as cq
import docker
import OCP.TopoDS
import pymeshlab
from OCP.Interface import Interface_Static
from cadquery import Shape, Workplane, Solid
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
from openbox import Optimizer, space as sp

env = Environment(loader=PackageLoader(f"coam", f"templates"))


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
    writer.SetTolerance(1e-8)
    Interface_Static.SetIVal_s("write.surfacecurve.mode", False)
    Interface_Static.SetCVal_s("write.step.schema", "AP203")
    writer.Transfer(sew.SewedShape(), STEPControl_ManifoldSolidBrep)
    writer.Write(f"{os.path.splitext(filename)[0]}.step")


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
    return cq.Workplane(f"XY").newObject([cq_shape]), no_intersect_face_ids


def get_faces_for_ids(face_ids: list[int], part: Workplane):
    return [part.val().Faces()[i] for i in face_ids]


def main():
    space = sp.Space()
    rotation = sp.Real("rotation", 0.0, 359.9)
    space.add_variables([rotation])
    opt = Optimizer(
        generate_cae_and_simulate,
        space,
        max_runs=50,
        task_id="OptimizeU",
        surrogate_type="prf",
        initial_runs=4,  # 2*(dim+1)
        init_strategy="sobol",
    )
    history = opt.run()
    print(history)


def generate_cae_and_simulate(config: dict):
    key = (
        "{date:%Y-%m-%d_%H-%M-%S}".format(date=datetime.datetime.now())
        + f"_Rot-{config['rotation']}"
    )
    Path(f"iterations/{key}").mkdir(parents=True, exist_ok=True)
    shutil.copy(f"resources/part_geometry.stl", f"iterations/{key}")
    remesh_simplify_part(f"iterations/{key}/part_geometry.stl", config["rotation"])
    client = docker.from_env()
    client.images.pull(f"pymesh/pymesh")
    client.containers.run(
        "pymesh/pymesh",
        command=[
            "python",
            "/tmp/minkowski/minkowski.py",
            "part_geometry_simplified.stl",
            f"10",
            f"{key}",
        ],
        volumes={os.getcwd(): {"bind": "/tmp/", f"mode": "rw"}},
        detach=False,
        auto_remove=True,
    )
    convert_stl_to_step(f"iterations/{key}/part_geometry_simplified.stl")
    convert_stl_to_step(f"iterations/{key}/part_geometry_minkowski.stl")
    part = cq.importers.importStep(f"iterations/{key}/part_geometry_minkowski.step")
    part_box = part.val().BoundingBox()
    jaw_actuated = (
        cq.Workplane(f"XY")
        .box(20, 80, part_box.zmax - part_box.zmin + 10)
        .translate((part_box.xmin, 0, (part_box.zmax + part_box.zmin) / 2))
    )
    jaw_fixed = (
        cq.Workplane(f"XY")
        .box(20, 80, part_box.zmax - part_box.zmin + 10)
        .translate((part_box.xmax, 0, (part_box.zmax + part_box.zmin) / 2))
    )
    cq.exporters.export(jaw_actuated, f"iterations/{key}/jaw_actuated.stl")
    cq.exporters.export(jaw_fixed, f"iterations/{key}/jaw_fixed.stl")
    jaw_actuated_stl = pymeshlab.MeshSet()
    jaw_actuated_stl.load_new_mesh(f"iterations/{key}/jaw_actuated.stl")
    jaw_actuated_stl.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(1.5)
    )
    jaw_actuated_stl.load_new_mesh(f"iterations/{key}/part_geometry_minkowski.stl")
    jaw_actuated_stl.generate_boolean_difference(first_mesh=0, second_mesh=1)
    jaw_actuated_stl.save_current_mesh(f"iterations/{key}/jaw_actuated_cut.stl")

    jaw_fixed_stl = pymeshlab.MeshSet()
    jaw_fixed_stl.load_new_mesh(f"iterations/{key}/jaw_fixed.stl")
    jaw_fixed_stl.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(1.5)
    )
    jaw_fixed_stl.load_new_mesh(f"iterations/{key}/part_geometry_minkowski.stl")
    jaw_fixed_stl.generate_boolean_difference(first_mesh=0, second_mesh=1)
    jaw_fixed_stl.save_current_mesh(f"iterations/{key}/jaw_fixed_cut.stl")

    convert_stl_to_step(f"iterations/{key}/jaw_actuated_cut.stl")
    convert_stl_to_step(f"iterations/{key}/jaw_fixed_cut.stl")
    jaw_actuated = cq.importers.importStep(f"iterations/{key}/jaw_actuated_cut.step")
    jaw_fixed = cq.importers.importStep(f"iterations/{key}/jaw_fixed_cut.step")
    jaw_actuated = jaw_actuated.translate((10, 0, 0))
    jaw_fixed = jaw_fixed.translate((-10, 0, 0))
    cq.exporters.export(jaw_actuated, f"iterations/{key}/jaw_actuated.step")
    cq.exporters.export(jaw_fixed, f"iterations/{key}/jaw_fixed.step")
    data = {
        "cwd": pathlib.Path().resolve().as_posix(),
        "iteration": f"{key}",
    }
    with open(f"iterations/{key}/clamp.py", f"w") as fh:
        fh.write(env.get_template(f"clamp.py.jinja").render(data))
    os.chdir(f"iterations/{key}")
    os.system(f"abaqus cae nogui=clamp.py")
    results = open("results.coam", "r")
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return {"objectives": [float(results.read())]}


def remesh_simplify_part(filename: str, rotation: float):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=10000,
        preserveboundary=True,
        preservenormal=True,
        preservetopology=True,
        planarquadric=True,
    )
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(1.5))
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.compute_matrix_from_rotation(rotaxis="Z axis", angle=rotation)
    ms.save_current_mesh(f"{os.path.splitext(filename)[0]}_simplified.stl")
    return ms


if __name__ == "__main__":
    main()
