import datetime
import glob
import json
import os
import shutil
from pathlib import Path

import cadquery as cq
import docker
import OCP.TopoDS
import pymeshlab
from cadquery import Shape, Workplane
from jinja2 import Environment, PackageLoader
from OCP import TopoDS
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.Interface import Interface_Static
from OCP.STEPControl import (
    STEPControl_ManifoldSolidBrep,
    STEPControl_Reader,
    STEPControl_Writer,
)
from OCP.StepShape import StepShape_AdvancedFace
from OCP.StlAPI import StlAPI
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from openbox import Advisor, Observation, logger
from openbox import space as sp
from tqdm import tqdm

from coam.util.choicebox import choicebox

env = Environment(loader=PackageLoader(f"coam", f"templates"))


def import_stl_to_cq(filename: str):
    shape = OCP.TopoDS.TopoDS_Shape()
    StlAPI.Read_s(shape, filename)
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    sew = BRepBuilderAPI_Sewing()
    while exp.More():
        sew.Add(exp.Current())
        exp.Next()
    sew.Perform()
    shape = sew.SewedShape()
    solid = BRepBuilderAPI_MakeSolid(OCP.TopoDS.TopoDS().Shell_s(shape))
    cq_shape = Shape.cast(solid.Shape())
    return cq.Workplane(f"XY").newObject([cq_shape])


def export_solid_to_step(shape, path):
    writer = STEPControl_Writer()
    writer.SetTolerance(1e-4)
    Interface_Static.SetIVal_s("write.surfacecurve.mode", False)
    Interface_Static.SetCVal_s("write.step.schema", "AP203")
    writer.Transfer(shape, STEPControl_ManifoldSolidBrep)
    writer.Write(path)


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
    rotation = sp.Real("rotation", 0.0, 120)
    space.add_variables([rotation])
    os.chdir("resources")
    part_name = choicebox(
        "Select part", [Path(file).stem for file in glob.glob("*.stl")]
    )
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    experiment_identifier = f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{part_name}"
    advisor = Advisor(
        space,
        num_objectives=2,
        task_id="OptimizeU",
        surrogate_type="prf",
        initial_trials=5,  # 2*(dim+1)
        init_strategy="sobol",
        acq_type="ehvi",
        acq_optimizer_type="local_random",
        ref_point=[0.25, 250],
    )
    max_runs = 55
    for i in tqdm(range(max_runs)):
        config = advisor.get_suggestion()
        result = generate_cae_and_simulate(
            config,
            f"{experiment_identifier}/Iter_{i}_Rot-{config['rotation']}",
            part_name,
        )
        if 0 in result["objectives"]:
            # Optionally resample around the config closely here until one converges, and use that config instead
            logger.info("FEM was not stable. Ignoring this iteration.")
            continue
        observation = Observation(config=config, objectives=result["objectives"])
        advisor.update_observation(observation)
        logger.info("Iter %d, objectives: %s." % (i, result["objectives"]))
    history = advisor.get_history()
    history.visualize_html(
        open_html=True, show_importance=True, verify_surrogate=True, advisor=advisor
    )


def generate_cae_and_simulate(config: dict, path: str, part_name: str):
    Path(f"iterations/{path}").mkdir(parents=True, exist_ok=True)
    shutil.copy(f"resources/{part_name}.stl", f"iterations/{path}/part_geometry.stl")
    # shutil.copy(f"resources/part_geometry.step", f"iterations/{path}")

    remesh_simplify_part(f"iterations/{path}/part_geometry.stl", config["rotation"])
    client = docker.from_env()
    client.images.pull(f"pymesh/pymesh")
    client.containers.run(
        "pymesh/pymesh",
        command=[
            "python",
            "/tmp/minkowski/minkowski.py",
            "part_geometry_simplified.stl",
            f"10.0",
            f"{path}",
        ],
        volumes={os.getcwd(): {"bind": "/tmp/", f"mode": "rw"}},
        detach=False,
        auto_remove=True,
    )
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(f"iterations/{path}/part_geometry_minkowski.stl")
    clean_bad_faces(ms)
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_repair_non_manifold_edges()
    ms.save_current_mesh(f"iterations/{path}/part_geometry_minkowski.stl")
    minkowski_of_part = import_stl_to_cq(
        f"iterations/{path}/part_geometry_minkowski.stl"
    )
    simplified_part = import_stl_to_cq(
        f"iterations/{path}/part_geometry_simplified.stl"
    )
    # part = cq.importers.importStep(f"iterations/{path}/part_geometry.step")
    # part = part.val().rotate((0, 0, 0), (0, 0, 1), config["rotation"])
    # part_box = part.BoundingBox()
    # part.exportStep(f"iterations/{path}/part_geometry.step")
    part_box = minkowski_of_part.val().BoundingBox()
    jaw_actuated = (
        (
            cq.Workplane(f"XY")
            .box(20, 80, part_box.zmax - part_box.zmin + 10)
            .translate((part_box.xmin, 0, (part_box.zmax + part_box.zmin) / 2))
        )
        - minkowski_of_part
    ).translate((10, 0, 0))
    jaw_fixed = (
        (
            cq.Workplane(f"XY")
            .box(20, 80, part_box.zmax - part_box.zmin + 10)
            .translate((part_box.xmax, 0, (part_box.zmax + part_box.zmin) / 2))
        )
        - minkowski_of_part
    ).translate((-10, 0, 0))
    export_solid_to_step(
        jaw_actuated.val().Solids()[0].wrapped, f"iterations/{path}/jaw_actuated.step"
    )
    export_solid_to_step(
        jaw_fixed.val().Solids()[0].wrapped, f"iterations/{path}/jaw_fixed.step"
    )
    export_solid_to_step(
        simplified_part.val().Solids()[0].wrapped,
        f"iterations/{path}/part_geometry_simplified.step",
    )
    with open(f"iterations/{path}/clamp.py", f"w") as fh:
        shutil.copy(f"templates/clamp.py", f"iterations/{path}/clamp.py")
    os.chdir(f"iterations/{path}")
    os.system(f"abaqus cae nogui=clamp.py")
    if os.path.exists("retry.flag"):
        # One retry with virtual topology for cases with slivered geometry
        os.system(f"abaqus cae nogui=clamp.py")

    results = json.load(open("results.coam"))
    displacement = results["u1"]
    stress = results["s"]
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return {"objectives": [displacement, stress]}


def remesh_simplify_part(filename: str, rotation: float):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    ms.meshing_isotropic_explicit_remeshing(
        adaptive=True, targetlen=pymeshlab.PercentageValue(1.5)
    )
    clean_bad_faces(ms)
    ms.compute_matrix_from_rotation(rotaxis="Z axis", angle=rotation)
    ms.save_current_mesh(f"{os.path.splitext(filename)[0]}_simplified.stl")
    return ms


def clean_bad_faces(ms):
    ms.compute_selection_by_self_intersections_per_face()
    ms.meshing_remove_selected_vertices_and_faces()
    # 5 passes of hole closing. If more were necessary, the original geometry was very degenerate and unsuitable.
    ms.meshing_close_holes(refinehole=True)
    ms.meshing_close_holes(refinehole=True)
    ms.meshing_close_holes(refinehole=True)
    ms.meshing_close_holes(refinehole=True)
    ms.meshing_close_holes(refinehole=True)
    return ms


if __name__ == "__main__":
    main()
