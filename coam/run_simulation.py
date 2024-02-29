import datetime
import glob
import json
import os
import shutil
import subprocess
import tkinter.simpledialog
from functools import reduce
from pathlib import Path

import cadquery as cq
import docker
import pymeshlab
from cadquery import Workplane
from jinja2 import Environment, PackageLoader
from openbox import Advisor, Observation, logger
from openbox import space as sp
from tqdm import tqdm

from coam.util.choicebox import choicebox
from coam.util.geometry_io import export_solid_to_step, import_stl_as_shape

env = Environment(loader=PackageLoader(f"coam", f"templates"))

part_mesh: pymeshlab.MeshSet = None
part: cq.Workplane = None


def get_faces_for_ids(face_ids: list[int], part: Workplane):
    return [part.val().Faces()[i] for i in face_ids]


def main():
    experiment_name = tkinter.simpledialog.askstring(
        "Experiment Name", "Name for optimisation: "
    )
    global part_mesh, part
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    space = sp.Space()

    num_parts = 2
    rotations = []
    mesh_sets = []
    cut_depths = []
    part_names = []
    for i in range(num_parts):
        rotations.append(sp.Real(f"rotation_{i}", 0.0, 360))
        os.chdir("resources")
        part_name = choicebox(
            "Select part",
            [
                Path(file).stem
                for file in glob.glob("*.stl")
                if Path(file).stem not in part_names
            ],
        )
        part_names.append(part_name)
        cut_depths.append(
            tkinter.simpledialog.askinteger(
                "Cut Depth",
                "Input a cut depth for orientation: ",
                initialvalue=(i + 1) * 3,
            )
        )
        mesh_set = remesh_simplify_part(f"{part_name}.stl")
        mesh_sets.append(mesh_set)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
    space.add_variables(rotations)
    experiment_identifier = (
        f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{experiment_name}"
    )
    max_value = 0.25
    advisor = Advisor(
        space,
        num_objectives=num_parts,
        task_id=experiment_name,
        surrogate_type="prf",
        initial_trials=6,  # 2*(dim+1)
        init_strategy="sobol",
        acq_type="ehvi",
        acq_optimizer_type="local_random",
        ref_point=[max_value] * num_parts,
        rand_prob=0.2,
    )
    max_runs = 56
    client = docker.from_env()
    client.images.pull(f"pymesh/pymesh")

    for i in tqdm(range(max_runs)):
        config = advisor.get_suggestion()
        print()
        logger.info(f"Config is: {config}")
        result = generate_cae_and_simulate(
            config, f"{experiment_identifier}/Iter_{i}", mesh_sets, cut_depths
        )
        # Resample if that was numerically not stable (perhaps overkill, but we like being very safe)
        # failed_fem_indexes = [
        #     index
        #     for index, objective in enumerate(result["objectives"])
        #     if objective == 0
        # ]
        # if len(failed_fem_indexes) > 0:
        #     new_config = copy.deepcopy(config)
        #     for j in failed_fem_indexes:
        #         new_config[f"rotation_{j}"] += 0.1
        #
        #     logger.info(
        #         f"FEM was not stable for orientation {j}. Resampling close to this iteration."
        #     )
        #     logger.info(f"New config is: {new_config}")
        #     result = generate_cae_and_simulate(
        #         new_config,
        #         f"{experiment_identifier}/Iter_{i}_Retry",
        #         mesh_sets,
        #         cut_depths,
        #     )
        # We are sufficiently sure that this is an issue with the actual rotation, leading to no contact
        for j, objective in enumerate(result["objectives"]):
            if objective != 0:
                continue
            logger.info(
                f"FEM was not stable for orientation {j}. Setting to maximum displacement ({max_value})."
            )
            result["objectives"][j] = max_value

        observation = Observation(config=config, objectives=result["objectives"])
        advisor.update_observation(observation)
        logger.info("Iter %d, objectives: %s." % (i, result["objectives"]))
    history = advisor.get_history()
    history.visualize_html(
        open_html=True, show_importance=True, verify_surrogate=True, advisor=advisor
    )


def generate_cae_and_simulate(
    config: dict, path: str, mesh_parts: list[pymeshlab.MeshSet], cut_depths: list[int]
):
    Path(f"iterations/{path}").mkdir(parents=True, exist_ok=True)
    actuated_composite_jaws = []
    fixed_composite_jaws = []
    fixed_jaw_locations = []
    actuated_jaw_locations = []
    regular_parts = []
    for i, mesh_part in enumerate(mesh_parts):
        rotated_mesh_part = pymeshlab.MeshSet()
        rotated_mesh_part.add_mesh(mesh_part.mesh(0))
        rotated_mesh_part.generate_copy_of_current_mesh()
        rotated_mesh_part.compute_matrix_from_rotation(
            rotaxis="Z axis", angle=config[f"rotation_{i}"]
        )
        translate_center = (
            rotated_mesh_part.mesh(0).bounding_box().max()
            + rotated_mesh_part.mesh(0).bounding_box().min()
        ) / 2
        rotated_mesh_part.compute_matrix_from_translation_rotation_scale(
            translationx=-translate_center[0],
            translationy=-translate_center[1],
            translationz=-translate_center[2],
        )
        rotated_mesh_part.save_current_mesh(f"iterations/{path}/part_geometry_{i}.stl")

        client = docker.from_env()
        client.containers.run(
            "pymesh/pymesh",
            command=[
                "python",
                "/tmp/minkowski/minkowski.py",
                f"part_geometry_{i}.stl",
                f"{cut_depths[i]}",
                f"{path}",
            ],
            volumes={os.getcwd(): {"bind": "/tmp/", f"mode": "rw"}},
            detach=False,
            auto_remove=True,
        )

        # Clean Minkowski Sum mesh to make sure we don't run into any trouble
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(f"iterations/{path}/part_geometry_{i}_minkowski.stl")
        clean_bad_faces(ms)
        ms.save_current_mesh(f"iterations/{path}/part_geometry_{i}_minkowski.stl")
        regular_part = import_stl_as_shape(f"iterations/{path}/part_geometry_{i}.stl")
        regular_parts.append(regular_part)
        minkowski_of_part = import_stl_as_shape(
            f"iterations/{path}/part_geometry_{i}_minkowski.stl"
        )
        part_bb = regular_part.val().BoundingBox()
        actuated_composite_jaws.append(
            cq.Workplane(f"XY")
            .box(30, 115, 25)
            .translate((part_bb.xmin - 15, 0, 0))
            .cut(minkowski_of_part)
            .translate((cut_depths[i], 0, 0))
        )
        actuated_jaw_locations.append(
            actuated_composite_jaws[-1].val().BoundingBox().xmin
        )
        actuated_composite_jaws[-1] = actuated_composite_jaws[-1].translate(
            (-actuated_jaw_locations[-1], 0, 0)
        )
        fixed_composite_jaws.append(
            cq.Workplane(f"XY")
            .box(30, 115, 25)
            .translate((part_bb.xmax + 15, 0, 0))
            .cut(minkowski_of_part)
            .translate((-cut_depths[i], 0, 0))
        )
        fixed_jaw_locations.append(fixed_composite_jaws[-1].val().BoundingBox().xmin)
        fixed_composite_jaws[-1] = fixed_composite_jaws[-1].translate(
            (-fixed_jaw_locations[-1], 0, 0)
        )
    composite_fixed_jaw = reduce(lambda a, b: a & b, fixed_composite_jaws)
    composite_actuated_jaw = reduce(lambda a, b: a & b, actuated_composite_jaws)
    displacements = []
    # return {"objectives": displacements}
    for i in range(len(cut_depths)):
        Path(f"iterations/{path}/{i}").mkdir(parents=True, exist_ok=True)
        export_solid_to_step(
            composite_actuated_jaw.translate((actuated_jaw_locations[i], 0, 0))
            .val()
            .Solids()[0]
            .wrapped,
            f"iterations/{path}/{i}/jaw_actuated.step",
        )
        export_solid_to_step(
            composite_fixed_jaw.translate((fixed_jaw_locations[i], 0, 0))
            .val()
            .Solids()[0]
            .wrapped,
            f"iterations/{path}/{i}/jaw_fixed.step",
        )
        export_solid_to_step(
            regular_parts[i].val().Solids()[0].wrapped,
            f"iterations/{path}/{i}/part_geometry.step",
        )
        shutil.copy(f"templates/clamp.py", f"iterations/{path}/{i}/clamp.py")
        os.chdir(f"iterations/{path}/{i}")
        try:
            process = subprocess.Popen(["abaqus", "cae", "nogui=clamp.py"], shell=True)
            process.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            os.system(f"TASKKILL /F /PID {process.pid} /T")
            logger.info(
                "FEM ran far longer than expected. Assuming this is an issue with the geometry."
            )
            displacements.append(0)
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
            continue
            # file = open("retry.flag", "w")
            # file.close()
        # if os.path.exists("retry.flag"):
        #     # One retry with virtual topology for cases with slivered geometry
        #     try:
        #         process = subprocess.Popen(
        #             ["abaqus", "cae", "nogui=clamp.py"],
        #             shell=True,
        #         )
        #         process.wait(timeout=600)
        #     except subprocess.TimeoutExpired:
        #         os.system(f"TASKKILL /F /PID {process.pid} /T")
        #         logger.info(
        #             "FEM ran longer than expected even with virtual geometry. Assuming that geometry is not feasible."
        #         )
        #         displacements.append(0)
        #         os.chdir(os.path.dirname(os.path.realpath(__file__)))
        #         break
        results = json.load(open("results.coam"))
        displacements.append(results["u1"])
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return {"objectives": displacements}


def remesh_simplify_part(filename: str):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    ms.meshing_isotropic_explicit_remeshing(
        adaptive=True, targetlen=pymeshlab.PercentageValue(1.5)
    )
    clean_bad_faces(ms)
    return ms


def clean_bad_faces(ms):
    ms.meshing_merge_close_vertices()
    ms.meshing_repair_non_manifold_edges(method=1)
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_remove_null_faces()

    # ms.compute_selection_by_self_intersections_per_face()
    # ms.meshing_remove_selected_vertices_and_faces()
    # ms.meshing_close_holes(maxholesize=3000, newfaceselected=False)
    return ms


if __name__ == "__main__":
    main()
