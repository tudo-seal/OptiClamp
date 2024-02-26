import copy
import datetime
import glob
import json
import os
import shutil
import tkinter.simpledialog
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
from coam.util.geometry_io import export_solid_to_step, import_stl_to_cq

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
    for i in range(num_parts):
        rotations.append(sp.Real(f"rotation_{i}", 0.0, 360))
        os.chdir("resources")
        part_name = choicebox(
            "Select part", [Path(file).stem for file in glob.glob("*.stl")]
        )
        cut_depths.append(
            tkinter.simpledialog.askinteger(
                "Cut Depth",
                "Input a cut depth for orientation: ",
                initialvalue=(i + 1) * 5,
            )
        )
        mesh_set = remesh_simplify_part(f"{part_name}.stl")
        mesh_sets.append(mesh_set)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
    space.add_variables(rotations)

    experiment_identifier = (
        f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{experiment_name}"
    )
    advisor = Advisor(
        space,
        num_objectives=2,
        task_id=experiment_name,
        surrogate_type="prf",
        initial_trials=17,  # 2*(dim+1)
        init_strategy="sobol",
        acq_type="ehvi",
        acq_optimizer_type="local_random",
        ref_point=[0.25, 0.25],
        rand_prob=0.2,
    )
    max_runs = 67
    for i in tqdm(range(max_runs)):
        config = advisor.get_suggestion()
        print()
        logger.info(f"Config is: {config}")
        result = generate_cae_and_simulate(
            config, f"{experiment_identifier}/Iter_{i}", mesh_sets, cut_depths
        )
        # Resample if that was numerically not stable (perhaps overkill, but we like being very safe)
        failed_fem_indexes = [
            index
            for index, objective in enumerate(result["objectives"])
            if objective == 0
        ]
        if len(failed_fem_indexes) > 0:
            new_config = copy.deepcopy(config)
            for j in failed_fem_indexes:
                new_config[f"rotation_{j}"] += 0.1

            logger.info(
                f"FEM was not stable for orientation {j}. Resampling close to this iteration."
            )
            logger.info(f"New config is: {new_config}")
            result = generate_cae_and_simulate(
                new_config,
                f"{experiment_identifier}/Iter_{i}_Retry",
                mesh_sets,
                cut_depths,
            )
        # We are sufficiently sure that this is an issue with the actual rotation, leading to no contact
        if 0 in result["objectives"]:
            logger.info(
                f"FEM was not stable for orientation. Assuming this geometry is not feasible."
            )
            observation = Observation(config=config, objectives=[999999] * num_parts)
            advisor.update_observation(observation)
            continue

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
    actuated_composite_jaw_meshset = pymeshlab.MeshSet()
    fixed_composite_jaw_meshset = pymeshlab.MeshSet()
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
        client.images.pull(f"pymesh/pymesh")
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
        ms.load_new_mesh("templates/jaw_block.stl")
        ms.load_new_mesh("templates/jaw_block.stl")
        ms.load_new_mesh(f"iterations/{path}/part_geometry_{i}_minkowski.stl")
        clean_bad_faces(ms)
        ms.set_current_mesh(0)
        ms.compute_matrix_from_translation_rotation_scale(
            translationx=ms.mesh(2).bounding_box().min()[0] - 15 + cut_depths[i]
        )
        ms.set_current_mesh(1)
        ms.compute_matrix_from_translation_rotation_scale(
            translationx=ms.mesh(2).bounding_box().max()[0] + 15 - cut_depths[i]
        )
        ms.generate_boolean_difference(first_mesh=0, second_mesh=2)
        ms.generate_boolean_difference(first_mesh=1, second_mesh=2)
        ms.set_current_mesh(3)
        clean_bad_faces(ms)
        ms.compute_matrix_from_translation_rotation_scale(translationx=cut_depths[i])
        ms.set_current_mesh(4)
        clean_bad_faces(ms)
        ms.compute_matrix_from_translation_rotation_scale(translationx=-cut_depths[i])
        actuated_composite_jaw_meshset.add_mesh(ms.mesh(3))
        fixed_composite_jaw_meshset.add_mesh(ms.mesh(4))

    fixed_jaw_locations = []
    actuated_jaw_locations = []
    for i in range(len(cut_depths)):
        actuated_jaw_locations.append(
            actuated_composite_jaw_meshset.mesh(i).bounding_box().min()[0]
        )
        fixed_jaw_locations.append(
            fixed_composite_jaw_meshset.mesh(i).bounding_box().max()[0]
        )
        actuated_composite_jaw_meshset.set_current_mesh(i)
        actuated_composite_jaw_meshset.compute_matrix_from_translation_rotation_scale(
            translationx=-actuated_jaw_locations[i],
        )
        fixed_composite_jaw_meshset.set_current_mesh(i)
        fixed_composite_jaw_meshset.compute_matrix_from_translation_rotation_scale(
            translationx=-fixed_jaw_locations[i],
        )
    for i in range(len(cut_depths) - 1):
        actuated_composite_jaw_meshset.generate_boolean_intersection(
            first_mesh=i, second_mesh=actuated_composite_jaw_meshset.mesh_number() - 1
        )
        fixed_composite_jaw_meshset.generate_boolean_intersection(
            first_mesh=i, second_mesh=fixed_composite_jaw_meshset.mesh_number() - 1
        )
    actuated_composite_jaw_meshset.set_current_mesh(
        actuated_composite_jaw_meshset.mesh_number() - 1
    )
    clean_bad_faces(actuated_composite_jaw_meshset)
    actuated_composite_jaw_meshset.save_current_mesh(
        f"iterations/{path}/jaw_actuated.stl"
    )
    fixed_composite_jaw_meshset.set_current_mesh(
        fixed_composite_jaw_meshset.mesh_number() - 1
    )
    clean_bad_faces(fixed_composite_jaw_meshset)
    fixed_composite_jaw_meshset.save_current_mesh(f"iterations/{path}/jaw_fixed.stl")
    composite_actuated_jaw = import_stl_to_cq(f"iterations/{path}/jaw_actuated.stl")
    composite_fixed_jaw = import_stl_to_cq(f"iterations/{path}/jaw_fixed.stl")
    displacements = []
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
        part = import_stl_to_cq(f"iterations/{path}/part_geometry_{i}.stl")
        export_solid_to_step(
            part.val().Solids()[0].wrapped,
            f"iterations/{path}/{i}/part_geometry.step",
        )
        shutil.copy(f"templates/clamp.py", f"iterations/{path}/{i}/clamp.py")
        os.chdir(f"iterations/{path}/{i}")
        os.system(f"abaqus cae nogui=clamp.py")
        if os.path.exists("retry.flag"):
            # One retry with virtual topology for cases with slivered geometry
            os.system(f"abaqus cae nogui=clamp.py")
        results = json.load(open("results.coam"))
        displacements.append(results["u1"])
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        if results["u1"] == 0:
            break
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

    # ms.compute_selection_by_self_intersections_per_face()
    # ms.meshing_remove_selected_vertices_and_faces()
    # ms.meshing_close_holes(maxholesize=3000, newfaceselected=False)
    return ms


if __name__ == "__main__":
    main()
