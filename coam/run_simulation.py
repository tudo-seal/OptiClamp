import datetime
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from functools import reduce
from pathlib import Path
from wsgiref.simple_server import make_server

import cadquery as cq
import colorlog
import customtkinter
import docker
import optuna
import pymeshlab
import torch
from botorch.settings import validate_input_scaling
from cadquery import Workplane
from jinja2 import Environment, PackageLoader
from optuna import Trial
from optuna_dashboard import wsgi

import coam.globals
import coam.samplers.singletask_gp_qnehvi_with_random_chance as st_gp_qevhi
from coam.globals import FAILURE_VALUE, NUM_PARTS
from coam.util.choicedialog import CTkChoiceDialog
from coam.util.geometry_io import export_solid_to_step, import_stl_as_shape
from coam.util.inputdialog import CTkInputDialog

env = Environment(loader=PackageLoader(f"coam", f"templates"))

part_mesh: pymeshlab.MeshSet = None
part: cq.Workplane = None


def get_faces_for_ids(face_ids: list[int], part: Workplane):
    return [part.val().Faces()[i] for i in face_ids]


def main():
    customtkinter.set_appearance_mode("dark")
    customtkinter.CTk()
    experiment_name = CTkInputDialog(
        text="Name for optimisation: ", title="Experiment Name", entry_text="AM_OPT"
    ).get_input()
    global part_mesh, part
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    part_names = []
    for i in range(NUM_PARTS):
        os.chdir("resources")
        part_name = CTkChoiceDialog(
            "Choose geometry",
            f"Select geometry for part {i}",
            [
                Path(file).stem
                for file in glob.glob("*.stl")
                if Path(file).stem not in part_names
            ],
        ).get_input()
        part_names.append(part_name)
        mesh_set = remesh_simplify_part(f"{part_name}.stl")
        coam.globals.mesh_sets.append(mesh_set)
        obstacle_name = CTkChoiceDialog(
            "Choose obstacle geometry",
            f"Select obstacle geometry for part {i}",
            [
                Path(file).stem
                for file in glob.glob("*.stl")
                if Path(file).stem not in part_names
            ],
        ).get_input()
        part_names.append(obstacle_name)
        obstacle_mesh_set = remesh_simplify_part(f"{obstacle_name}.stl")
        coam.globals.obstacle_sets.append(obstacle_mesh_set)
        coam.globals.cut_depths.append(
            float(
                CTkInputDialog(
                    text="Input a cut depth for orientation: ",
                    title="Cut Depth",
                    entry_text=float((i + 1) * 3),
                ).get_input()
            )
        )
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

    coam.globals.experiment_identifier = (
        f"{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}_{experiment_name}"
    )
    Path(f"iterations/{coam.globals.experiment_identifier}").mkdir(
        parents=True, exist_ok=True
    )
    client = docker.from_env()
    client.images.pull(f"pymesh/pymesh")

    validate_input_scaling(True)
    sampler = optuna.integration.BoTorchSampler(
        candidates_func=st_gp_qevhi.singletask_qehvi_candidates_func,
        n_startup_trials=17,
        independent_sampler=optuna.samplers.QMCSampler(),
    )
    study = optuna.create_study(
        storage="sqlite:///results.sqlite3",
        directions=["minimize", "minimize"],
        sampler=sampler,
        study_name=coam.globals.experiment_identifier,
    )

    sys.stderr = open(os.devnull, "w")
    handler = colorlog.StreamHandler(stream=sys.__stdout__)
    handler.setFormatter(
        colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(name)s:%(message)s")
    )

    coam.globals.logger = colorlog.getLogger(experiment_name)
    coam.globals.logger.addHandler(handler)
    coam.globals.logger.setLevel(logging.INFO)

    storage = optuna.storages.RDBStorage("sqlite:///results.sqlite3")
    app = wsgi(storage)
    httpd = make_server("localhost", 8080, app)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    webbrowser.open("http://localhost:8080/", new=0, autoraise=True)

    # Run study till the end of all time
    study.optimize(
        generate_cae_and_simulate,
        catch=(
            IndexError,
            IOError,
            ValueError,
        ),
    )


def generate_result_geometry():
    rotations = []
    customtkinter.set_appearance_mode("dark")
    customtkinter.CTk()
    experiment_name = CTkInputDialog(
        text="Name for result: ", title="Experiment Name", entry_text="AM_RESULT"
    ).get_input()
    global part_mesh, part
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    part_names = []
    for i in range(NUM_PARTS):
        os.chdir("resources")
        part_name = CTkChoiceDialog(
            "Choose geometry",
            f"Select geometry for part {i}",
            [
                Path(file).stem
                for file in glob.glob("*.stl")
                if Path(file).stem not in part_names
            ],
        ).get_input()
        part_names.append(part_name)
        mesh_set = pymeshlab.MeshSet()
        mesh_set.load_new_mesh(f"{part_name}.stl")
        coam.globals.mesh_sets.append(mesh_set)
        obstacle_name = CTkChoiceDialog(
            "Choose obstacle geometry",
            f"Select obstacle geometry for part {i}",
            [
                Path(file).stem
                for file in glob.glob("*.stl")
                if Path(file).stem not in part_names
            ],
        ).get_input()
        part_names.append(obstacle_name)
        obstacle_mesh_set = pymeshlab.MeshSet()
        obstacle_mesh_set.load_new_mesh(f"{obstacle_name}.stl")
        coam.globals.obstacle_sets.append(obstacle_mesh_set)
        coam.globals.cut_depths.append(
            float(
                CTkInputDialog(
                    text="Input a cut depth for orientation: ",
                    title="Cut Depth",
                    entry_text=float((i + 1) * 3),
                ).get_input()
            )
        )
        rotations.append(
            float(
                CTkInputDialog(
                    text="Input rotation of results to be examined: ",
                    title="Rotation",
                    entry_text=float(0),
                ).get_input()
            )
        )
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

    coam.globals.experiment_identifier = f"{experiment_name}"
    Path(f"iterations/{coam.globals.experiment_identifier}").mkdir(
        parents=True, exist_ok=True
    )
    client = docker.from_env()
    client.images.pull(f"pymesh/pymesh")

    generate_high_res_trial(rotations)


def generate_cae_and_simulate(trial: Trial):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    actuated_composite_jaws = []
    fixed_composite_jaws = []
    fixed_jaw_locations = []
    actuated_jaw_locations = []
    regular_parts = []
    path = f"{coam.globals.experiment_identifier}/Trial{trial.number}"
    Path(f"iterations/{path}").mkdir(parents=True, exist_ok=True)
    for i in range(NUM_PARTS):
        rotation = trial.suggest_float(f"rotation_{i}", 0.0, 360.0)
        coam.globals.logger.info(
            f"[TRIAL {trial.number}] Preparing FEM for orientation {i} with rotation of: {rotation}"
        )
        mesh_part = coam.globals.mesh_sets[i]
        obstacle_part = coam.globals.obstacle_sets[i]
        translate_center, _ = rotate_and_save_mesh(
            rotation,
            mesh_part,
            f"iterations/{path}/part_geometry_{i}.stl",
        )
        rotate_and_save_mesh(
            rotation,
            obstacle_part,
            f"iterations/{path}/obstacle_geometry_{i}.stl",
            seed_translate_center=translate_center,
        )

        create_minkowski_stl(
            coam.globals.cut_depths[i], f"part_geometry_{i}.stl", f"iterations/{path}"
        )
        create_minkowski_stl(
            coam.globals.cut_depths[i],
            f"obstacle_geometry_{i}.stl",
            f"iterations/{path}",
        )

        # Clean Minkowski Sum meshes to make sure we don't run into any trouble
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(f"iterations/{path}/part_geometry_{i}_minkowski.stl")
        clean_bad_faces(ms)
        ms.save_current_mesh(f"iterations/{path}/part_geometry_{i}_minkowski.stl")
        ms.load_new_mesh(f"iterations/{path}/obstacle_geometry_{i}_minkowski.stl")
        clean_bad_faces(ms)
        ms.save_current_mesh(f"iterations/{path}/obstacle_geometry_{i}_minkowski.stl")
        regular_part = import_stl_as_shape(f"iterations/{path}/part_geometry_{i}.stl")
        regular_parts.append(regular_part)
        minkowski_of_part = import_stl_as_shape(
            f"iterations/{path}/part_geometry_{i}_minkowski.stl"
        )
        minkowski_of_obstacle = import_stl_as_shape(
            f"iterations/{path}/obstacle_geometry_{i}_minkowski.stl"
        )
        part_bb = regular_part.val().BoundingBox()
        actuated_composite_jaws.append(
            cq.Workplane(f"XY")
            .box(30, 115, 25)
            .translate((part_bb.xmin - 15, 0, 0))
            .cut(minkowski_of_part)
            .cut(minkowski_of_obstacle)
            .translate((coam.globals.cut_depths[i], 0, 0))
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
            .cut(minkowski_of_obstacle)
            .translate((-coam.globals.cut_depths[i], 0, 0))
        )
        fixed_jaw_locations.append(fixed_composite_jaws[-1].val().BoundingBox().xmin)
        fixed_composite_jaws[-1] = fixed_composite_jaws[-1].translate(
            (-fixed_jaw_locations[-1], 0, 0)
        )

    # Make final geometries
    composite_fixed_jaw = reduce(
        lambda a, b: a.intersect(b, tol=1e-4), fixed_composite_jaws
    )
    composite_actuated_jaw = reduce(
        lambda a, b: a.intersect(b, tol=1e-4), actuated_composite_jaws
    )
    displacements = []
    # return {"objectives": displacements}
    for i in range(len(coam.globals.cut_depths)):
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
            process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.system(f"TASKKILL /F /PID {process.pid} /T")
            coam.globals.logger.warning(
                f"TRIAL {trial.number}] FEM ran far longer than expected. Assuming this is an issue with the geometry."
            )
            displacements.append(FAILURE_VALUE)
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
            continue
        results = json.load(open("results.coam"))
        u1 = results["u1"]
        if u1 == 0:
            coam.globals.logger.info(
                f"TRIAL {trial.number}] FEM failed, geometry is invalid. Setting to FAILURE_VALUE."
            )
            u1 = FAILURE_VALUE
        if u1 > 2 * FAILURE_VALUE:
            coam.globals.logger.info(
                f"TRIAL {trial.number}] FEM was far outside the expected displacements. "
                f"Assuming geometry generation failed, failing Trial."
            )
            raise ValueError("Trial geometry generation likely to be invalid.")
            u1 = FAILURE_VALUE
        if u1 > FAILURE_VALUE:
            coam.globals.logger.info(
                f"TRIAL {trial.number}] FEM failed, geometry is invalid. Setting to FAILURE_VALUE."
            )
            u1 = FAILURE_VALUE
        displacements.append(u1)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
    coam.globals.logger.info(
        f"TRIAL {trial.number}] Completed, Persisting Model. Values are: {tuple(displacements)}"
    )
    persist_model(path)
    persist_model(coam.globals.experiment_identifier)
    return tuple(displacements)


def generate_high_res_trial(rotations: list):
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    path = f"{coam.globals.experiment_identifier}"
    Path(f"results/{path}").mkdir(parents=True, exist_ok=True)
    fixed_jaws = pymeshlab.MeshSet()
    actuated_jaws = pymeshlab.MeshSet()
    box_thickness = max(coam.globals.cut_depths) + 10
    (
        cq.Workplane("XY")
        .box(box_thickness, 100, 25)
        .faces(">X")
        .workplane()
        .pushPoints([(38, 0), (-38, 0)])
        .circle(8.5 / 2)
        .extrude(10.0)
        .edges(">X")
        .chamfer(0.2)
        .val()
        .exportStl("results/fixed_jaw.stl")
    )
    (
        cq.Workplane("XY")
        .box(box_thickness, 100, 25)
        .faces("<X")
        .workplane()
        .pushPoints([(38, 0), (-38, 0)])
        .circle(8.5 / 2)
        .extrude(10.0)
        .edges("<X")
        .chamfer(0.2)
        .val()
        .exportStl("results/actuated_jaw.stl")
    )
    for i in range(NUM_PARTS):
        rotation = rotations[i]
        mesh_part = coam.globals.mesh_sets[i]
        obstacle_part = coam.globals.obstacle_sets[i]
        translate_center, part_bb = rotate_and_save_mesh(
            rotation,
            mesh_part,
            f"results/{path}/part_geometry_{i}.stl",
        )
        rotate_and_save_mesh(
            rotation,
            obstacle_part,
            f"results/{path}/obstacle_geometry_{i}.stl",
            seed_translate_center=translate_center,
        )

        create_minkowski_stl(
            coam.globals.cut_depths[i], f"part_geometry_{i}.stl", f"results/{path}"
        )
        create_minkowski_stl(
            coam.globals.cut_depths[i], f"obstacle_geometry_{i}.stl", f"results/{path}"
        )

        # Clean Minkowski Sum meshes to make sure we don't run into any trouble
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(f"results/{path}/part_geometry_{i}_minkowski.stl")
        clean_bad_faces(ms)
        ms.save_current_mesh(f"results/{path}/part_geometry_{i}_minkowski.stl")
        ms.load_new_mesh(f"results/{path}/obstacle_geometry_{i}_minkowski.stl")
        clean_bad_faces(ms)
        ms.save_current_mesh(f"results/{path}/obstacle_geometry_{i}_minkowski.stl")
        ms.load_new_mesh("results/fixed_jaw.stl")
        ms.compute_matrix_from_translation_rotation_scale(
            translationx=part_bb.max()[0] + box_thickness / 2,
            translationy=0,
            translationz=0,
        )
        ms.generate_boolean_difference(first_mesh=ms.current_mesh_id(), second_mesh=0)
        ms.generate_boolean_difference(first_mesh=ms.current_mesh_id(), second_mesh=1)
        ms.compute_matrix_from_translation_rotation_scale(
            translationx=-part_bb.max()[0] - box_thickness / 2,
            translationy=0,
            translationz=0,
        )
        fixed_jaws.add_mesh(ms.mesh(ms.current_mesh_id()))
        ms.save_current_mesh(f"results/{path}/fixed_jaw_{i}.stl")
        ms.load_new_mesh("results/actuated_jaw.stl")
        ms.compute_matrix_from_translation_rotation_scale(
            translationx=part_bb.min()[0] - box_thickness / 2,
            translationy=0,
            translationz=0,
        )
        ms.generate_boolean_difference(first_mesh=ms.current_mesh_id(), second_mesh=0)
        ms.generate_boolean_difference(first_mesh=ms.current_mesh_id(), second_mesh=1)
        ms.compute_matrix_from_translation_rotation_scale(
            translationx=-part_bb.min()[0] + box_thickness / 2,
            translationy=0,
            translationz=0,
        )
        actuated_jaws.add_mesh(ms.mesh(ms.current_mesh_id()))
        ms.save_current_mesh(f"results/{path}/actuated_jaw_{i}.stl")

    for i in range(NUM_PARTS - 1):
        fixed_jaws.generate_boolean_intersection(
            first_mesh=fixed_jaws.current_mesh_id(), second_mesh=i
        )
        actuated_jaws.generate_boolean_intersection(
            first_mesh=actuated_jaws.current_mesh_id(), second_mesh=i
        )
    fixed_jaws.save_current_mesh(f"results/{path}/fixed_jaw.stl")
    actuated_jaws.save_current_mesh(f"results/{path}/actuated_jaw.stl")


def persist_model(path):
    try:
        torch.save(st_gp_qevhi.train_x, f"iterations/{path}/x.pt")
        torch.save(st_gp_qevhi.train_y, f"iterations/{path}/y.pt")
        torch.save(st_gp_qevhi.bounds, f"iterations/{path}/bounds.pt")
        torch.save(st_gp_qevhi.model.state_dict(), f"iterations/{path}/model.pth")
    except Exception:
        pass


def create_minkowski_stl(cut_depth, name, path):
    client = docker.from_env()
    client.containers.run(
        "pymesh/pymesh",
        command=[
            "python",
            "/tmp/minkowski/minkowski.py",
            name,
            f"{cut_depth}",
            f"{path}",
        ],
        volumes={os.getcwd(): {"bind": "/tmp/", f"mode": "rw"}},
        detach=False,
        auto_remove=True,
    )


def rotate_and_save_mesh(rotation, mesh_part, path, seed_translate_center=None):
    rotated_mesh_part = pymeshlab.MeshSet()
    rotated_mesh_part.add_mesh(mesh_part.mesh(0))
    rotated_mesh_part.generate_copy_of_current_mesh()
    rotated_mesh_part.compute_matrix_from_rotation(rotaxis="Z axis", angle=rotation)
    translate_center = (
        rotated_mesh_part.mesh(0).bounding_box().max()
        + rotated_mesh_part.mesh(0).bounding_box().min()
    ) / 2
    if seed_translate_center is not None:
        translate_center = seed_translate_center
    rotated_mesh_part.compute_matrix_from_translation_rotation_scale(
        translationx=-translate_center[0],
        translationy=-translate_center[1],
        translationz=-translate_center[2],
    )
    rotated_mesh_part.save_current_mesh(path)
    return (
        translate_center,
        rotated_mesh_part.mesh(rotated_mesh_part.current_mesh_id()).bounding_box(),
    )


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
    generate_result_geometry()
