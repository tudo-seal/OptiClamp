import random

import cadquery as cq
import OCP
from cadquery import Shape, Workplane
from OCP import TopoDS
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.BRepFeat import BRepFeat_SplitShape
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.gp import gp, gp_Pln, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.Interface import Interface_Static
from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Wireframe
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPControl import (
    STEPControl_ManifoldSolidBrep,
    STEPControl_Reader,
    STEPControl_Writer,
)
from OCP.StepShape import StepShape_AdvancedFace
from OCP.StlAPI import StlAPI
from OCP.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_Orientation,
    TopAbs_REVERSED,
    TopAbs_ShapeEnum,
    TopAbs_SHELL,
)
from OCP.TopExp import TopExp_Explorer


def import_stl_as_shape(filename: str):
    shape = OCP.TopoDS.TopoDS_Shape()
    StlAPI.Read_s(shape, filename)
    sew = BRepBuilderAPI_Sewing()
    sew.Add(shape)
    # sew.SetTolerance(1e-4)
    sew.Perform()
    shape: OCP.TopoDS.TopoDS_Shape = sew.SewedShape()

    if shape.IsNull():
        raise Exception("Critical error, STL could not be sewn into a shape.")
    if shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_COMPOUND:
        print("Shape is a compound, finding largest shell.")
        iterator = TopExp_Explorer(shape, TopAbs_SHELL)
        max_children = 0
        while iterator.More():
            print(f"Children: {iterator.Current().NbChildren()}")
            if iterator.Current().NbChildren() > max_children:
                max_children = iterator.Current().NbChildren()
                shape = iterator.Current()
            iterator.Next()
        if max_children == 0:
            raise Exception("Critical error, no shell had any content.")

    solid_builder = BRepBuilderAPI_MakeSolid(OCP.TopoDS.TopoDS().Shell_s(shape))
    solid_builder.Build()

    shape_upgrade = ShapeUpgrade_UnifySameDomain(solid_builder.Shape())
    # The value of these are very important
    # Too large leads to geometry degenerating in some cases, wrongly failing FEM
    # Too small does not recover much topology, and too small linear can lead to invalid STEPs that abaqus can not facet
    # Current value works well in all cases observed so far (obtained by manual binary search)
    shape_upgrade.SetAngularTolerance(4e-5)
    shape_upgrade.SetLinearTolerance(2e-4)
    shape_upgrade.Build()

    shape_fix = ShapeFix_Shape(shape_upgrade.Shape())
    shape_fix.Perform()
    wire_fix = ShapeFix_Wireframe(shape_fix.Shape())
    wire_fix.ModeDropSmallEdges = True
    wire_fix.FixSmallEdges()
    wire_fix.FixWireGaps()
    shape = wire_fix.Shape()

    if shape.ShapeType() == TopAbs_ShapeEnum.TopAbs_SHELL:
        solid_builder = BRepBuilderAPI_MakeSolid(OCP.TopoDS.TopoDS().Shell_s(shape))
        solid_builder.Build()
        shape = solid_builder.Shape()

    cq_shape = Shape.cast(shape)
    return cq.Workplane(f"XY").newObject([cq_shape])


def export_solid_to_step(shape, path):
    writer = STEPControl_Writer()
    # The Tolerance has a large effect on the speed and success of the FEMs
    # Too large can cause imprecise geometry errors or lead to slight clipping and over-closures needing to be resolved
    # Too small previously lead to zero elements occurring, this might be fixed by switching to pymesh boolean ops.
    # Current value works well though and leads to a fast FEM
    # writer.SetTolerance(1e-6)
    Interface_Static.SetIVal_s("write.surfacecurve.mode", False)
    Interface_Static.SetCVal_s("write.step.schema", "AP203")
    writer.Transfer(shape, STEPControl_ManifoldSolidBrep)
    # writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(path)


def import_step_with_minkowski(file_name: str) -> Workplane:
    # This ONLY works if the step file has no faces with normals that are orthogonal to the sweep vector
    reader = STEPControl_Reader()
    reader.ReadFile(file_name)
    reader.TransferRoots()
    shape = reader.OneShape()
    cq_shape = Shape.cast(shape)
    cq_shape = cq_shape.rotate((0, 0, 0), (0, 0, 1), random.randint(0, 360))
    exp = TopExp_Explorer(cq_shape.wrapped, TopAbs_FACE)
    shapes = []
    while exp.More():
        face: TopoDS.TopoDS_Face = OCP.TopoDS.TopoDS().Face_s(exp.Current())
        surface = BRepAdaptor_Surface(face)
        if surface.IsUClosed() or surface.IsVClosed():
            splitter = split_shape_in_center(face)
            for split_shape in splitter.DirectLeft():
                shapes.append(
                    BRepPrimAPI_MakePrism(split_shape, gp_Vec(3, 0, 0)).Shape()
                )
                shapes.append(
                    BRepPrimAPI_MakePrism(split_shape, gp_Vec(-3, 0, 0)).Shape()
                )
            for split_shape in splitter.Right():
                shapes.append(
                    BRepPrimAPI_MakePrism(split_shape, gp_Vec(3, 0, 0)).Shape()
                )
                shapes.append(
                    BRepPrimAPI_MakePrism(split_shape, gp_Vec(-3, 0, 0)).Shape()
                )
        elif surface.GetType() == GeomAbs_SurfaceType.GeomAbs_Plane:
            normal_vec = get_normal_vector_of_face(face, surface)
            if normal_vec[0] == 0:
                # These can never be relevant for the minkowski sum if one is only interested in the cut of that depth
                exp.Next()
                continue
            shapes.append(BRepPrimAPI_MakePrism(exp.Current(), gp_Vec(3, 0, 0)).Shape())
            shapes.append(
                BRepPrimAPI_MakePrism(exp.Current(), gp_Vec(-3, 0, 0)).Shape()
            )
        else:
            splitter = split_shape_in_center(exp.Current())
            for split_shape in splitter.DirectLeft():
                shapes.append(
                    BRepPrimAPI_MakePrism(split_shape, gp_Vec(3, 0, 0)).Shape()
                )
                shapes.append(
                    BRepPrimAPI_MakePrism(split_shape, gp_Vec(-3, 0, 0)).Shape()
                )
            for split_shape in splitter.Right():
                shapes.append(
                    BRepPrimAPI_MakePrism(split_shape, gp_Vec(3, 0, 0)).Shape()
                )
                shapes.append(
                    BRepPrimAPI_MakePrism(split_shape, gp_Vec(-3, 0, 0)).Shape()
                )
        exp.Next()

    fused_shape = cq_shape.copy()
    fused_shape = fused_shape.fuse(cq_shape.copy().translate((-3, 0, 0)))
    fused_shape = fused_shape.fuse(cq_shape.copy().translate((3, 0, 0)))
    shapes.append(fused_shape.wrapped)

    export_solid_to_step(
        cq.Compound.makeCompound([Shape.cast(shape) for shape in shapes]).wrapped,
        f"result.step",
    )


def get_normal_vector_of_face(face, surface):
    plane: gp_Pln = surface.Plane()
    orientation: TopAbs_Orientation = face.Orientation()
    normal_vec = (
        round(plane.Axis().Direction().X(), 4)
        if orientation != TopAbs_REVERSED
        else -round(plane.Axis().Direction().X(), 4),
        round(plane.Axis().Direction().Y(), 4)
        if orientation != TopAbs_REVERSED
        else -round(plane.Axis().Direction().Y(), 4),
        round(plane.Axis().Direction().Z(), 4)
        if orientation != TopAbs_REVERSED
        else -round(plane.Axis().Direction().Z(), 4),
    )
    return normal_vec


def split_shape_in_center(face):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    section = BRepAlgoAPI_Section(face, gp_Pln(props.CentreOfMass(), gp.DX_s()), False)
    section.ComputePCurveOn1(True)
    section.Approximation(True)
    export_solid_to_step(face, f"precut.step")
    section.Build()
    export_solid_to_step(section.Shape(), f"cut.step")
    exp_i = TopExp_Explorer(section.Shape(), TopAbs_EDGE)
    splitter = BRepFeat_SplitShape(face)
    while exp_i.More():
        face = OCP.TopoDS.TopoDS_Shape()
        if section.HasAncestorFaceOn1(exp_i.Current(), face):
            splitter.Add(
                OCP.TopoDS.TopoDS().Edge_s(exp_i.Current()),
                OCP.TopoDS.TopoDS().Face_s(face),
            )
        exp_i.Next()
    splitter.Build()
    return splitter


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
