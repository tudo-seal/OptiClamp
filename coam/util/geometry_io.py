import cadquery as cq
import OCP
from cadquery import Shape, Workplane
from OCP import TopoDS
from OCP.BOPAlgo import BOPAlgo_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCP.gp import gp_Vec
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
from OCP.TopAbs import TopAbs_FACE, TopAbs_ShapeEnum, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer


def import_stl_as_shape(filename: str):
    shape = OCP.TopoDS.TopoDS_Shape()
    StlAPI.Read_s(shape, filename)
    sew = BRepBuilderAPI_Sewing()
    # sew.SetTolerance(1e-4)
    sew.Add(shape)
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
    # shape_fix.FixVertexPositionMode = True
    # face_tool = shape_fix.FixFaceTool()
    # face_tool.FixSmallAreaWireMode = True
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
    writer.Write(path)


def import_step_with_minkowski(file_name: str) -> Workplane:
    # This ONLY works if the step file has no faces with normals that are orthogonal to the sweep vector
    reader = STEPControl_Reader()
    reader.ReadFile(file_name)
    reader.TransferRoots()
    shape = reader.OneShape()

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    builder = BOPAlgo_Builder()
    while exp.More():
        face: TopoDS.TopoDS_Shape = exp.Current()
        exp.Next()
        shape = BRepPrimAPI_MakePrism(face, gp_Vec(10, 0, 0)).Shape()
        builder.AddArgument(shape)
    print("Fusing")
    builder.SetRunParallel(True)
    builder.Perform()

    cq_shape = Shape.cast(builder.Shape())
    cq_shape.exportStep("test.step")
    return cq.Workplane(f"XY").newObject([cq_shape])


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
