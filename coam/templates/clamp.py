import json
import optimization
import os


import visualization
from abaqus import *
from abaqusConstants import *
from mesh import *
from optimization import *
import part
import interaction
import step

job_name = "ClampSimulation"
# Jinja varies this as well, part of hyperparameters
clamping_force = 1280.0

mdb.models.changeKey(fromName="Model-1", toName="ClampSimulation")
model = mdb.models["ClampSimulation"]

jaw_actuated_acis = session.openStep(
    "jaw_actuated.step",
    scaleFromFile=OFF,
)
jaw_actuated = model.PartFromGeometryFile(
    combine=False,
    dimensionality=THREE_D,
    geometryFile=jaw_actuated_acis,
    name="jaw_actuated",
    type=DISCRETE_RIGID_SURFACE,
)
jaw_actuated = model.parts["jaw_actuated"]

jaw_fixed_acis = session.openStep(
    "jaw_fixed.step",
    scaleFromFile=OFF,
)
jaw_fixed = model.PartFromGeometryFile(
    combine=False,
    dimensionality=THREE_D,
    geometryFile=jaw_fixed_acis,
    name="jaw_fixed",
    type=DISCRETE_RIGID_SURFACE,
)
jaw_fixed = model.parts["jaw_fixed"]

part_geometry_acis = session.openStep(
    "part_geometry.step",
    scaleFromFile=OFF,
)
part_geometry = model.PartFromGeometryFile(
    combine=False,
    dimensionality=THREE_D,
    geometryFile=part_geometry_acis,
    name="part_geometry",
    type=DEFORMABLE_BODY,
)
part_geometry = model.parts["part_geometry"]

reference_point_load = jaw_actuated.ReferencePoint(point=(-10.0, 0.0, 0.0))
reference_point_static = jaw_fixed.ReferencePoint(point=(10.0, 0.0, 0.0))

jaw_actuated.Surface(name="Surface", side1Faces=jaw_actuated.faces)
jaw_fixed.Surface(name="Surface", side1Faces=jaw_fixed.faces)
part_geometry.Surface(name="Surface", side1Faces=part_geometry.faces)

if os.path.exists("retry.flag"):
    try:
        jaw_fixed.createVirtualTopology(
            applyBlendControls=False,
            cornerAngleTolerance=30.0,
            ignoreRedundantEntities=False,
            mergeShortEdges=False,
            mergeSliverFaces=False,
            mergeSmallAngleFaces=False,
            mergeSmallFaces=True,
            mergeThinStairFaces=False,
            smallFaceAreaThreshold=0.1,
        )
        jaw_actuated.createVirtualTopology(
            applyBlendControls=False,
            cornerAngleTolerance=30.0,
            ignoreRedundantEntities=False,
            mergeShortEdges=False,
            mergeSliverFaces=False,
            mergeSmallAngleFaces=False,
            mergeSmallFaces=True,
            mergeThinStairFaces=False,
            smallFaceAreaThreshold=0.1,
        )
    except Exception:
        pass

additive_steel_material = model.Material(name="Stainless_Steel_316L")
additive_steel_material.setValues(description="")
additive_steel_material.Elastic(
    dependencies=0,
    moduli=LONG_TERM,
    noCompression=OFF,
    noTension=OFF,
    table=((193000.0, 0.3),),
    temperatureDependency=OFF,
    type=ISOTROPIC,
)
additive_steel_material.Density(
    dependencies=0,
    distributionType=UNIFORM,
    fieldName="",
    table=((7.9e-09,),),
    temperatureDependency=OFF,
)
additive_steel_material.Plastic(
    dataType=HALF_CYCLE,
    dependencies=0,
    extrapolation=CONSTANT,
    hardening=ISOTROPIC,
    numBackstresses=1,
    rate=OFF,
    scaleStress=None,
    staticRecovery=OFF,
    strainRangeDependency=OFF,
    table=((520.0, 0.0), (605.0, 0.3)),
    temperatureDependency=OFF,
)

steel = model.Material(name="Steel")
steel.Density(
    dependencies=0,
    distributionType=UNIFORM,
    fieldName="",
    table=((7.85e-09,),),
    temperatureDependency=OFF,
)
steel.setValues(materialIdentifier="")
steel.Elastic(
    dependencies=0,
    moduli=LONG_TERM,
    noCompression=OFF,
    noTension=OFF,
    table=((210000.0, 0.3),),
    temperatureDependency=OFF,
    type=ISOTROPIC,
)
steel.setValues(description="")

part_section = model.HomogeneousSolidSection(
    material="Stainless_Steel_316L", name="part-section", thickness=None
)
model.HomogeneousSolidSection(material="Steel", name="actuated-section", thickness=None)
model.HomogeneousSolidSection(material="Steel", name="fixed-section", thickness=None)

part_set = part_geometry.Set(
    cells=part_geometry.cells,
    name="part-set",
)
part_geometry.SectionAssignment(
    offset=0.0,
    offsetField="",
    offsetType=MIDDLE_SURFACE,
    region=part_set,
    sectionName="part-section",
    thicknessAssignment=FROM_SECTION,
)

try:
    jaw_actuated.RemoveCells(cellList=jaw_actuated.cells)
except Exception:
    pass

try:
    jaw_fixed.RemoveCells(cellList=jaw_fixed.cells)
except Exception:
    pass


part_geometry.setMeshControls(
    elemShape=TET,
    regions=part_geometry.cells,
    technique=FREE,
)
part_geometry.setElementType(
    elemTypes=(ElemType(elemCode=C3D10, elemLibrary=STANDARD),),
    regions=part_set,
)

part_geometry.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=4)
part_geometry.generateMesh()

jaw_actuated.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=500)
jaw_actuated.generateMesh()

jaw_fixed.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=500)
jaw_fixed.generateMesh()

part_instance = model.rootAssembly.Instance(
    dependent=ON, name="part_geometry-1", part=part_geometry
)
jaw_actuated_instance = model.rootAssembly.Instance(
    dependent=ON, name="jaw_actuated-1", part=jaw_actuated
)
jaw_fixed_instance = model.rootAssembly.Instance(
    dependent=ON, name="jaw_fixed-1", part=jaw_fixed
)
jaw_actuated_instance_set = model.rootAssembly.Set(
    name="jaw_actuated_Set", referencePoints=(jaw_actuated_instance.referencePoints[2],)
)

jaw_fixed_instance_set = model.rootAssembly.Set(
    name="jaw_fixed_Set", referencePoints=(jaw_fixed_instance.referencePoints[2],)
)

model.rootAssembly.regenerate()
model.StaticStep(
    initialInc=1,
    maxInc=1,
    maxNumInc=10000,
    minInc=1e-10,
    name="Apply-Force",
    nlgeom=ON,
    previous="Initial",
)

model.fieldOutputRequests["F-Output-1"].setValues(
    variables=(
        "S",
        "U",
    ),
)

model.EncastreBC(
    createStepName="Initial",
    localCsys=None,
    name="BC-jaw_fixed",
    region=jaw_fixed_instance_set,
)

model.XasymmBC(
    createStepName="Initial",
    localCsys=None,
    name="BC-jaw_actuated",
    region=jaw_actuated_instance_set,
)

model.DisplacementBC(
    amplitude=UNSET,
    createStepName="Initial",
    distributionType=UNIFORM,
    fieldName="",
    localCsys=None,
    name="BC-3",
    region=jaw_actuated_instance_set,
    u1=UNSET,
    u2=UNSET,
    u3=UNSET,
    ur1=UNSET,
    ur2=SET,
    ur3=SET,
)

model.ContactProperty("IntProp-Fric")
model.interactionProperties["IntProp-Fric"].TangentialBehavior(
    dependencies=0,
    directionality=ISOTROPIC,
    elasticSlipStiffness=None,
    formulation=PENALTY,
    fraction=0.005,
    maximumElasticSlip=FRACTION,
    pressureDependency=OFF,
    shearStressLimit=None,
    slipRateDependency=OFF,
    table=((0.24,),),
    temperatureDependency=OFF,
)
model.interactionProperties["IntProp-Fric"].NormalBehavior(
    allowSeparation=ON, constraintEnforcementMethod=DEFAULT, pressureOverclosure=HARD
)

global_contact = model.ContactStd(createStepName="Initial", name="GlobalContact")
global_contact.setValues(globalSmoothing=False)
global_contact.contactPropertyAssignments.appendInStep(
    assignments=((GLOBAL, SELF, "IntProp-Fric"),), stepName="Initial"
)
global_contact.includedPairs.setValuesInStep(
    addPairs=(
        (jaw_actuated_instance.surfaces["Surface"], part_instance.surfaces["Surface"]),
        (jaw_fixed_instance.surfaces["Surface"], part_instance.surfaces["Surface"]),
    ),
    stepName="Initial",
    useAllstar=OFF,
)
model.StdInitialization(name="ContactInitialization")
global_contact.initializationAssignments.appendInStep(
    assignments=(
        (
            jaw_actuated_instance.surfaces["Surface"],
            part_instance.surfaces["Surface"],
            "ContactInitialization",
        ),
        (
            jaw_fixed_instance.surfaces["Surface"],
            part_instance.surfaces["Surface"],
            "ContactInitialization",
        ),
    ),
    stepName="Initial",
)

# This makes the FEM run faster I think if it works
# model.contactDetection(interactionProperty="IntProp-Fric", separationTolerance=0.0, extendByAngle=None,
#                       includeMeshShell=ON, includeMeshSolid=ON, includeMeshMembrane=ON,
#                       includeNonOverlapping=OFF, createUnionOfMainSecondarySurfaces=ON, createUnionOfMainSurfaces=ON,
#                       createUnionOfSecondarySurfaces=ON,
#                       includeOverclosed=OFF, mergeWithinAngle=None,
#                       meshedGeometrySearchTechnique=USE_MESH)

clamping_set = model.rootAssembly.Set(
    name="Clamping-Load", referencePoints=(jaw_actuated_instance.referencePoints[2],)
)

model.ConcentratedForce(
    cf1=clamping_force,
    createStepName="Apply-Force",
    distributionType=UNIFORM,
    field="",
    localCsys=None,
    name="Load-Clamping_Force",
    region=clamping_set,
)

job = mdb.Job(
    contactPrint=OFF,
    echoPrint=OFF,
    explicitPrecision=SINGLE,
    historyPrint=OFF,
    model="ClampSimulation",
    modelPrint=OFF,
    name=job_name,
    nodalOutputPrecision=SINGLE,
    numCpus=24,
    numDomains=24,
    numGPUs=1,
    type=ANALYSIS,
)

mdb.jobs[job_name].writeInput()
mdb.saveAs(str(job_name) + ".cae")
job.submit()
job.waitForCompletion()
file = open("ClampSimulation.log")
if not "Abaqus JOB ClampSimulation COMPLETED" in file.read():
    # Try to fix with virtual topology if failed
    file = open("retry.flag", "w")
    file.close()
    file = open("results.coam", "w")
    file.write(json.dumps({"u1": 0, "s": 0}))
    file.close()
    sys.exit()

odb = visualization.openOdb(str(job_name) + ".odb")
frame = odb.steps["Apply-Force"].frames[-1]
max_u1 = 0
max_stress = 0

dispField = frame.fieldOutputs["U"]
for i in range(len(dispField.values)):
    u1 = dispField.values[i].data[0]
    max_u1 = u1 if u1 > max_u1 else max_u1

dispField = frame.fieldOutputs["S"]
for i in range(len(dispField.values)):
    stress = dispField.values[i].data[0]
    max_stress = stress if stress > max_stress else max_stress

file = open("results.coam", "w")
file.write(json.dumps({"u1": float(max_u1), "s": float(max_stress)}))
file.close()

# print(dispField)
