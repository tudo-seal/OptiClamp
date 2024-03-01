import json
import time

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
    type=DEFORMABLE_BODY,
)
jaw_actuated = model.parts["jaw_actuated"]
if not jaw_actuated.geometryValidity:
    jaw_actuated.setValues(geometryValidity=True)
    file = open("retry.flag", "w")
    file.close()
jaw_actuated.ConvertToAnalytical()

jaw_fixed_acis = session.openStep(
    "jaw_fixed.step",
    scaleFromFile=OFF,
)
jaw_fixed = model.PartFromGeometryFile(
    combine=False,
    dimensionality=THREE_D,
    geometryFile=jaw_fixed_acis,
    name="jaw_fixed",
    type=DEFORMABLE_BODY,
)
jaw_fixed = model.parts["jaw_fixed"]
if not jaw_fixed.geometryValidity:
    jaw_fixed.setValues(geometryValidity=True)
    file = open("retry.flag", "w")
    file.close()
jaw_fixed.ConvertToAnalytical()

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
if not part_geometry.geometryValidity:
    part_geometry.setValues(geometryValidity=True)
    file = open("retry.flag", "w")
    file.close()
part_geometry.ConvertToAnalytical()

# Having to set any validity manually necessitates virtual topology to mesh the whole part
# So no retry is needed

reference_point_load = jaw_actuated.ReferencePoint(point=(-10.0, 0.0, 0.0))
reference_point_static = jaw_fixed.ReferencePoint(point=(10.0, 0.0, 0.0))

jaw_actuated.Surface(name="Surface", side1Faces=jaw_actuated.faces)
jaw_fixed.Surface(name="Surface", side1Faces=jaw_fixed.faces)
part_geometry.Surface(name="Surface", side1Faces=part_geometry.faces)


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
jaw_actuated_set = jaw_actuated.Set(
    cells=jaw_actuated.cells,
    name="jaw_actuated-set",
)
jaw_fixed_set = jaw_fixed.Set(
    cells=jaw_fixed.cells,
    name="jaw_fixed-set",
)
part_geometry.SectionAssignment(
    offset=0.0,
    offsetField="",
    offsetType=MIDDLE_SURFACE,
    region=part_set,
    sectionName="part-section",
    thicknessAssignment=FROM_SECTION,
)
jaw_actuated.SectionAssignment(
    offset=0.0,
    offsetField="",
    offsetType=MIDDLE_SURFACE,
    region=jaw_actuated_set,
    sectionName="actuated-section",
    thicknessAssignment=FROM_SECTION,
)
jaw_fixed.SectionAssignment(
    offset=0.0,
    offsetField="",
    offsetType=MIDDLE_SURFACE,
    region=jaw_fixed_set,
    sectionName="fixed-section",
    thicknessAssignment=FROM_SECTION,
)


part_geometry.setMeshControls(
    elemShape=TET,
    regions=part_geometry.cells,
    technique=FREE,
)
part_geometry.setElementType(
    elemTypes=(ElemType(elemCode=C3D10, elemLibrary=STANDARD),),
    regions=part_set,
)
jaw_actuated.setMeshControls(
    elemShape=TET, regions=jaw_actuated.cells, technique=FREE, sizeGrowth=MAXIMUM
)
jaw_actuated.setElementType(
    elemTypes=(ElemType(elemCode=C3D10, elemLibrary=STANDARD),),
    regions=jaw_actuated_set,
)
jaw_fixed.setMeshControls(
    elemShape=TET, regions=jaw_fixed.cells, technique=FREE, sizeGrowth=MAXIMUM
)
jaw_fixed.setElementType(
    elemTypes=(ElemType(elemCode=C3D10, elemLibrary=STANDARD),),
    regions=jaw_fixed_set,
)

part_geometry.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=4)
part_geometry.generateMesh()

jaw_actuated.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=4)
jaw_actuated.generateMesh()

jaw_fixed.seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=4)
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

part_instance_set = model.rootAssembly.Set(
    name="jaw_actuated_Set",
    cells=part_instance.cells,
)

jaw_actuated_instance_set = model.rootAssembly.Set(
    name="jaw_actuated_Set",
    cells=jaw_actuated_instance.cells,
)

jaw_fixed_instance_set = model.rootAssembly.Set(
    name="jaw_fixed_Set",
    cells=jaw_fixed_instance.cells,
)

model.RigidBody(
    bodyRegion=jaw_actuated_instance_set,
    name="jaw_actuated_Rigid",
    refPointRegion=Region(
        referencePoints=(jaw_actuated_instance.referencePoints.values()[0],)
    ),
    refPointAtCOM=ON,
)
model.RigidBody(
    bodyRegion=jaw_fixed_instance_set,
    name="jaw_fixed_Rigid",
    refPointRegion=Region(
        referencePoints=(jaw_fixed_instance.referencePoints.values()[0],)
    ),
    refPointAtCOM=ON,
)

model.rootAssembly.regenerate()
analysis_step = model.StaticStep(
    initialInc=1,
    maxInc=1,
    maxNumInc=100,
    minInc=1e-10,
    name="Apply-Force",
    nlgeom=ON,
    previous="Initial",
)
analysis_step.control.setValues(
    allowPropagation=OFF,
    resetDefaultValues=OFF,
    timeIncrementation=(4.0, 8.0, 9.0, 16.0, 10.0, 4.0, 12.0, 8.0, 6.0, 3.0, 50.0),
)

model.fieldOutputRequests["F-Output-1"].setValues(
    variables=("U",),
)

model.EncastreBC(
    createStepName="Initial",
    localCsys=None,
    name="BC-jaw_fixed",
    region=jaw_fixed_instance_set,
)

model.DisplacementBC(
    amplitude=UNSET,
    createStepName="Initial",
    distributionType=UNIFORM,
    fieldName="",
    localCsys=None,
    name="BC-jaw_actuated",
    region=jaw_actuated_instance_set,
    u1=UNSET,
    u2=SET,
    u3=SET,
    ur1=SET,
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
    table=((0.685,),),
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
global_contact.mainSecondaryAssignments.appendInStep(
    assignments=(
        (
            jaw_actuated_instance.surfaces["Surface"],
            part_instance.surfaces["Surface"],
            MAIN,
        ),
        (
            jaw_fixed_instance.surfaces["Surface"],
            part_instance.surfaces["Surface"],
            MAIN,
        ),
    ),
    stepName="Initial",
)

global_contact.slidingFormulationAssignments.appendInStep(
    assignments=(
        (
            jaw_actuated_instance.surfaces["Surface"],
            part_instance.surfaces["Surface"],
            SMALL_SLIDING,
        ),
        (
            jaw_fixed_instance.surfaces["Surface"],
            part_instance.surfaces["Surface"],
            SMALL_SLIDING,
        ),
    ),
    stepName="Initial",
)

model.StdInitialization(
    name="ContactInitialization", overclosureTolerance=0.2, overclosureType=ADJUST
)
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

clamping_set = model.rootAssembly.Set(
    name="Clamping-Load",
    referencePoints=(jaw_actuated_instance.referencePoints.values()[0],),
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
    multiprocessingMode=THREADS,
)

job.writeInput()
mdb.saveAs(str(job_name) + ".cae")
job.submit()
job.waitForCompletion()
file = open(job_name + ".log")
if "*** ERROR CATEGORY:  PRE" in file.read():
    job_name = "ClampSimulation_no_multi"
    job_single = mdb.Job(
        contactPrint=OFF,
        echoPrint=OFF,
        explicitPrecision=SINGLE,
        historyPrint=OFF,
        model="ClampSimulation",
        modelPrint=OFF,
        name=job_name,
        nodalOutputPrecision=SINGLE,
        numCpus=1,
        numDomains=1,
        numGPUs=1,
        type=ANALYSIS,
        multiprocessingMode=THREADS,
    )
    job_single.writeInput()
    mdb.saveAs(str(job_name) + ".cae")
    job_single.submit()
    job_single.waitForCompletion()

file = open(job_name + ".log")
if not "Abaqus JOB " + job_name + " COMPLETED" in file.read():
    file = open("results.coam", "w")
    file.write(json.dumps({"u1": 0, "s": 0}))
    file.close()
    sys.exit()

odb = visualization.openOdb(str(job_name) + affix + ".odb")
frame = odb.steps["Apply-Force"].frames[-1]
max_u1 = 0
max_stress = 10

dispField = frame.fieldOutputs["U"]
for i in range(len(dispField.values)):
    u1 = abs(dispField.values[i].data[0])
    max_u1 = u1 if u1 > max_u1 else max_u1

file = open("results.coam", "w")
file.write(json.dumps({"u1": float(max_u1), "s": float(max_stress)}))
file.close()
