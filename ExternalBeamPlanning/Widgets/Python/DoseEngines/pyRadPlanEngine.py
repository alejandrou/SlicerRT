import slicer
import sitkUtils
import numpy as np
import logging
import time
from scipy.sparse import coo_matrix
from DoseEngines import *
from Python.PyRadPlanUtils import prepareCt, prepareCst, preparePln

from slicer.i18n import tr as _

class pyRadPlanEngine(AbstractScriptedDoseEngine):
  """ pyRadPlan dose engine for SlicerRT External Beam Planning Module.
  """

  #------------------------------------------------------------------------------
  def __init__(self, scriptedEngine):
    scriptedEngine.name = 'pyRadPlan'
    scriptedEngine.isInverse = True #pyRadPlan has Inverse planning capabilities, i.e., it can compute a dose influence matrix
    scriptedEngine.canDoIonPlan = True
    scriptedEngine.supportsBodySegment = True
    AbstractScriptedDoseEngine.__init__(self, scriptedEngine)

  #------------------------------------------------------------------------------
  def defineBeamParameters(self):
    self.scriptedEngine.addBeamParameterSpinBox(
    _("pyRadPlan parameters"), "numOfFractions", _("number of fractions"),
    _("Range of noise added to the prescription dose (+- half of the percentage of the Rx dose)"),
    0.0, 99.99, 30.0, 1.0, 2 )

    self.scriptedEngine.addBeamParameterComboBox(
    _("pyRadPlan parameters"), "radiationMode", _("radiation mode"), _("comment"),
    [_("photons"), _("protons"), _("carbons")], 0)

    self.scriptedEngine.addBeamParameterComboBox(
    _("pyRadPlan parameters"), "machine", _("machine"), _("comment"), [_("generic")], 0)

  #------------------------------------------------------------------------------
  def updateBeamParametersForIonPlan(self, isIonPlanActive):
    if isIonPlanActive:
      availableRadiationModes = [_("protons"), _("carbons")]
      parameterLabel = _("radiation mode (ion)")
    else:
      availableRadiationModes = [_("photons"), _("protons"), _("carbons")]
      parameterLabel = _("radiation mode")
    
    self.scriptedEngine.updateBeamParameterComboBox(
    _("pyRadPlan parameters"), "radiationMode", parameterLabel,
    _("comment"), availableRadiationModes, 0)

  #------------------------------------------------------------------------------
  def calculateDoseUsingEngine(self, beamNode, resultDoseVolumeNode):

    #################################### Import pyRadPlan libraries ######################################
    from pyRadPlan import (
      generate_stf,
      calc_dose_influence
    )


    ##################################### Prepare data structures ########################################
    # Prepare the ct
    tStart = time.time()
    ct = prepareCt(beamNode)
    tEnd = time.time()
    logging.info(f"Time to prepare CT: {tEnd - tStart}")

    # Prepare the cst (segmentations)
    tStart = time.time()
    cst = prepareCst(beamNode, ct)
    tEnd = time.time()
    logging.info(f"Time to prepare CST: {tEnd - tStart}")

    # Prepare the plan configuration
    tStart = time.time()
    pln = preparePln(beamNode, ct)
    tEnd = time.time()
    logging.info(f"Time to prepare PLN: {tEnd - tStart}")

    # Generate Steering Geometry ("stf")
    tStart = time.time()
    stf = generate_stf(ct, cst, pln)
    tEnd = time.time()
    logging.info(f"Time to generate STF: {tEnd - tStart}")

    # Calculate Dose Influence Matrix ("dij")
    tStart = time.time()
    dij = calc_dose_influence(ct, cst, stf, pln)
    tEnd = time.time()
    logging.info(f"Time to calculate Dij: {tEnd - tStart}")


    ##################################### Visualize dose in Slicer #######################################
    # TEMPORARY: use uniform fluence
    fluence = np.ones(dij.total_num_of_bixels)
    result = dij.compute_result_ct_grid(fluence)
    totalDose = result["physical_dose"]

    planNode = beamNode.GetParentPlanNode()
    referenceVolumeNode = planNode.GetReferenceVolumeNode()

    # Push total dose to volumeNode in Slicer, set name & overlay on CT
    sitkUtils.PushVolumeToSlicer(totalDose, targetNode = resultDoseVolumeNode)
    resultDoseNodeName = str(planNode.GetName())+"_pyRadPlanDose_" + beamNode.GetName()
    resultDoseVolumeNode.SetName(resultDoseNodeName)
    slicer.util.setSliceViewerLayers(background=referenceVolumeNode, foreground=resultDoseVolumeNode)
    slicer.util.setSliceViewerLayers(foregroundOpacity=1)

    return str() #return empty string to indicate success
    
  #------------------------------------------------------------------------------
  def calculateDoseInfluenceMatrixUsingEngine(self, beamNode):
        
    #################################### Import pyRadPlan libraries ######################################
    from pyRadPlan import (
      generate_stf,
      calc_dose_influence
    )


    ##################################### Prepare data structures ########################################
    # Prepare the ct
    tStart = time.time()
    ct = prepareCt(beamNode)
    tEnd = time.time()
    logging.info(f"Time to prepare CT: {tEnd - tStart}")

    # Prepare the cst (segmentations)
    tStart = time.time()
    cst = prepareCst(beamNode, ct, needBody=True)
    tEnd = time.time()
    logging.info(f"Time to prepare CST: {tEnd - tStart}")

    # Prepare the plan configuration
    tStart = time.time()
    pln = preparePln(beamNode, ct)
    tEnd = time.time()
    logging.info(f"Time to prepare PLN: {tEnd - tStart}")

    # Generate Steering Geometry ("stf")
    tStart = time.time()
    stf = generate_stf(ct, cst, pln)
    tEnd = time.time()
    logging.info(f"Time to generate STF: {tEnd - tStart}")

    # Calculate Dose Influence Matrix ("dij")
    tStart = time.time()
    dij = calc_dose_influence(ct, cst, stf, pln)
    tEnd = time.time()
    logging.info(f"Time to calculate Dij: {tEnd - tStart}")


    ###################################### Store dose in beamNode ########################################

    # Optimize storage such that we don't have multiple instances in memory
    # we use a coo matrix here as it is the most efficient way to get the matrix into slicer
    doseMatrix = coo_matrix(dij.physical_dose.flat[0])

    beamNode.SetDoseInfluenceMatrixFromTriplets(
      doseMatrix.shape[0], doseMatrix.shape[1],
      doseMatrix.row,
      doseMatrix.col,
      doseMatrix.data,
      dij.dose_grid.dimensions, #set dimensions of dose grid in beamNode for which the dose influence matrix is defined
      dij.dose_grid.resolution_vector #set spacing of dose grid in beamNode for which the dose influence matrix is defined
    )
    
    return str() #return empty string to indicate success
    
