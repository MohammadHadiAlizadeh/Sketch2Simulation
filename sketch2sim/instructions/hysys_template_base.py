import os
import re

from win32com.client import Dispatch

# === Step 1: Launch HYSYS and create a new case (Agent 1) ===
case_name = "FILL_CASE_NAME"  # Basis Agent must fill
hysys = Dispatch("HYSYS.Application")
hysys.Visible = True

# Close any existing case with the same name so the script is safe to re-run.
try:
    hysys.SimulationCases.Item(case_name).Close(False, "")
except Exception:
    pass

case = hysys.SimulationCases.Add(case_name)
case.Visible = True

# === Step 2: Setup Property Package (Agent 1) ===
# --------------------------------------------------------------------------
# PROPERTY PACKAGE TOKEN - HARD CONSTRAINT (read before writing line below)
# --------------------------------------------------------------------------
# `PropertyPackageName` accepts ONLY the internal HYSYS tokens listed here.
# It does NOT accept the display names printed in the HYSYS GUI.
#
#   VALID tokens - use exactly ONE of these, verbatim, spelled exactly:
#       "SRK"        "PRSV"       "LKP"        "MBWR"
#       "SourPR"     "SourSRK"    "NRTL"       "UNIQUAC"
#       "Antoine"    "BraunK10"   "Wilson"     "VanLaar"     "Margules"
#
#   INVALID - these raise com_error -2147024809 (E_INVALIDARG) and abort the
#   entire script on this line. NEVER emit any of them:
#       "Peng-Robinson"  "PengRobinson"  "Peng Robinson"  "PR"  "PR78"
#       "PR-BM"  "RK-Soave"  "RKS"  "RK"  "SRKKD"  "KabadiDanner"
#       "Ideal"  "RaoultsLaw"  "ChaoSeader"  "GraysonStreed"  "ZJ"
#       "LeeKeslerPlocker"  "PCSAFT"  "UNIFAC"  "ASME Steam"  "NBS Steam"
#
#   HOW TO CHOOSE (keep it simple - a valid token always beats a "better" one):
#       hydrocarbon / crude oil / gas processing ....... "PRSV" (or "SRK")
#       polar, aqueous, alcohol-rich, amine systems .... "NRTL"
#       sour / acid-gas systems ........................ "SourPR" or "SourSRK"
#       not sure ....................................... keep the default "SRK"
# --------------------------------------------------------------------------
hybasis = case.BasisManager
fluidpkg = hybasis.FluidPackages.Add()
fluidpkg.PropertyPackageName = "SRK"  # VALID TOKENS ONLY: SRK PRSV LKP MBWR SourPR SourSRK NRTL UNIQUAC Antoine BraunK10 Wilson VanLaar Margules - NEVER "Peng-Robinson"

# === Step 3: Add Components ===
components = [
    "COMPONENT_1",
    "COMPONENT_2",
]  # Basis Agent must fill the component list

# --- Component replacement blocks ---
# Add as many replacement blocks as needed.
if "COMPONENT_1" in components:
    components.remove("COMPONENT_1")
    components.extend(["HYSYS_COMP_1", "HYSYS_COMP_2"])  # Basis Agent must fill

if "COMPONENT_2" in components:
    components.remove("COMPONENT_2")
    components.extend(["HYSYS_COMP_1", "HYSYS_COMP_2"])  # Basis Agent must fill

for comp in components:
    fluidpkg.Components.Add(comp)
hybasis.EndBasisChange()
flowsheet = case.Flowsheet

# === Step 4: Add Unit Operations (Agent 2) ===
units = {}  # Instantiation Agent must fill, for example: "P-101" or "Pump-1"
# Instantiation Agent must add each unit operation in units according to the instructions file.

# === Step 5: Add Material Streams (Agent 2) ===
streams = {}  # Instantiation Agent must fill and must follow the unit_streamname pattern
# Instantiation Agent must add each material stream in streams according to the instructions file.

# === Step 6: Connect Streams (Agent 3) ===
# Configuration Agent must connect streams to units here according to the instructions file.
# Connections must be grouped per unit.

# === Step 7: Save Simulation ===
base_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(base_dir):
    base_dir = os.getcwd()

safe_case_name = re.sub(r'[\\/*?:"<>|]', "_", case_name)
save_filename = safe_case_name + ".hsc"
save_path = os.path.join(base_dir, save_filename)
case.SaveAs(save_path)
print(f"Simulation saved at: {save_path}")

# === End of Template ===
