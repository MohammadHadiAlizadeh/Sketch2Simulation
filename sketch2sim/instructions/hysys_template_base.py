import os
import re

from win32com.client import Dispatch


# === Step 1: Launch HYSYS and create a new case (Agent 1) ===
case_name = "FILL_CASE_NAME"  # Basis Agent must fill
hysys = Dispatch("HYSYS.Application")
hysys.Visible = True
case = hysys.SimulationCases.Add(case_name)
case.Visible = True

# === Step 2: Setup Property Package (Agent 1) ===
hybasis = case.BasisManager
fluidpkg = hybasis.FluidPackages.Add()
fluidpkg.PropertyPackageName = "FILL_PROPERTY_PACKAGE"  # Basis Agent must fill

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
base_dir = r""

safe_case_name = re.sub(r'[\\/*?:"<>|]', "_", case_name)
save_filename = safe_case_name + ".hsc"
save_path = os.path.join(base_dir, save_filename)
case.SaveAs(save_path)
print(f"Simulation saved at: {save_path}")

# === End of Template ===