# run this file to run the scripts in the correct order.

import os
import shutil
import sys

# make the folders for output
if os.path.exists("out"):
    shutil.rmtree("out")


os.mkdir("out")
os.mkdir("out/predictions")
os.mkdir("out/maps")
os.mkdir("out/maps/actual")
os.mkdir("out/maps/predicted")
os.mkdir("out/maps/forecasts")
os.mkdir("out/analysis")
os.mkdir("out/analysis/intervention")
os.mkdir("out/analysis/categorization")
os.mkdir("out/analysis/roc")
os.mkdir("out/analysis/comparison")
os.mkdir("out/analysis/pipes")


py = sys.executable

# run all the python scripts
print("Running GATPred.py...")
os.system(f'"{py}" GATPred.py')

print("Running ActualGraph.py...")
os.system(f'"{py}" ActualGraph.py')

print("Running comparison.py...")
os.system(f'"{py}" comparison.py')

print("Running lead_categorization.py...")
os.system(f'"{py}" lead_categorization.py')

print("Running book1_categorization.py...")
os.system(f'"{py}" book1_categorization.py')

print("Running dem2.py...")
os.system(f'"{py}" dem2.py')

print("Running Material.py...")
os.system(f'"{py}" Material.py')

print("Running future_forecast.py...")
os.system(f'"{py}" future_forecast.py')

print("Running intervention_analysis.py...")
os.system(f'"{py}" intervention_analysis.py')

print("Running roc_curve_analysis.py...")
os.system(f'"{py}" roc_curve_analysis.py')

print("Done!")
