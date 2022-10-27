# Multiclassifier

Current classes 
- LABEL: 0: "High PT" : PT < -0.2 GeV or PT > 0.2 GeV
- LABEL: 1: 0.2 GeV < PT < 0
- LABEL: 2: 0 < PT < 0.2 GeV

Plotting turn-on curve for multiclassifier
- Get y_values and errors from running negativeSideRebin.ipynb and positiveSideRebinUpper.ipynb
- Then, to make a graph, run graphErrors.ipynb. This will take in .out values from the previous step and make a graph.
- Older script was kerasShruti.C, which is a ROOT macro (requires ROOT and you to manually put in the values)
