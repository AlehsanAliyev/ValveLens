from roboflow import Roboflow

rf = Roboflow(api_key="vXRycTABOCITZJ1iN1Si")

# Valve dataset
project = rf.workspace().project("valve-detection-xcwdr")
dataset = project.version(1).download("yolov8", location="data/detection/valves")

# Gauge dataset
project = rf.workspace().project("gauge-fvzbs-dd2p2")
dataset = project.version(1).download("yolov8", location="data/detection/gauges")