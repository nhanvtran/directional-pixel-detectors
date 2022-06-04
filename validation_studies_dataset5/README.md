# Note

To fix the bug where there was an extra factor of 10 applied in the dataset, I applied a threshold of 8000 electrons instead of 800. When the datagen.py script and dataset gets updated, these clusterData.py scripts need to be revised to have a 800 electron threshold.
