import nbformat
from nbclient import NotebookClient

with open("stock_direction_predictor.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")
print("Executing notebook cells...")
client.execute()

with open("stock_direction_predictor.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

import shutil
shutil.copy("stock_direction_predictor.ipynb", "backend/stock_direction_predictor.ipynb")
print("Notebook executed and saved successfully with all rendered outputs!")
