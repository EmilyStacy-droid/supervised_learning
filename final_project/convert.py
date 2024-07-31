import nbformat
from nbconvert import PythonExporter

# Load the notebook
with open("Deliverable_1.ipynb") as f:
    notebook_content = nbformat.read(f, as_version=4)

# Convert the notebook to a Python script
python_exporter = PythonExporter()
python_script, _ = python_exporter.from_notebook_node(notebook_content)

# Save the Python script
with open("part_1.py", "w") as f:
    f.write(python_script)


