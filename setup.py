from pathlib import Path

from setuptools import find_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
requirements_path = Path(BASE_DIR, "requirements.txt")
required_packages = []
dependency_links = []

with open(requirements_path, encoding="utf8") as file:
    for ln in file:
        ln = ln.strip()
        if ln.startswith("./"):  # handle local directory
            dependency_links.append(ln)
        else:
            required_packages.append(ln)

docs_packages = ["mkdocs", "mkdocstrings"]

style_packages = ["black", "flake8", "isort"]

dev_packages = [
    "mkdocstrings[python]",
    "black[jupyter]",
    "autopep8",
    "pip-tools",
    "pandas",
    "pytest",
    "pytest-asyncio",
    "pytest-mock",
]

# Define our package
setup(
    name="LabeLMaker",
    version=0.1,
    description="Using generative AI and machine learning to categorize text",
    author="The Perioperative Data Science Team at The University of Alabama at Birmingham",
    author_email="ryangodwin@uabmc.edu",
    url="https://github.com/UABPeriopAI/label_maker.git",
    python_requires=">=3.11",
    packages=find_packages(),  # only look in directores with __init__.py
    install_requires=[required_packages],
    extras_require={"dev": docs_packages + style_packages, "docs": docs_packages},
    dependency_links=dependency_links,
)
