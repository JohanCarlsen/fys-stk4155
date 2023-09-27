# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys 
import os 
sys.path.insert(0, os.path.abspath('../../project1')) # Add here any underlying folders!
project = 'FYS-STK4155'
copyright = '2023, Johan Carlsen'
author = 'Johan Carlsen'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc', 
    'sphinx.ext.napoleon', 
    'numpydoc', 
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints'
]
autosummary_generate = True
autosummary_imported_members = True
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
