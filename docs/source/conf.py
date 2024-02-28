# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "Malpolon"
copyright = "2024, Théo Larcher, Titouan Lorieul"
author = "Théo Larcher, Titouan Lorieul"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_rtd_theme",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


todo_include_todos = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
"""
html_theme = "nature"
html_theme = "sphinx_rtd_theme"
html_theme = "classic"
html_theme = "pydata_sphinx_theme"
html_theme = "sphinx_book_theme"
"""
html_theme = "sphinx_rtd_theme"


html_theme_options = {
    "body_max_width": "none"
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for autodoc ----------------------------------------------------

# autodoc_class_signature = "separated"

# autodoc_member_order = "groupwise"

autodoc_default_options = {
    "show-inheritance": True,
    "special-members": "__len__, __getitem__",
    "undoc-members": True,
}

autodoc_inherit_docstrings = False


# -- Options for napoleon --------------------------------------------------

napoleon_numpy_docstring = True
napoleon_google_docstring = False

napoleon_include_init_with_doc = True
