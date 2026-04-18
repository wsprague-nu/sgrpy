"""Sphinx configuration module."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime

project = "sgrpy"
year = datetime.datetime.now().year
copyright = f"2025-{year}, W.W. Sprague"
author = "W.W. Sprague"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    # "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

autoapi_dirs = ["../../python/sgrpy"]
autoapi_file_patterns = ["*.py"]
autoapi_ignore = ["**/tests/**"]
# autoapi_keep_files = True
autoapi_member_order = "groupwise"
autoapi_options = [
    "members",
    "inherited-members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_own_page_level = "attribute"
autoapi_type = "python"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
