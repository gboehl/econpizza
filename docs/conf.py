# -- Path setup --------------------------------------------------------------
import os
import sys
# autopep8: off
sys.path.insert(0, os.path.abspath(".."))
# must be called AFTER the above:
from econpizza import __version__
# autopep8: on


# -- Project information -----------------------------------------------------
project = 'econpizza'
copyright = '2023, Gregor Boehl'
author = 'Gregor Boehl'
version = __version__
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates", "**.ipynb_checkpoints"]
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""
html_title = "econpizza"
html_theme_options = {
    "path_to_docs": "docs",
    "show_toc_level": 9,
    "repository_url": "https://github.com/gboehl/econpizza",
    "repository_branch": "main",
    "launch_buttons": {
        "notebook_interface": "classic",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": False,
}

autoclass_content = "both"
autodoc_member_order = "bysource"
master_doc = 'content'
latex_use_parts = False
