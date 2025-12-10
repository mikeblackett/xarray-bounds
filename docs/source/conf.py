# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os

import xarray_bounds

project = 'xarray-bounds'
copyright = '2025, Mike Blackett'
author = 'Mike Blackett'
version = xarray_bounds.__version__
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    # 3rd party
    'myst_nb',
    'sphinxemoji.sphinxemoji',
]

# AutoDoc configuration
autosummary_generate = True
autodoc_typehints = 'none'

# Napoleon configuration
napoleon_google_docstring = False
napoleon_numpy_docstring = True

source_suffix = ['rst', '.md']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    'cf_xarray': ('https://cf-xarray.readthedocs.io/en/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'python': ('https://docs.python.org/3', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_baseurl = os.environ.get('READTHEDOCS_CANONICAL_URL', '/')
html_theme = 'sphinx_book_theme'
html_title = 'xarray-bounds'
html_context = {
    'github_user': 'mikeblackett',
    'github_repo': 'xarray-bounds',
    'github_version': 'main',
    'doc_path': 'doc',
}
html_theme_options = {
    'repository_url': 'https://github.com/mikeblackett/xarray-bounds',
    'use_repository_button': True,
    'header_links_before_dropdown': 8,
    'navbar_align': 'left',
    'footer_center': ['last-updated'],
}
templates_path = ['_templates']

# MyST configuration
myst_enable_extensions = [
    'colon_fence',
    'tasklist',
    'substitution',
]
myst_substitutions = {
    'project': project,
    'version': version,
    'release': release,
}
