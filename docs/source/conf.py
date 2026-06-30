import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'sparse-kappa'
author = 'Erin Carson, Xinye Chen'
copyright = '2026, Erin Carson, Xinye Chen'
release = '0.0.2'

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_title = 'sparse-kappa documentation'
html_static_path = ['_static']
html_theme_options = {
    'source_repository': 'https://github.com/chenxinye/sparse-kappa/',
    'source_branch': 'master',
    'source_directory': 'docs/source/',
    'navigation_with_keys': True,
}

myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
]
