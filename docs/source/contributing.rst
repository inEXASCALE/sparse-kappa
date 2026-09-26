Contributing
============

Development setup
-----------------

.. code-block:: bash

   git clone https://github.com/inEXASCALE/sparse-kappa.git
   cd sparse-kappa
   python -m pip install -e ".[dev,docs]"

Testing
-------

Run the test suite from repository root:

.. code-block:: bash

   python -m pytest tests -q

Documentation
-------------

Build docs locally:

.. code-block:: bash

   python -m sphinx -W --keep-going -b html docs/source /tmp/sparse-kappa-docs

Read :doc:`architecture` for module contracts and the complete validation workflow.
