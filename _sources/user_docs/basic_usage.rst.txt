Basic Usage
===========

Locate the directory containing the FARGO3D output you want to open. For demonstrative purposes, we will use "/path/to/fargo3d/outputs", and use the "fargo" output, created when running the example "fargo" setup.

Open the Python interpreter::

  $ python

Import the top-level ``Output`` class::

  >>> from fargonaut.output import Output

Use this class to create an object for handling your output::

  >>> output = Output("/path/to/fargo3d/outputs/fargo")

Using the ``Output`` class::

  >>> help(output)

Retrieve variables defined in the variables.par file using the ``get_var`` method, e.g.::

  >>> output.get_var("FRAME")
  'G'

Some commonly used variables are stored as properties, in particular those related to the grid, e.g.::

  >>> output.ny
  128
  >>> output.nghy
  3
  >>> len(output.ydomain)
  135
  >>> output.xdomain
  array([-3.14159265, -3.12523019, -3.10886773, ..., 0, ... 3.14159265])

Use the ``get_field`` method to retrieve a field at a given output time. The names of the fields are those of the corresponding output files, e.g. to retrieve the gas density::

  >>> gasdens50 = output.get_field("gasdens", 50)

