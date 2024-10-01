Basic Usage
===========

Opening an output
-----------------

Open the Python interpreter::

  $ python

Import the top-level ``Output`` class::

  >>> from fargonaut.output import Output

Use this class to create an object for handling your output::

  >>> output = Output("/path/to/fargo3d/outputs/fargo")

Note that you should replace /path/to/fargo3d/outputs/fargo with the path to the output you wish to open. For demonstrative purposes, here we will use the "fargo" output that is created when running the example "fargo" setup.

The ``output`` object contains the high-level details of the simulation output, such as the variables set and the compilation options used, amongst other things::

  >>> help(output)
  Help on Output in module fargonaut.output object:

  class Output(builtins.object)
  |  Output(directory: str) -> None
  |
  |  A FARGO3D simulation output.
  ...

You can etrieve variables defined in the variables.par file using the ``get_var`` method, e.g.::

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

Opening field data files
------------------------

Use the ``get_field`` method to retrieve a field at a given output time. The names of the fields are those of the corresponding output files, e.g. to retrieve the gas density::

  >>> gasdens50 = output.get_field("gasdens", 50)

``gasdens50`` contains the gas density data at output number 50, along with associated information such as the coordinates at which it is defined::

  >>> help(gasdens50)
  Help on Density in module fargonaut.fields.density object:

  class Density(fargonaut.field.Field)
  |  Density(output, num: int) -> None
  |
  |  A FARGO3D gas density field.
  ...

  >>> gasdens50.x
  array([-3.13341142, -3.11704896, -3.1006865, ..., 3.13341142]).
  >>> gasdens50.y
  array([0.40820312, 0.42460938, 0.44101563, ..., 2.49179687])
  >>> gasdens50.z
  array([0.])
  >>> gasdens50.data
  array([[[0.00060339],
          [0.00058012],
          [0.00067794],
          ...,
  
Use a ``Field``'s ``plot`` method to generate a 1D line or 2D surface plot of the field as a function of one or more coordinates, e.g.::

  >>> gasdens50.plot(dims="xy")

will create a 2D surface plot of the gas density as a function of its :math:`x`- and :math:`y`-coordinates. As the coordinate system used in this simulation is cylindrical, this will be a function of azimuth and radius. To convert this to cartesian :math:`x` and :math:`y`, use::

  >>> gasdens50.plot("cartesian", "xy")

Generating 1D line plots of a field requires specifying the indices at which to slice the data. For example, to plot the gas density as a function of radius, at :math:`\phi = 0`, use::

  >>> gasdens50.plot("polar", "y", (288, 0))

as 288 is the index closest to :math:`\phi = \pi` (and 0 is the :math:`z` index, as ``nz`` is 1 for this simulation).

Operations on fields
--------------------

You can perform binary arithmetic operations on fields, provided they are defined at the same coordinates. You can therefore perform operations on scalar fields, as they are defined at cell centres, or on the same component of vector fields, as they are staggered along the same dimension. For example::

  >>> pressure50 = gasdens50 * gasenergy50**2
  >>> vx_bx_ratio50 = gasvx50 / bx50

are valid operations, while::

  >>> gasvx50 / gasdens50
  Exception: Cannot divide fields defined at different coordinates.

You can call the ``set_symbol`` method on derived fields for labelling axes when plotting::

  >>> pressure50 = gasdens50 * gasenergy50**2
  >>> print(pressure50.symbol)
  \Sigma_\mathrm{g} \times c_\mathrm{s}^2
  >>> pressure50.set_symbol("P")
  >>> print(pressure50.symbol)
  P
