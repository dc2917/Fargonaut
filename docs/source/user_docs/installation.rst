Installation
============

From PyPI
---------

Create a virtual environment to keep the Python packages separate from system packages
and other local environments::

  $ python -m venv .venv

Activate the environment::

  $ source .venv/bin/activate

Install the package along with its dependencies::

  $ pip install Fargonaut


From source
-----------

The only requirements for installing Fargonaut are `Git`_ and `Python`_.

Assuming you have these installed, first, clone the repository::

  $ git clone https://github.com/dc2917/Fargonaut

Create a virtual environment to keep the Python packages separate from system packages
and other local environments::

  $ cd Fargonaut
  $ python -m venv .venv

Activate the environment::

  $ source .venv/bin/activate

Install the dependencies::

  $ pip install -r requirements.txt


Choosing a Matplotlib backend
-----------------------------

Fargonaut uses `Matplotlib`_ to generate plots of FARGO3D output data. Matplotlib
provides users with the ability to specify both the GUI and graphics libraries with
which to draw and display plots - collectively referred to as the "backend".

You can install the Python packges required to use either the GTK or Qt GUI libraries
with the Cairo graphics library with either::

  $ pip install Fargonaut[mpl_gtk]

or::

  $ pip install Fargonaut[mpl_qt]

Alternatively, install the requirements for your preferred Matplotlib backend. See
`here`_ for further details.

.. _Git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
.. _Python: https://www.python.org/downloads/
.. _Matplotlib: https://matplotlib.org/
.. _here: https://matplotlib.org/stable/users/explain/figure/backends.html
