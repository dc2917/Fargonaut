Installation
============

The only requirements for installing Fargonaut are `Git`_ and `Python`_.

Assuming you have these installed, first, clone the repository::

  $ git clone https://github.com/dc2917/Fargonaut

Create a virtual environment to keep the Python packages separate from system packages and other local environments::

  $ cd Fargonaut
  $ python -m venv .venv

Activate the environment::

  $ source .venv/bin/activate

Install the dependencies::

  $ pip install -r requirements.txt


.. _Git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
.. _Python: https://www.python.org/downloads/
