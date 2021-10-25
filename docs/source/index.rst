Pyporcc |version|
=================


Description
-----------
PyPorCC is a package that allows the detection and classification of Harbor Porpoises' clicks.
The detection of clicks in continuous files is a python adaptation of the PAMGuard click detector algorithm.

"Gillespie D, Gordon J, McHugh R, McLaren D, Mellinger DK, Redmond P, Thode A, Trinder P, Deng XY (2008) PAMGUARD:
Semiautomated, open source software for real-time acoustic detection and localisation of cetaceans.
Proceedings of the Institute of Acoustics 30:54–62."

The classification is done using the PorCC algorithm, adapted to python from the paper:

"Cosentino, M., Guarato, F., Tougaard, J., Nairn, D., Jackson, J. C., & Windmill, J. F. C. (2019).
Porpoise click classifier (PorCC): A high-accuracy classifier to study harbour porpoises ( Phocoena phocoena ) in
the wild. The Journal of the Acoustical Society of America, 145(6), 3427–3434. https://doi.org/10.1121/1.5110908"

Also other models can be trained. The implemented ones so far are:

- `svc`: Support Vector Machines
- `lsvc`: Linear Support Vector Machines
- `RandomForest`: Random Forest
- `knn`: K-Nearest Neighbor


.. toctree::
  :maxdepth: 1
  :caption: Getting Started

  Installation <install>
  Examples <examples>

.. toctree::
  :maxdepth: 1
  :caption: User Guide

  Data Structures <data_structures>

.. toctree::
  :maxdepth: 1
  :caption: Reference Guide

  Classes, Attributes and Methods <reference>

Contact
------------

For any questions please relate to clea.parcerisas@vliz.be



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
