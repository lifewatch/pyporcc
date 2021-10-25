.. currentmodule:: pyporcc

Data Structures
=========================================

pyporcc implements several main data structures:
:class:`ClickDetector` (or :class:`ClickDetectorSoundTrapHF`)
:class:`ClickConverter`
:class:`Click`
:class:`PorCC` (or :class:`PorCCModel`)
:class:`PorpoiseClassifier`


ClickDetector
-------------

The :class:`ClickDetector` allows to run through files (continuous or SoundTrapHF) and detect possible clicks.
Return the output in a pandas DataFrame or on csv files (+h5 files to store the clicks snippets).


Overview of Attributes and Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A short summary of a few attributes and methods for ClickDetector is
presented here, and a full list can be found in the :doc:`all attributes and methods page <reference>`.


PorCC
------------------------------

A :class:`PorCC` is the representation of the model to classify the clicks in porpoises clicks or not.


Overview of Attributes and Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A short summary of a few attributes and methods for PorCC is
presented here, and a full list can be found in the :doc:`all attributes and methods page <reference>`.


ClickConverter
------------------------------

A :class:`ClickConverter` calculates all the parameters needed to classify the clicks. Therefore it converts a click
snippet into a pandas row (or a :class:`Click` object).


Overview of Attributes and Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A short summary of a few attributes and methods for ClickConverter is
presented here, and a full list can be found in the :doc:`all attributes and methods page <reference>`.