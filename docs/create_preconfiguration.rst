=========================
Guide for adding new data
=========================

Overview
========
``siibra`` uses preconfigured json files to instantiate objects for lazy loading
mechanism. This method allows easy addition of new atlases, parcellations,
reference spaces, maps, and data features. This page is meant as guide for
developers, curators, and siibra-python users who would like to utilize
capabilities of siibra for data analysis.

How to structure jsons?
=======================
In ``config_schema`` folder within
`siibra-python <https://github.com/FZJ-INM1-BDA/siibra-python>`_, the existing
schemes are provied as json files. These are meant to be a guide, particularly
for the minimum requirements.

Step-by-step
============

I. Determine the type
---------------------

1. Atlas
2. Parcellation
3. Reference space
4. Labelled map
5. Statistical map
6. Data feature:

    * Regional connectivity
    * Receptor fingerprint
    * Cortical or receptor Profile
    * Image Section
    * Volume of interest
    * Parcellation-based activity timeseries (tabular)

**Not listed:** If the data is not fitting to any of the categories and would
like to see your data in siibra, then a new schema is needed as otherwise CI
will fail. It is expected that each new schema should be as general as possible.
For example, instead of a schema for regional BOLD data, a schema for activity
timeseries represented in tabular format where the columns are regions is
preferred. (*Note: If you are not a siibra developer or currator, please contact
the siibra team to get help with the best way top implement a new schema.*)

II. Create the json
-------------------
Based on the examples and schema, create a json with

1. a unique and descriptive name
2. a valid data destination:

    * URL
    * If local, you can create a localhost with python and write the a url using
      your localhost (i.e. http://localhost:7001/filename.csv where the server
      is started in the folder filename.csv is located.)
    * Alternatively, a locat repository. (This is particularly useful when
      one would like to work with sensitive data.)

* If it provides images, make sure to write the correct provider and see if the
  image or mesh type is supported by siibra.
  (see ``siibra.volumes.volume.SUPPORTED_FORMATS``)
3. ``siibra.retrieval.requests.DECODERS`` lists all the file types that siibra
   can directly digest. Note that you can specify csv decoders in further detail,
   see connectivity configurations for examples.

III. Test the configuration
---------------------------

1. One can digest a preconfiguration by ``siibra`` via
   ``obj = siibra.from_json("path_to_json")``. This is a good first check to see
   if the object type is achieved (``print(type(obj))``).
2. Check the properties of ``obj``.
3. Fetch the data connected to ``obj`` based on the object type.

If there are any issues, trace back your steps and see if everything is in
order.

IV. Use/share your preconfiguration
-----------------------------------

You can now use this json to load your desired data as a siibra object and take
advantage its functionalities. You can also share this json with your colleagues
to collabrate.

At this stage, if you'd like to see your data in siibra-toolbox as default,
please contact the siibra team. (Most configurations are also automatically
added to ``siibra-explorer``.)


