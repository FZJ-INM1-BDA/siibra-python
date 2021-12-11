.. _parcmaps:

Maps and templates in different reference spaces
------------------------------------------------

In order to actually work with the brain regions defined in a parcellation, we use their image or mesh representations defined in a particular reference space. This includes in particular brain reference templates, discrete parcellation maps, probabilistic parcellation maps, or region masks. Other than the concepts described before, these are spatial and not purely semantic objects. Therefore, they are always explicitely linked to a particular coordinate system, or brain reference space. ``siibra`` supports a range of different reference spaces, include the commonly used MNI space for the human brain but also the BigBrain histological space and the freesurfer surface. Therefore, many brain parcellations and regions have maps in different spaces linked to them.



