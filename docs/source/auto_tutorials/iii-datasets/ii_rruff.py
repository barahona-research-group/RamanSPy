"""
Loading the RRUFF dataset
--------------------------------------

In this tutorial, we will see how to load the :ref:`RRUFF data` using `RamanSPy`.

"""
import ramanspy

# %%
# We can use `RamanSPy` to easily downloaded a specific dataset from RRUFF database available on the Internet by simply
# indicating the name of the dataset to load within the :meth:`ramanspy.datasets.rruff` method.
ramanspy.datasets.rruff('fair_oriented')

# %%
# If the dataset of interest has already been downloaded on your machine, you can load it by specifying the folder it is
# present at and setting ``download=False``.
ramanspy.datasets.rruff('<PATH>', download=False)
