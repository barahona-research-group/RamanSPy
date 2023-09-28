:orphan:

Tutorials
###############

In this section, different code snippets are provided to illustrate how to use some of the main features of `raman`,
including data loading, preprocessing and analysis, and visualisation.



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    </div>

Data containers
***************



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The backbone of RamanSPy&#x27;s data management core is the ramanspy.SpectralContainer class. It ser...">

.. only:: html

  .. image:: /auto_tutorials/i-classes/images/thumb/sphx_glr_plot_i_generic_container_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_i-classes_plot_i_generic_container.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Storing generic data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The management of single Raman spectra in RamanSPy is guided through the ramanspy.Spectrum clas...">

.. only:: html

  .. image:: /auto_tutorials/i-classes/images/thumb/sphx_glr_plot_ii_spectrum_container_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_i-classes_plot_ii_spectrum_container.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Storing spectra</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The management of imaging Raman spectroscopic data in RamanSPy is guided through the ramanspy.S...">

.. only:: html

  .. image:: /auto_tutorials/i-classes/images/thumb/sphx_glr_plot_iii_image_container_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_i-classes_plot_iii_image_container.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Storing imaging data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The management of volumetric Raman spectroscopic data in RamanSPy is guided through the ramansp...">

.. only:: html

  .. image:: /auto_tutorials/i-classes/images/thumb/sphx_glr_plot_iv_volume_container_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_i-classes_plot_iv_volume_container.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Storing volumetric data</div>
    </div>


.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_tutorials/i-classes/plot_i_generic_container
   /auto_tutorials/i-classes/plot_ii_spectrum_container
   /auto_tutorials/i-classes/plot_iii_image_container
   /auto_tutorials/i-classes/plot_iv_volume_container

Data loading
*************



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Users can use RamanSPy to load MATLAB files as exported from WITec&#x27;s Suite FIVE software.">

.. only:: html

  .. image:: /auto_tutorials/ii-instrumental/images/thumb/sphx_glr_i_witec_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_ii-instrumental_i_witec.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Loading WITec data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Users can use RamanSPy to load .wdf files as exported from Renishaw&#x27;s WiRE software. This can b...">

.. only:: html

  .. image:: /auto_tutorials/ii-instrumental/images/thumb/sphx_glr_ii_renishaw_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_ii-instrumental_ii_renishaw.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Loading Renishaw data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Users can use RamanSPy to load single spectra .txt files acquired using Ocean Insight Raman ins...">

.. only:: html

  .. image:: /auto_tutorials/ii-instrumental/images/thumb/sphx_glr_iii_ocean_insight_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_ii-instrumental_iii_ocean_insight.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Loading Ocean Insight data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Users can use RamanSPy to load other data files, too. To do so, one simply has to parse the fil...">

.. only:: html

  .. image:: /auto_tutorials/ii-instrumental/images/thumb/sphx_glr_iv_other_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_ii-instrumental_iv_other.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Loading other data</div>
    </div>


.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_tutorials/ii-instrumental/i_witec
   /auto_tutorials/ii-instrumental/ii_renishaw
   /auto_tutorials/ii-instrumental/iii_ocean_insight
   /auto_tutorials/ii-instrumental/iv_other

Datasets and metrics
*********************



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will see how to load the RRUFF data using RamanSPy.">

.. only:: html

  .. image:: /auto_tutorials/iii-datasets/images/thumb/sphx_glr_ii_rruff_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_iii-datasets_ii_rruff.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Loading the RRUFF dataset</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will see how to load the Bacteria data available in RamanSPy.">

.. only:: html

  .. image:: /auto_tutorials/iii-datasets/images/thumb/sphx_glr_plot_i_bacteria_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_iii-datasets_plot_i_bacteria.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Loading the Bacteria dataset</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will see how to access and use the built-in metrics available in RamanSPy.">

.. only:: html

  .. image:: /auto_tutorials/iii-datasets/images/thumb/sphx_glr_plot_ii_metrics_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_iii-datasets_plot_ii_metrics.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Using built-in metrics</div>
    </div>


.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_tutorials/iii-datasets/ii_rruff
   /auto_tutorials/iii-datasets/plot_i_bacteria
   /auto_tutorials/iii-datasets/plot_ii_metrics

Data visualisation
************************



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="RamanSPy provides a broad selection of visualisation tools for the visualisation of Raman spect...">

.. only:: html

  .. image:: /auto_tutorials/iv-viz/images/thumb/sphx_glr_plot_i_spectra_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_iv-viz_plot_i_spectra.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Visualising spectra</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="We can visualise the peaks in a spectrum by using the ramanspy.plot.peaks method.">

.. only:: html

  .. image:: /auto_tutorials/iv-viz/images/thumb/sphx_glr_plot_ii_peaks_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_iv-viz_plot_ii_peaks.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Visualising peaks</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sometimes, we have plenty of spectra we wish to visualise. To appropriately do that, it is more...">

.. only:: html

  .. image:: /auto_tutorials/iv-viz/images/thumb/sphx_glr_plot_ii_spectra_mean_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_iv-viz_plot_ii_spectra_mean.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Visualising spectral distributions</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="RamanSPy allows the visualisation of Raman imaging data. Visualising imaging data can be achiev...">

.. only:: html

  .. image:: /auto_tutorials/iv-viz/images/thumb/sphx_glr_plot_iii_image_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_iv-viz_plot_iii_image.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Visualising imaging data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="RamanSPy aids the visualisation of volumetric Raman spectroscopic data. This can be done by usi...">

.. only:: html

  .. image:: /auto_tutorials/iv-viz/images/thumb/sphx_glr_plot_iv_volume_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_iv-viz_plot_iv_volume.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Visualising volumetric data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One of the data visualisation tools RamanSPy offers is the ramanspy.plot.peak_dist - a method i...">

.. only:: html

  .. image:: /auto_tutorials/iv-viz/images/thumb/sphx_glr_plot_v_peak_dist_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_iv-viz_plot_v_peak_dist.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Visualising peak distributions</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="RamanSPy&#x27;s plotting methods are built on top of matplotlib and so inherit most of matplotlib&#x27;s ...">

.. only:: html

  .. image:: /auto_tutorials/iv-viz/images/thumb/sphx_glr_plot_vi_customisation_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_iv-viz_plot_vi_customisation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Customising plots</div>
    </div>


.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_tutorials/iv-viz/plot_i_spectra
   /auto_tutorials/iv-viz/plot_ii_peaks
   /auto_tutorials/iv-viz/plot_ii_spectra_mean
   /auto_tutorials/iv-viz/plot_iii_image
   /auto_tutorials/iv-viz/plot_iv_volume
   /auto_tutorials/iv-viz/plot_v_peak_dist
   /auto_tutorials/iv-viz/plot_vi_customisation

Preprocessing
****************



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="RamanSPy provides a collection of various preprocessing methods, which users can directly acces...">

.. only:: html

  .. image:: /auto_tutorials/v-preprocessing/images/thumb/sphx_glr_plot_i_predefined_methods_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_v-preprocessing_plot_i_predefined_methods.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Built-in methods</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Users can use RamanSPy to also define their own preprocessing methods, which can then be direct...">

.. only:: html

  .. image:: /auto_tutorials/v-preprocessing/images/thumb/sphx_glr_plot_ii_custom_method_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_v-preprocessing_plot_ii_custom_method.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Custom methods</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="RamanSPy makes the construction and execution of diverse preprocessing pipelines significantly ...">

.. only:: html

  .. image:: /auto_tutorials/v-preprocessing/images/thumb/sphx_glr_plot_iii_custom_pipeline_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_v-preprocessing_plot_iii_custom_pipeline.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Custom pipelines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="To further ease the preprocessing workflow, RamanSPy provides a selection of established prepro...">

.. only:: html

  .. image:: /auto_tutorials/v-preprocessing/images/thumb/sphx_glr_plot_iv_predefined_pipeline_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_v-preprocessing_plot_iv_predefined_pipeline.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Built-in protocols</div>
    </div>


.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_tutorials/v-preprocessing/plot_i_predefined_methods
   /auto_tutorials/v-preprocessing/plot_ii_custom_method
   /auto_tutorials/v-preprocessing/plot_iii_custom_pipeline
   /auto_tutorials/v-preprocessing/plot_iv_predefined_pipeline

Analysis
************



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we will use RamanSPy to perform Principal Component Analysis (PCA) to decompos...">

.. only:: html

  .. image:: /auto_tutorials/vi-analysis/images/thumb/sphx_glr_plot_i_decomposition_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_vi-analysis_plot_i_decomposition.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Built-in decomposition methods</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Below, we will use RamanSPy&#x27;s built-in clustering methods to perform KMeans clustering and clus...">

.. only:: html

  .. image:: /auto_tutorials/vi-analysis/images/thumb/sphx_glr_plot_ii_kmeans_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_vi-analysis_plot_ii_kmeans.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Built-in clustering methods</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this tutorial, we will use the RamanSPy&#x27;s built-in methods for spectral unmixing to perform ...">

.. only:: html

  .. image:: /auto_tutorials/vi-analysis/images/thumb/sphx_glr_plot_iii_unmixing_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_vi-analysis_plot_iii_unmixing.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Built-in unmixing methods</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we will showcase RamanSPy&#x27;s integrability by integrating a Support Vector Mach...">

.. only:: html

  .. image:: /auto_tutorials/vi-analysis/images/thumb/sphx_glr_plot_iv_integrative_svm_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_vi-analysis_plot_iv_integrative_svm.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Integrative analysis: Support Vector Machine (SVM) classification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we will showcase RamanSPy&#x27;s integrability by integrating a Neural Network (NN)...">

.. only:: html

  .. image:: /auto_tutorials/vi-analysis/images/thumb/sphx_glr_plot_v_integrative_nn_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_vi-analysis_plot_v_integrative_nn.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Integrative analysis: Neural Network (NN) classification</div>
    </div>


.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_tutorials/vi-analysis/plot_i_decomposition
   /auto_tutorials/vi-analysis/plot_ii_kmeans
   /auto_tutorials/vi-analysis/plot_iii_unmixing
   /auto_tutorials/vi-analysis/plot_iv_integrative_svm
   /auto_tutorials/vi-analysis/plot_v_integrative_nn

Synthetic data generation
**************************



.. raw:: html

    <div class="sphx-glr-thumbnails">


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In this example, we will use RamanSPy to generate synthetic spectra.">

.. only:: html

  .. image:: /auto_tutorials/vii-synth/images/thumb/sphx_glr_plot_i_endmembers_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_vii-synth_plot_i_endmembers.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Generate synthetic spectra</div>
    </div>


.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_tutorials/vii-synth/plot_i_endmembers


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_tutorials_python.zip </auto_tutorials/auto_tutorials_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_tutorials_jupyter.zip </auto_tutorials/auto_tutorials_jupyter.zip>`
