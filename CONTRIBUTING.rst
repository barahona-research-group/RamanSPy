========================
Contributing to RamanSPy
========================

There are many `Ways to contribute`_ to ``RamanSPy``. Check `How to contribute`_ for more information about how to do that.
  
All contributions are welcome and greatly appreciated.

  
Ways to contribute
----------------------
  
Reporting bugs
~~~~~~~~~~~~~~~~~
We recognize that we are not error-free. If you encounter a bug in our package, please inform us about the bug by creating an `issue on GitHub <https://github.com/barahona-research-group/RamanSPy/issues/new?assignees=&labels=bug&template=bug_report.md&title=>`_.

When reporting a bug, **please include the following information**:
  
- Mention the specific version of ``RamanSPy`` you are using.
- Provide a clear description of the bug, explaining what is happening and how it is affecting your use of the package.
- Include information about your environment that might be relevant, such as your operating system, Python version, and dependencies.
- Provide a step-by-step guide about how to reproduce the bug. This is crucial for identifying the source of the problem.
- Include any other relevant information like error messages, logs, or screenshots.

Your report can significantly help us in identifying and fixing the issue.

Fixing bugs
~~~~~~~~~~~~~~

You can also contribute to ``RamanSPy`` by fixing already reported bugs. You can find a list of reported bugs under `Issues <https://github.com/barahona-research-group/RamanSPy/issues>`_. Look for issues tagged with "bug".

To fix a bug:
  
- **Inform the community:** Comment on the issue you are interested in contributing to let others know you are working on it. This prevents duplicate efforts.  
- **Start implementing:** Begin working on fixing the bug by following the instructions in `How to contribute`_. 
  
If you encounter a bug that has not been reported yet, please report the bug before you start working on it (see `Reporting bugs`_). 


Adding new features
~~~~~~~~~~~~~~~~~~~~~~
Contributing by adding new features to ``RamanSPy`` is another excellent way to help. Look for issues tagged with "enhancement" in our `Issues section <https://github.com/barahona-research-group/RamanSPy/issues>`_.

If you have a new feature idea:

- **Create an issue:** First, `open a new issue on GitHub <https://github.com/barahona-research-group/RamanSPy/issues/new?assignees=&labels=enhancement&title=>`_ to discuss your idea.
- **Get feedback:** Ensure the feature aligns with the project's direction and hasn't been implemented or suggested before.
- **Start implementing:** After the discussion, follow the `How to contribute`_ guidelines to implement the feature.

Contributing to Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Good documentation is crucial. Help us by:

- **Improving existing documentation:** Enhance, clarify, or correct the existing documentation.
- **Adding missing documentation:** Identify areas lacking documentation and contribute by writing it.

Look for issues tagged with "documentation" for areas that need attention.
  
Open a `new documentation issue <https://github.com/barahona-research-group/RamanSPy/issues/new?assignees=&labels=documentation&title=>`_ if you spot problems or inconsistencies.

  
Writing and Updating Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests are essential for maintaining the quality of ``RamanSPy``. You can contribute by:

- **Writing new tests:** For existing code without sufficient tests.
- **Improving existing tests:** Enhance available tests for better coverage or clarity.


How to contribute
----------------------
Follow these to ensure that your contributions are effectively integrated into ``RamanSPy``.

1) Create an issue on GitHub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Before starting your work, create an issue on the ``RamanSPy`` GitHub `Issues page <https://github.com/barahona-research-group/RamanSPy/issues>`_. This helps us track and discuss contributions, ensuring no duplication of effort.

2) Fork repository
~~~~~~~~~~~~~~~~~~~~~
Visit the ``RamanSPy`` `GitHub repository <https://github.com/barahona-research-group/RamanSPy>`_ and click the "Fork" button. This creates a copy of the repository in your GitHub account.
  
3) Clone your fork locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Clone the forked repository to your local machine using the following command:

.. code-block:: bash
  
  git clone https://github.com/barahona-research-group/RamanSPy.git
  cd RamanSPy
  
4) Set up virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a virtual environment for the project to manage dependencies separately. Use the following commands:

.. code-block:: bash
  
  python -m venv venv
  source venv/bin/activate  # On Windows, use: venv\Scripts\activate
  pip install .
  
5) Create a new branch
~~~~~~~~~~~~~~~~~~~~~~~
Create a branch for your changes, preferably named after the issue number:

.. code-block:: bash
  
  git checkout -b issue-<number>-feature-or-bugfix
  
6) Commit and Push Changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Make your changes and commit them with clear, descriptive messages. Then, push the branch to your fork:

.. code-block:: bash
  
  git add .
  git commit -m "Describe your changes here"
  git push origin issue-<number>-feature-or-bugfix
    
7) Check documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ensure any changes or additions you've made are properly documented. Update README, docstrings, or the official documentation as needed. Also ensure that documentation is properly compiled.

8) Test
~~~~~~~~~~~~~~~~~~~~~~
Run existing tests to ensure your changes haven't broken anything, and write new tests if adding new features or fixing bugs.

9) Submit a pull request
~~~~~~~~~~~~~~~~~~~~~~~~~~
Go to your fork on GitHub and click “New pull request”. Compare your branch with the original ``RamanSPy`` repository's main branch. Review the changes, then create the pull request with a clear title and description.


Licensing and Contributions
----------------------------

Understanding the License
~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
``RamanSPy`` is released under the **BSD 3-Clause License**. This license governs how the software can be used and shared. Please familiarize yourself with its terms before contributing. You can find the full license text in the `LICENSE file <https://github.com/barahona-research-group/RamanSPy/blob/main/LICENSE>`_ in our repository.

Contributions Under the Same License
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By contributing to ``RamanSPy``, you agree that your contributions will be licensed under the same license as the project. This ensures consistency and legal clarity, allowing your contributions to benefit the community under the same open and permissive terms.

Intellectual Property
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Your contributions must be your own original work and not violate the intellectual property rights of others. If you use or incorporate work that isn't your own, it must be appropriately credited and conform to the legal requirements of the original work's license.

Questions and Clarifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you have any questions about the license or how it applies to your contributions, please don't hesitate to open an issue for discussion. We want to ensure that everyone is clear about the legal aspects of contributing to ``RamanSPy``.
