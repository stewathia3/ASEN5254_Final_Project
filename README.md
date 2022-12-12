###############################################################################
# Filename: README.md
# Author(s): Riana Gagnon, Sanjana Tewathia
#
# Final Project
# ASEN 5254, Fall 2022
###############################################################################

#------------------------------------------------------------------------------
# How to run our code:
#------------------------------------------------------------------------------

To run this final project code, all one has to do is call:

    python3 ConsecMotionPlanning.py 

in this directory.

Alternatively, press "run" in your IDE of choice (as long is it has a python 
extension) for this file. 

All of our code is available in this Github Repo: https://github.com/stewathia3/ASEN5254_Final_Project. 

If you would like to be added to this project, or if this link does not work,
feel free to email sanjana.tewathia@colorado.edu for access. 

#------------------------------------------------------------------------------

***DEPENDENCIES:

Make sure that you have the "numpy", "matplotlib", and "hpp-fcl" libraries 
installed, since our code will not run without these libraries. If you 
do not have those installed, you can get them with these commands in terminal:

    pip install numpy
    pip install pillow
    pip install matplotlib
    pip install hpp-fcl

Note that the "pillow" library must be installed in order to successfully 
install the "matplotlib" library.

#------------------------------------------------------------------------------

The scenario that runs is the one for Subtask 5, which incorporates the drone 
landing on the rover, and the achievement of consecutive tasks after.

As the RRT planner runs for each task, it may take a few tries to converge on
a solution, so the terminal will show this progress as the algorithm takes a 
while to run. An example of this terminal output is provided in Figure 1 of 
the project report.

The Matplotlib plots do take a while to open sometimes, so please be 
patient with it. Even if it prompts you with a pop-up asking to 
"Force-Quit" or "Wait", just click "Wait" since it usually does show the 
plot after some time.

