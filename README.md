# swayfreq - a toolkit for measuring the sway frequency of a vibrating object using video data
## Introduction
This toolkit enables one to measure the sway frequency of a vibrating object from video data. The code was developed for measuring the sway frequency of trees, which can be used to infer other properties of the trees, over greater spatial extents. However, the code generalizes nicely to other applications. 

Users can choose from 2 different video processing algorithms:
1. The Virtual Vision Sensor (VVS) algorithm based on the methods of Schumacher and Shariati (2013)
2. The Multilevel Binary Thresholding (MBT) algorithm based on the methods Ferrer et al. (2013)

A video processing pipeline has been developed so that users may can simply choose a video processing algorithm, configure important processing parameters, and conveniently compute the sway frequency. The pipeline was designed to be modular so that users can easily modify existing components of the workflow or optionally plug in their own custom functions without having to reimplement other components of the pipeline. 

## How to get started
### Running the code
1. Create and activagte an environment with the provided 'environment.yml' file. The pipeline uses a relatively standard scientific computing stack but additionally invokes opencv and ffmpeg. For example, for a conda environment:
   ```
   conda env create -f environment.yml
   ```
   and
   ```
   conda activate swayfreq
   ```
3. Clone the repo.
4. Install the package in developer mode after cd'ing to the newly cloned directory
   ```
   pip install . -e
   ```
6. Experiment with an existing notebook or implement your own.

### Sample notebooks
Several sample notebooks used to facilitate analysis for an accompanying manuscript have been included.
- manitou_manuscript.ipynb - a notebook that analyzes the sway of two trees in a stand of trees at the Manitou Experimental Forest, CO using VVS and accelerometer data.
- niwotridge_manuscript.ipynb - a notebook that analyzes the sway of a fir tree in Niwot Ridge, CO using VVS, MBT, and accelerometer data.
- snoqualmie_manuscript.ipynb - a notebook that illustrates using the video processing to investigate a snow unloading event.
- troutlake_manuscript.ipynb - a notebook that analyzes the sway of a tree in Trout Lake, Wisconsin using both VVS, MBT, and accelerometer data

### Analyzer objects
The VVSAnalyzer and MBTAnalyzer objects provide a convenient interfce for invoking the video processing algorithms. After first defining key parameters, the user can create a MBT analyzer object and step through each of the video processing pipeline (to examine intermediate output) or call `analyze()` to run the entire pipeline (see trout_manuscript.ipynb for example).

### Memory burden
Working with uncompressed video data often requires a large amount of RAM. Consider working with small regions of interest (ROIs) or on a machine with a suitable amount of RAM. See the hpc directory for example scripts used to process the entire frame on a HPC. 

## Video processing pipeline
The logic for both video processing algorithms has been abstracted into a modular pipeline with 4 abstract steps:
1. Translating a video to vibration signals (vid2vib_utils.py)
2. Estimating the power spectral density (PSD) of each vibration signal (spectra_utils.py)
3. Aggregating frequency content across all vibration signals (aggregate_utils.py)
4. Reporting the output peak frequency (plotting_utils.py)

Functions related to each abstract step are implemented in the util python files in parentheses above.

Each video processing algorithm then has an Analyzer object that combines functions from the util files to create an end-to-end algorithm. Each Analyzer object has one function corresponding to each abstract step. These functions serve as wrappers and enable users users to customize key elements of the video processing pipeline without modifying the pipeline itself. For key computation steps, users can invoke existing functions or define their own. For example, for step 2, users can choose from existing methods for estimating the power spectral density (see spectra_utils.py) or define their own.

Once processing parameters have been configured, the user can simiply call each step of the analyer object to process the video.

## References

Ferrer, B., Espinosa, J., Roig, A. B., Perez, J., & Mas, D. (2013). Vibration frequency 
    measurement using a local multithreshold technique. Optics Express, 21(22), 26198–26208. 
    https://doi.org/10.1364/OE.21.026198

Schumacher, T., & Shariati, A. (2013). Monitoring of Structures and Mechanical Systems Using 
    Virtual Visual Sensors for Video Analysis: Fundamental Concept and Proof of Feasibility. 
    Sensors, 13(12), Article 12. https://doi.org/10.3390/s131216551
