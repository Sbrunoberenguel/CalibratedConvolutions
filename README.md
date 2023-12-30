# Calibrated Convolutions

The presented code belongs to the investigation from the Oral presentation at the BMVC23 Conference: [Convolution kernel adaptation to calibrated fisheye](https://papers.bmvc2023.org/0721.pdf). <p>
This presentation is also in [video](https://bmvc2022.mpi-inf.mpg.de/BMVC2023/0721_video.mp4).

We provide 2 implementations. 

### offsetCalcFE.py
This code generates a PyTorch dictionary where offsets for deformable convolutions are computed for a given fisheye calibration. The calibration parameters used are acording the [Kannala-Brandt's camera model](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1642666). This program is quite rigid, since the dictionary has to be created prior the use with a neural network for fixed image resolution and convolution parameters. However, onces the dictionary is created the inference time is similar to any other convolution.

### FEConv_online.py
This code generates de offsets for the deformable convolution on-line. That means, as the feature map gets to the convolution, it computed and applies the calibrated deformation. This method is more flexible, but way more slower (it is a future work to create a faster implementation of this code). As in the previous programs, the calibration parameters used are acording the [Kannala-Brandt's camera model](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1642666). 


# Note from the authors
This code has not been thoroughly tested, which means it may have some bugs. **Please use with caution.**

The authors and developers of this code want the best for the users and have gone to great lengths to make it easy to use and accessible. 
Be nice to them and enjoy their work.

If any problem may appear, do not hesitate and we will do our best to solve it (at least we will try).


# License
This software is under GNU General Public License Version 3 (GPLv3), please see GNU License

For commercial purposes, please contact the authors: 
- [Bruno Berenguel-Baeta](https://github.com/SBrunoberenguel) (berenguel@unizar.es) 
- [Maria Santos-Villafranca](https://github.com/Maria-SanVil) (m.santos@unizar.es)
