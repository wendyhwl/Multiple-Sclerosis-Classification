# Multiple-Sclerosis-Classification
Distinguish between patients with advanced and early stages of [Multiple Sclerosis (MS)](https://en.wikipedia.org/wiki/Multiple_sclerosis) patients using [Linear Discriminate Analysis (LDA)](https://en.wikipedia.org/wiki/Linear_discriminant_analysis). Classification is made with only information from the patients' spinal cord in an MRI.

## Dataset
The [SC.mat](https://github.com/wendyhwl/Multiple-Sclerosis-Classification/tree/main/data) file contains MATLAB-formatted data with 13 features extracted from 30 patients with different stages of MS. The features are measurements of the spinal cord in an MRI. For example, the volume of the spinal cord, the average intensity within the cord, etc. The corresponding severity of MS for each patient is also
included. 

### Preprocessing
We first start by visualizing the original data. The below graph shows the distribution of all observations and severity.

<img src="/graphs/MS_original_data.png" width="400">

To separate the dataset, we convert the original dataset into binary data using an arbitrarily chosen cut-off point of 0.3. After the separation, we obtain an even distribution of early (marked as 0) and advanced patients (marked as 1). 

<img src="/graphs/MS_binary_data.png" width="400">

After that, we an observation matrix with all the featires and the class label at the end. The observation matrix consists of 30 rows (patients) and 14 columns (13 features and class label). We then create two matrices for early and advanced patients.

### Data Exploration
Looking at the "volume" feature (volume of the patients' spinal cord), we get the following distribution.

<img src="/graphs/MS_volume_histogram.png" width="400">

We will only use this feature to compute the covariance matrices for the observations and the vector (V) that maximally separates the data.

## Result
After plotting the ROC curve, we choose a threshold that obtains the highest sum of **True Nagative (TN) plus True Positive (1-FN)**. Under the chosen threshold, we can classify whether the patient falls in the "early" or "advanced" stage of MS.

<img src="/graphs/MS_roc.png" width="400"> <img src="/graphs/MS_threshold.png" width="400">


## Acknowledgements
- Assignment designed by CMPT 340 instructor Ghassan Hamarneh from **Simon Fraser University**
- [SFU Academic Integrity](http://www.sfu.ca/students/academicintegrity.html)
