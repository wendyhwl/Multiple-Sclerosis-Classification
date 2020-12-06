%% Classification using Linear Discriminate Analysis (LDA)
% CMPT 340 
%
% Examine data from patients with Multiple Sclerosis (MS). 
% Specifically, we are going to try to distinguish between patients with 
% advanced stages of MS vs earlier stages of MS. 
% We are going to make this distinction (classification) using ONLY
% information from the patients' spinal cord in an MRI.
%
%
% *Data files:*
% Extracted features from 30 patients with different stages of
% MS. The features are measurements of the spinal cord in an MRI. For
% example, the volume of the spinal cord, the average intensity within the
% cord, etc. The corresponding severity of MS for each patient is also
% included. 


% Clean-up
clear; close all; clc

%% Load the MS dataset.
SC = load('data/SC');
SC = SC.SC;

% original data.
targets = SC.target;
figure(1)
plot(targets,'.');
xlabel('observations (patients)'); ylabel('MS severity (lower is more severe)');
size(targets)
title('Visualization of the original data')
%%

% Convert to our binary class data of "early" vs "advanced" stage of MS.
C = zeros(size(targets,1),1);

% A cut-off of 0.3 is arbitrarily chosen so we have equal number of early
% and advanced patients.
C(targets<0.3) = 1;

% Examine the binary data.
% predict if a patient should be 0 (early) or advanced (1).
figure(2)
plot(C,'.')
axis([0,30,-1,2]); xlabel('observations (patients)'); ylabel('severity (0 is early, 1 is advanced)');
title('Examine the binary data')
%%

% Make an observation matrix with all our features and the class label at the end.
obs = [...
    SC.mCordPx_mn, SC.mCordPx_std, SC.mDistMax, SC.mDistMn, ...
    SC.mDistMnMax, SC.mEcc, SC.mMajAx, SC.mMinAx, ...
    SC.mPerMax, SC.mPerMn, SC.mPerStd, SC.mPerMin, ... % based on the the perimeter of the cord.
    SC.volume./20, ... % volume of the cord. We divide the volume by 20 since the other features were averaged over 20 slices.
    C % the class label.
    ];

% Our observation matrix consists of:
% 30 rows (observations or patients),
% with 13 features for each patient,
% and the class label (14 columns).
size(obs)


%% Create two matrices for early and advanced patients
% Create two separate observation matrices where one contains only the
% patients with the "early" (0) class labels, and the other contains
% patients with only the "advanced" (1) class labels.

obsEarly = [];
obsAd = [];

for k = 1:30
    if obs(k,14) == 1
        obsAd = [obsAd ; obs(k,1:13)]
    else 
        obsEarly = [obsEarly ; obs(k,1:13)]
    end
end

%--- check --------------------
display(sprintf('obsEarly size: %i', isequal(size(obsEarly), [15,13])))
display(sprintf('obsAd size: %i', isequal(size(obsAd), [15,13])))
display(sprintf('obsEarly vals: %i', isequal(round(obsEarly(2,3:8)*1000), [  2983        1519         509         864        5508        2746])))
display(sprintf('obsAd vals: %i', isequal(round(obsAd(5,8:end)*1000), [  2818        5734        3840         938        2318       55724
])))
%------------------------------

%% Explore separating the data based on 1 feature (1D) 
% Using the last feature, which should correspond to the "volume"
% features. This is the volume of the patients' spinal cord.

% Plot the "volume" feature from the "early" observation matrix in blue, and the "volume" feature from the "advanced"
% matrix in red. 

figure(3);
plot(obsEarly(:,13),'Color','b','DisplayName','early obs')
hold on
plot(obsAd(:,13),'Color','r','DisplayName','advanced obs')
hold off
title('Volumn')
legend('show','Location','northwest')

figure(4)
histogram(obsEarly(:,13),'BinWidth',3,'FaceColor','b','DisplayName','early obs')
hold on
histogram(obsAd(:,13),'BinWidth',3,'FaceColor','r','DisplayName','advanced obs')
hold off
title('Volumn in histogram')
legend('show','Location','northwest')



%% Compute the mean features observation matrices 
% Compute the mean features (_meanEarly_, _meanAd_) for the early and advanced observation
% matrices.

meanEarly = [];
meanAd = [];

for k=1:13
    meanEarly = [meanEarly, mean(obsEarly(:,k))];
    meanAd = [meanAd, mean(obsAd(:,k))];
end

%----- check -----
display(sprintf('size meanEarly:%i',isequal(size(meanEarly),[  1    13])));
display(['meanAd vals:', num2str(round(meanAd(2:7)*1000))]);

%-----------------


%% Compute the covariance matrices for each observation matrix
% Compute two covariance matrices (_covEarly_, _covAd_) for the early and 
% advanced observations.

covEarly = cov(obsEarly);
covAd = cov(obsAd);

%----- check -----
display(['covEarly vals:', num2str(round(covEarly(1:5)*10000))]);
display(sprintf('size covAd:%i',isequal(size(covAd),[  13    13])));
%-----------------


%% Compute the vector (V) that maximally separates the data

V = inv(covEarly + covAd)*(meanEarly-meanAd)';

%----- check -----
display(sprintf('size V:%i',isequal(size(V),[  13    1])));
display(['V vals:',num2str(round(V(2:7)*1000)')]);
%-----------------


%% Project observation matrices onto 1D using V
% Using the computed v, project each observation matrix onto a 1D line.

obsEarly_1D = obsEarly * V;
obsAd_1D = obsAd * V;

%----- check -----
display(sprintf('size obsEarly_1D:%i',isequal( size(obsEarly_1D),[  15    1])));
display(['obsAd_1D vals:', num2str(round(obsAd_1D(7:12)*1000)')]);
%-----------------


%% Fit a Gaussian distribution to the 1D projection
% First estimate the mean and standard deviation of the 1D early and advanced projections.

meanEarly1d = mean(obsEarly_1D);
stdEarly1d = std(obsEarly_1D);
meanAd1d = mean(obsAd_1D);
stdAd1d = std(obsAd_1D);

%----- check -----
display(['mEarly1d vals:',num2str(round(meanEarly1d*10000))]);
display(['sAd1d vals:', num2str(round(stdAd1d*10000))]);
%-----------------


% Create some variables to plot the Gaussian. 
x = -8:0.01:5;

figure(5)

% use the normal distribution.
earlyY = 1/(stdEarly1d* sqrt(2*pi)) * exp(- (((x-meanEarly1d).^2)./(2*stdEarly1d^2)));
h1 = plot(x,earlyY,'.b');

%% Fit a Gaussian distribution for the advanced observations
% Examine the above equation for the normal distrubtion used to compute
% _earlyY_. In a similar fashion, plot the normal distrubtion for the 
% advanced observations using the mean and std dev computed for 
% the 1D advanced projection earlier.

figure(5)
adY = 1/(stdAd1d* sqrt(2*pi)) * exp(- (((x-meanAd1d).^2)./(2*stdAd1d^2)));
h1 = plot(x,adY,'.b');
title('Distribution for Advanced Obs.')

%% Decide on set of thresholds to make the ROC curve.
% We want to get a bunch of thresholds to sample the data from.
% We have already seen this type of practices from Activity 08

if meanEarly1d <= meanAd1d
    T= linspace(meanEarly1d-5,meanAd1d+5,1000);
else
    T= linspace(meanAd1d-5,meanEarly1d+5,1000);
end


%% Reveiver Operating Characteristics [1]
% Compute the True Negative (TN) and False Negative (FN) values for the ROC curve.
% Compute values for each threshold (T) value.
% i.e. for every element in T, we calculate an element in TN and an element
% in FN.
%
% To compute the TN and FN, we will assume that the Gaussain that we computed
% earlier models the distribution.
%
% We want to find the area under the Gaussian distrubution up until the
% threshold point.


TN = [];
FN = [];

for k=1:1000
    FN = [FN , normcdf(T(1,k), meanEarly1d, stdEarly1d)];
    TN = [TN , normcdf(T(1,k), meanAd1d, stdAd1d)];
end

%----- check ----
display(sprintf('size TN:%i', isequal(size(TN), size(T))));
%----------------

%% Plot the ROC curve

figure(6)

plot(FN,TN,'Marker',".");
xlabel('FN (%)');
ylabel('TN (%)');
axis([0 1 0 1]);
title('ROC Curve');

%% Choose a threshold 
% Choose this threshold based on where we get the highest sum of 
% "True Negative plus True Positive" 
%
% Note that "True Positive" is (1-FN).
%
% Get the index (_threshIdx_) into T that corresponds to this threshold.


max = 0;
threshIdx = 0;

% compute sum and get max sum
for k=1:1000
    sum = TN(1,k) + (1-FN(1,k));
    if sum > max
        max = sum;
        threshIdx = k;
    end
end

%----- check ----
display(sprintf('theshIdx:%i', round(T(threshIdx)*10000)));
%----------------


%% Visualize the threshold.
figure(6)
hold on; % Note we are adding to the figure you created above.
plot(FN(threshIdx), TN(threshIdx),'*r','MarkerSize',20);
title('Visualization of ROC curve and chosen threshold')
% Thus T(threshIdx) is our threshold we want to use to separate our 1D data.

% This is what T(threshIdx) looks like on the 1D plot.
figure(7)
h61 = plot(x,earlyY,'.b');
hold on;
h62 = plot(x, adY,'.r');
h63 = plot(T(threshIdx),(0:0.001:0.5),'.k');
legend([h61, h62, h63(1)], 'early', 'advanced', 'chosen threshold')
title('Show the threshold on the Gaussian distributions')

%% Classify the 1D projected data 
% we have chosen our threshold (T(threshIdx)) to separate early from advanced
% patients, we can go back our 1D data. Use the 1D early and advanced
% observation matrices to chold on;lassify if the patient is in the early or
% advanced stages of MS. Use our computed threshold to decide if the
% patient is "early" or "advanced".

correctEarly = 0;
correctAdvanced = 0;

for k=1:15
    if obsEarly_1D(k,1) > T(threshIdx)
        correctEarly = correctEarly + 1;
    end
    if obsAd_1D(k,1) < T(threshIdx)
        correctAdvanced = correctAdvanced + 1;
    end
end

%----- output ----
display(sprintf('Correct early:%i/15', correctEarly));
display(sprintf('Correct advanced:%i/15', correctAdvanced));
%-----------------
