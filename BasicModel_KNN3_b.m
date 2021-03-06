%
% Assess performance of basic model using training/validation/testing
% approach with shuffling - KNN using the Alive_train 
%
% NOTE: Codes used here are a combination of original codes and codes
% provided in the "PerformanceExample.m" script provided by Dr Lee Cooper
% in CS-534 class
%
% add relevant paths
clear; close all; clc;
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/')

% turn off warnings
% warning('off','all')

%% Using basic model as provided by Dr Lee (PerformanceExample.m) for a starter

% read in data and extract minimal features
load 'BasicModel.mat';

Features = BasicModel.Features;
Survival = BasicModel.Survival +3; %add 3 to ignore negative survival
Censored = BasicModel.Censored;

[p,N] = size(Features);

%% Determine initial parameters

K_max = 50; %gradient descent takes care of optimal value for K in a continuous fashion

%Filters = 'Euclidian';
%Filters = 'Gaussian';
Filters = 'Both';
%Filters = 'None';

sigma_init = 3; % initial sigma of gaussian filter (lower values result in more emphasis on closes neighbours)
Beta_init = ones(length(Features(:,1)),1); %initial beta (shrinking factor for features)

%Parameters for gradient descent on sigma
Gamma_sigma = 10; %learning rate
Pert_sigma = 0.1; %this controls how much to perturb beta to get a feeling for gradient
Conv_Thresh_sigma = 0.0005; %convergence threshold for sigma

% Parameters for gradient descent on beta
Gamma_Beta = 15; %learning rate
Pert_Beta = 5; %this controls how much to perturb beta to get a feeling for gradient
Conv_Thresh_Beta = 0.0001; %convergence threshold 

%Descent = 'Sigma'; %faster but (very slightly) less optimal
Descent = 'Sigma+Beta'; %slow, especially with more features

trial_No = 1; % no of times to shuffle

%%

C = zeros(trial_No,1);
MSE = zeros(trial_No,1);

for trial = 1:trial_No

    clc
    trial
    
    %% Shuffle samples

    Idx_New = randperm(N,N);
    Features_New = zeros(size(Features));
    Survival_New = zeros(size(Survival));
    Censored_New = zeros(size(Censored));
    for i = 1:N
    Features_New(:,i) = Features(:,Idx_New(1,i));
    Survival_New(:,i) = Survival(:,Idx_New(1,i));
    Censored_New(:,i) = Censored(:,Idx_New(1,i));
    end
    Features = Features_New;
    Survival = Survival_New;
    Censored = Censored_New;

    %% Assign samples to PROTOTYPE set, validation set (for model selection) ... 
    %  and testing set (for model assessment):
    %  The reason we call it "prototype set" rather than training set is 
    %  because there is no training involved. Simply, the patients in the 
    %  validation/testing set are matched to similar ones in the prototype
    %  ("database") set.
    
    K_cv = 3;
    Folds = ceil([1:N] / (N/K_cv));

    X_prototype = Features(:, Folds == 1);
    X_valid = Features(:, Folds == 2);
    X_test = Features(:, Folds == 3);

    Survival_prototype = Survival(:, Folds == 1);
    Survival_valid = Survival(:, Folds == 2);
    Survival_test = Survival(:, Folds == 3);

    Censored_prototype = Censored(:, Folds == 1);
    Censored_valid = Censored(:, Folds == 2);
    Censored_test = Censored(:, Folds == 3);

    % Convert outcome from survival to alive/dead status using time indicator
    t_min = min(Survival)-1;
    t_max = max(Survival);
    time = [t_min:1:t_max]';
    Alive_prototype = TimeIndicator(Survival_prototype,Censored_prototype,t_min,t_max);
    Alive_valid = TimeIndicator(Survival_valid,Censored_valid,t_min,t_max);
    
    %% Determine optimal model parameters using validation set
    
    % Gradient descent on sigma
    if strcmp(Descent,'Sigma') ==1 || strcmp(Descent,'Sigma+Beta') ==1
        %sigma_star = KNN_Survival_Decend2a(X_valid,X_prototype,Alive_prototype,Alive_valid,K_max,Beta_init,Filters,Gamma_sigma,Pert_sigma,Conv_Thresh_sigma,sigma_init);
        [sigma_star,Progress_sigma] = KNN_Survival_Decend2a(X_valid,X_prototype,Alive_prototype,Alive_valid,K_max,Beta_init,Filters,Gamma_sigma,Pert_sigma,Conv_Thresh_sigma,sigma_init,'SaveProgress');
    else
        sigma_star = sigma_init;
    end
    % Gradient descent on beta
    if strcmp(Descent,'Sigma+Beta') ==1
        %Beta_star = KNN_Survival_Decend2b(X_valid,X_prototype,Alive_prototype,Alive_valid,K_max,Beta_init,Filters,Gamma_Beta,Pert_Beta,Conv_Thresh_Beta,sigma_star);
        [Beta_star,Progress_Beta] = KNN_Survival_Decend2b(X_valid,X_prototype,Alive_prototype,Alive_valid,K_max,Beta_init,Filters,Gamma_Beta,Pert_Beta,Conv_Thresh_Beta,sigma_star,'SaveProgress');
    else
        Beta_star = Beta_init;
    end
    
    %% Determining testing accuracy (c-index) using testing set
    
    Alive_test_hat = KNN_Survival3(X_test,X_prototype,Alive_prototype,K_max,Beta_star,Filters,sigma_star);
    Alive_test_hat = sum(Alive_test_hat);
    C(trial,1) = cIndex2(Alive_test_hat,Survival_test,Censored_test);
    % mean squared error
    MSE(trial,1) = mean((Alive_test_hat(Censored_test==0) - Survival_test(Censored_test==0)) .^ 2);
    
end
 