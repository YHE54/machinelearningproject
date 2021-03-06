%
% KNN MODEL INTERPRETATION !!!
%

% add relevant paths
clear; close all; clc;
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/')

% turn off warnings
% warning('off','all')

%% Choose which model to use

%WhichModel = 'Basic';
WhichModel = 'Reduced';
%WhichModel = 'Unprocessed';

if strcmp(WhichModel, 'Basic') == 1
load 'BasicModel.mat';
Features = BasicModel.Features;
Survival = BasicModel.Survival +3; %add 3 to ignore negative survival
Censored = BasicModel.Censored;

elseif strcmp(WhichModel, 'Reduced') == 1
load 'ReducedModel.mat';
Features = ReducedModel.Features;
Survival = ReducedModel.Survival +3; %add 3 to ignore negative survival
Censored = ReducedModel.Censored;

elseif strcmp(WhichModel, 'Unprocessed') == 1
load 'GBMLGG.Data.mat';
Survival = Survival +3; %add 3 to ignore negative survival
end

% remove NAN survival or censorship values
Features(:,isnan(Survival)==1) = [];
Censored(:,isnan(Survival)==1) = [];
Survival(:,isnan(Survival)==1) = [];

Features(:,isnan(Censored)==1) = [];
Survival(:,isnan(Censored)==1) = [];
Censored(:,isnan(Censored)==1) = [];

% removing mRNA features
% Features(399:end,:) = [];

[p,N] = size(Features);

% NEW!!! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Add NAN values at random to simulate missing data
pNaN = 0.75; %proportion of NAN values

NaN_Idx = randperm(N*p,N*p); 
NaN_Idx = NaN_Idx(1:pNaN * N*p);

Features(NaN_Idx) = nan;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[p,N] = size(Features);

%% Determine initial parameters

K_min = 20;
K_max = 50;
%K_min = 150; 
%K_max = 230;

%K_min = 15; K_max = 15;
%K_min = 30; K_max = 30;
%K_min = 180; K_max = 180;

Filters = 'None';
%Filters = 'Both'; %choose this if performing gradient descent on sigma

Beta_init = ones(length(Features(:,1)),1); %initial beta (shrinking factor for features)
sigma_init = 7;

Lambda = 1; %the less the higher penality on lack of common dimensions

% Parameters for gradient descent on beta
Gamma_Beta = 15; %learning rate
Pert_Beta = 5; %this controls how much to perturb beta to get a feeling for gradient
Conv_Thresh_Beta = 0.0001; %convergence threshold 

Gamma_sigma = 10; %learning rate
Pert_sigma = 0.1; %this controls how much to sigma beta to get a feeling for gradient
Conv_Thresh_sigma = 0.0005; %convergence threshold for sigma

Descent = 'None'; %fast
%Descent = 'Beta'; %slow, especially with more features
%Descent = 'sigma'; %slow, especially with more features

trial_No = 100; % no of times to shuffle

%%

C = zeros(trial_No,1);
NNvar = zeros(trial_No,398);

for trial = 1:trial_No

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

    %% Assign samples to PROTOTYPE set, validation set 
    %  The reason we call it "prototype set" rather than training set is 
    %  because there is no training involved. Simply, the patients in the 
    %  validation/testing set are matched to similar ones in the prototype
    %  ("database") set.
    
    K_cv = 2;
    Folds = ceil([1:N] / (N/K_cv));

    X_prototype = Features(:, Folds == 1);
    X_valid = Features(:, Folds == 2);

    Survival_prototype = Survival(:, Folds == 1);
    Survival_valid = Survival(:, Folds == 2);

    Censored_prototype = Censored(:, Folds == 1);
    Censored_valid = Censored(:, Folds == 2);

    % Convert outcome from survival to alive/dead status using time indicator
    t_min = min(Survival)-1;
    t_max = max(Survival);
    time = [t_min:1:t_max]';
    Alive_prototype = TimeIndicator(Survival_prototype,Censored_prototype,t_min,t_max);
    Alive_valid = TimeIndicator(Survival_valid,Censored_valid,t_min,t_max);
    
    %% Determine optimal model parameters using validation set
    
    % Determine optimal K
    K_star = 0;
    Accuracy_star = 0;
    
    for K = K_min:2:K_max
        
        clc
        trial
        K 
                      
        [Y_valid_hat,~] = KNN_Survival5(X_valid,X_prototype,Alive_prototype,K,Beta_init,Filters,sigma_init,Lambda);
        Y_valid_hat = sum(Y_valid_hat);
        Accuracy = cIndex2(Y_valid_hat,Survival_valid,Censored_valid);
        
        if Accuracy > Accuracy_star
            K_star = K;
            Accuracy_star = Accuracy;
        end
        
    end

    Beta_star = Beta_init;
    sigma_star = sigma_init;
    
    % Gradient descent on beta
%     if strcmp(Descent,'Beta') ==1
%         Beta_star = KNN_Survival_Decend2b(X_valid,X_prototype,Alive_prototype,Alive_valid,K_star,Beta_init,Filters,Gamma_Beta,Pert_Beta,Conv_Thresh_Beta,sigma_init);
%     elseif strcmp(Descent,'sigma') ==1    
%         sigma_star = KNN_Survival_Decend2a(X_valid,X_prototype,Alive_prototype,Alive_valid,K_star,Beta_init,Filters,Gamma_sigma,Pert_sigma,Conv_Thresh_sigma,sigma_init);      
%     end
    
    %% Determining accuracy
    
    [Alive_valid_hat, X_valid_NNvar] = KNN_Survival5(X_valid,X_prototype,Alive_prototype,K_star,Beta_star,Filters,sigma_star,Lambda);
    
    % c-index
    Survival_valid_hat = sum(Alive_valid_hat);
    C(trial,1) = cIndex2(Survival_valid_hat,Survival_valid,Censored_valid);
    
    NNvar_current = (sum(X_valid_NNvar,2))'; 
    
    % removing mRNA from interpretation
    NNvar_current(:,399:end) = [];
    
    % total NN variance over all patients for each feature
    NNvar(trial,:) = NNvar_current; 
    
    
end


% Z-score standardizing feature variance
NNvar_mean = mean(NNvar);
[NNvar_mean,~] = meshgrid(NNvar_mean,1:length(C));
NNvar_std = std(NNvar);
[NNvar_std,~] = meshgrid(NNvar_std,1:length(C));
NNvar_std(NNvar_std == 0) = 0.0000000000001;
NNvar_Zscored = (NNvar - NNvar_mean) ./ NNvar_std;

% correlate feature variance with c-index
FeatureCorr = corr(NNvar_Zscored,C);
% add feature index
FeatureCorr(:,2) = [1:length(NNvar(1,:))]';
% sort so that negative correlated features are on top
% i.e. features that vary least in nearest neighbours must've had a
% positive impact on prediction accuracy
FeatureCorr = sortrows(FeatureCorr,1);

% Getting names of important features
for i = 1:length(FeatureCorr(:,2))
    
    Important_features{i,1} = ReducedModel.Symbols{FeatureCorr(i,2),1};
    Important_features{i,2} = ReducedModel.SymbolTypes{FeatureCorr(i,2),1};
end