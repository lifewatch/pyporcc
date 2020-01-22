%% Select clicks with the frequencies of interest
load('LogModel1a.mat'); % C:\MEL\PhD\Project\ClickTrain\Classifier\Modelling
load('click_model.mat');

%% Logistic regression model
id = [LogModel1a.id]';
P = [LogModel1a.P]'; % will be my y (dependent variable)
Q = [LogModel1a.Q]'; % one of the predictor variable
duration = [LogModel1a.duration]'; % my other predictor variable
ratio = [LogModel1a.ratio]';
XC = [LogModel1a.XC]';
CF = [LogModel1a.CF]';
BW = [LogModel1a.BW]';

VariableT = table(Q, duration, ratio, XC, CF, BW, P, id);
ResultsT = table(Q, duration, ratio, XC, CF, BW);
clear BW CF duration P Q ratio XC id
ResultsT([65:end],:) = [];
for j = 1:1440
    ResultsT.Q(j) = 0;
    ResultsT.duration(j) = 0;
    ResultsT.ratio(j) = 0;
    ResultsT.XC(j) = 0;
    ResultsT.CF(j) = 0;
    ResultsT.BW(j) = 0;
    if j < 721
        ResultsT.Model(j) = 1;
    else
        ResultsT.Model(j) = 2;
    end
    ResultsT.AIC(j) = 0;
    ResultsT.id(j) = j;
end

m = 0; % row number, until reaching 720 
for j = 1:6 % 6 variables
  clear myAIC myGLM ModX ModY Ncol Nrow VarComb X
  VarComb = combnk(1:6,j);
  [Nrow, ~] = size(VarComb);
     for k = 1:Nrow % within the VarComb to extract the combinations
          m = m+1;
          ModY = VariableT(:,7);
          X = VarComb(k,:);
          ModX = VariableT(:, X); 
          ModY = table2array(ModY);
          ModX = table2array(ModX);
          myGLM = fitglm(ModX, ModY, 'Distribution', 'binomial');
          myAIC = myGLM.ModelCriterion.AIC; % 28.42 - Q, duration
          ResultsT.AIC(m) = myAIC;
         for h = 1:j
           col = X(h);
           ResultsT{m,col} = 1;  
         end 
     end 
end 

ResultsT = sortrows(ResultsT,'AIC','ascend');
for i= 1:63
  ResultsT.Diff(i) = ResultsT.AIC(i) - ResultsT.AIC(1);   
end
save('ResultsT.mat', 'ResultsT');


%% apply Prob to other file
%myGLM8 =
%Generalized linear regression model:
%    logit(y) ~ 1 + x1 + x2
%    Distribution = Binomial
%
%Estimated Coefficients:
%                   Estimate        SE        tStat      pValue
%                   _________    ________    _______    _________
%
%    (Intercept)      -13.355       3.136    -4.2587    2.056e-05
%    x1                 1.578     0.52043     3.0321    0.0024285
%    x2             -0.040513    0.015331    -2.6425    0.0082292
%
%
%5000 observations, 4997 error degrees of freedom
%Dispersion: 1
%Chi^2-statistic vs. constant model: 3.23e+03, p-value = 0

[logitCoef,~] = glmfit(X8,y8,'binomial','logit');
% [-13.355355867717691;1.578011650012690;-0.040513248546648]
save('logitCoef.mat', 'logitCoef');
