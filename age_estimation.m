clc;clear;
%% init
addpath('tools');

%% Load data
% contact salah AT bekhouche.com to get this file
load data/fgnet.mat
db.path = 'C:\Projects\ESWA2017\data\fgnet';

%% Face processing
%{
mkdir('data/faces/');
for i=1:db.stats.images
    img = imread([db.path '\' db.images(i).folder '\' db.images(i).file]); % Read Image
    face = face_alignment(img,[0.5 1 1.75],[224 224],db.images(i).eyes); % Align Image
    if (size(face,3) == 3)
        face = rgb2gray(face); % RGB to Gray
    end
    imwrite(face,['data/faces/' db.images(i).file]); % Save Aligned Image
end
%}

%% Feature Extraction & Selection
%{
mkdir('data/features/');
options.level = 7;
% LPQ
options.descriptor = 'LPQ';
options.winSize = 7;
for i=1:db.stats.images
    img = imread(['data/faces/' db.images(i).file]); % Read Image
    feats_lpq(i,:) = PML(img,options);
end
% BSIF
options.descriptor = 'BSIF';
load ICAtextureFilters_9x9_8bit.mat
options.filter = ICAtextureFilters;
for i=1:db.stats.images
    img = imread(['data/faces/' db.images(i).file]); % Read Image
    feats_bsif(i,:) = PML(img,options);
end

for i=1:db.stats.images
    labels(i) = db.images(i).age;
    folds(i) = db.images(i).subject;
end

features = feats_lpq;
save(['data/features/age_pml_lpq_' num2str(options.level)],'features','labels','folds');
features = feats_bsif;
save(['data/features/age_pml_bsif_' num2str(options.level)],'features','labels','folds');
features = [feats_bsif feats_lpq];
save(['data/features/age_pml_bsif_lpq_' num2str(options.level)],'features','labels','folds');
%}

%% Facial age estimation
load('data/features/age_pml_bsif_lpq_2');
predicted_labels = zeros(1,numel(labels));
for i=1:db.stats.subjects
    test = folds == i;
    train = ~test;
    
    % features
    X_tr = features(train,:);
    X_ts = features(test,:);
    
    % Selection
    %{%
    ratio = 0.025;
    feats = 1500;
    [~,ranks] = FisherScore(X_tr,labels(train),ratio);
    X_tr = X_tr(:,ranks(1:feats));
    X_ts = X_ts(:,ranks(1:feats));
    %}
    
    % Estimate
    SVR = fitrsvm(X_tr,labels(train),'KernelFunction','gaussian','KernelScale','auto',...
    'Standardize',true);
    predicted_labels(test) = predict(SVR,X_ts);
    MAE = sum(abs(predicted_labels(test)  - labels(test)))/numel(labels(test));
    fprintf('Subject %d/%d MAE = %2.2f\n',i,db.stats.subjects,MAE);
end
MAE = sum(abs(predicted_labels  - labels))/numel(labels);
fprintf('MAE = %2.2f\n',MAE);