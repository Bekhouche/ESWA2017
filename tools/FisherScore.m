function [weights,ranks] = FisherScore(features,labels,ratio)

numMax = 1e2;
labels = round(labels*ratio);
ages = unique(labels);
numAges = numel(ages);
numFeats = size(features,2);

for i = 1:numAges
    ids{i} = find(labels == ages(i));
    nums(i) = numel(ids{i});
end
weights = zeros(numFeats,1);
for i=1:numFeats
    mean_all = mean(features(:,i));
    weight_mean = 0;
    weight_var = 0;
    for j=1:numAges
        mean_one = mean(features(ids{j},i));
        var_one = var(features(ids{j},i));
        weight_mean = weight_mean + nums(j) * (mean_one - mean_all)^2;
        weight_var = weight_var + nums(j) * var_one;
    end
    
    if (weight_mean == 0)
        weights(i) = 0;
    else
        if (weight_var == 0)
            weights(i) = numMax;
        else
            weights(i) = weight_mean/weight_var;
        end
    end
    
end
[~, ranks] = sort(weights, 'descend');
end