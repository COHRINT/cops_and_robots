function [ output_weights, output_means, output_covariances ] =...
    runnalls_gm_test()
load('data/from_python.mat')
% runnalls_gm_test(weights, means, covariances, max_num_mixands)

new_covariances = zeros(2,2, size(covariances, 1));
for i = 1:size(covariances, 1)
    new_covariances(1,:,i) = covariances(i,:,1);
    new_covariances(2,:,i) = covariances(i,:,2);
end

gm = gmdistribution(means, new_covariances, weights);
gm_runnals = runnallsjoingmm(gm, max_num_mixands);

weights = gm_runnals.ComponentProportion;
means = gm_runnals.mu;
covariances = gm_runnals.Sigma;
save('data/from_matlab.mat', 'weights', 'means', 'covariances')
end

