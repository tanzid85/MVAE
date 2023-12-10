% Load Data

joint_pos = obs_data(:, 1:16);
joint_velo = obs_data(:, 17:32);

% Concatenate the features
features = [joint_pos, joint_velo];

% Calculate Mean and Standard Deviation
mean_features = mean(features);
std_features = std(features);

% Check for zero standard deviation and set to 1 to avoid division by zero
std_features(std_features == 0) = 1;

% Normalize - Z-score normalization
normalized_features = (features - mean_features) ./ std_features;

% Display first few rows of normalized data
disp(normalized_features(1:5, :));

%%
% Assuming your matrix is named 'yourMatrix'
% Generate a logical matrix indicating where the elements start with '0.19'
logicalIndex = startsWith(string(combined_data), '0.765');

% Find the indices where the condition is true
[row, col] = find(logicalIndex);

% Check if any element starts with '0.19'
if ~isempty(row)
    disp('Found at least one element starting with ''0.765''.');
    disp(['Row: ', num2str(row(1)), ', Column: ', num2str(col(1))]);
else
    disp('No element starting with ''0.1905'' found.');
end

