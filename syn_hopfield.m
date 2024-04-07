load clean_dataset.mat;
load noise_dataset.mat;
pattern0=resize(pattern0);
pattern1=resize(pattern1);
pattern2=resize(pattern2);
pattern3=resize(pattern3);
pattern4=resize(pattern4);
pattern6=resize(pattern6);
patternsqaure=resize(patternsqaure);
pattern9=resize(pattern9);
patterns = [pattern0; pattern1; pattern2; pattern3; pattern4; pattern6; patternsqaure; pattern9];
[numPatterns, inputNeurons] = size(patterns);
W = zeros(inputNeurons, inputNeurons);
%train
for i = 1:inputNeurons
    for j = 1:inputNeurons
        if i ~= j
            for s = 1:numPatterns
                W(i,j) = W(i,j) + patterns(s,i) * patterns(s,j);
            end
        end
    end
end

%validate
noise0 = resize(noise0);
noise1 = resize(noise1);
noise2 = resize(noise2);
noise3 = resize(noise3);
noise4 = resize(noise4);
noise6 = resize(noise6);
noisesquare = resize(noisesqaure);
noise9 = resize(noise9);
noise_patterns = [noise0; noise1; noise2; noise3; noise4; noise6; noisesquare; noise9];


initialPattern = noise6;
recallPattern = initialPattern;
numIterations = 3;
E = zeros(1, numIterations);
drawPattern(initialPattern, 'Initial State');
for t = 1:numIterations
    updates = sign(W * recallPattern.');
    updates(updates == 0) = -1;
    recallPattern = updates.';
    E(t) = energyFunction(W, recallPattern);
    drawPattern(recallPattern, ['Recall Iteration ' num2str(t)]);
end
drawPattern(recallPattern, 'Final Recalled State');
figure;
plot(1:numIterations, E, 'LineWidth', 2);
xlabel('Iteration');
ylabel('Energy');
title('Energy vs. Iteration');
grid on;

function E = energyFunction(W, pattern)
    E = -0.5 * sum(sum(W .* (pattern' * pattern)));
end
function drawPattern(pattern, titleText)
    figure;
    imagesc(reshape(pattern, 12, 10)); 
    colormap(flipud(gray));
    title(titleText);
    axis square off;
end
function rowVector = resize(pattern)
    [rows, cols] = size(pattern);
    rowVector = reshape(pattern, 1, rows * cols);
end