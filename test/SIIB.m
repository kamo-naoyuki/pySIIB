function [I] = SIIB(x, y, fs_signal, knn_flag, execpath)
% Speech intelligibility in bits (SIIB)
%
%--------------------------------------------------------------------------
% Copyright 2017: Steven Van Kuyk.
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%--------------------------------------------------------------------------
%
% Contact: steven.jvk@gmail.com
%
% References:
%   [1] S. Van Kuyk, W. B. Kleijn, R. C. Hendriks, 'An instrumental
%       intelligibility metric based on information theory', 2017
%   [2] S. Van Kuyk, W. B. Kleijn, R. C. Hendriks, 'An evaluation of
%       intrusive instrumental intelligibility metrics', 2017
%
% x: samples of a clean acoustic speech signal
% y: samples of a distorted acoustic speech signal
% fs_signal: sampling rate of x and y
% I: estimate of the mutual information rate between x and y in b/s
%
% IMPORTANT: SIIB assumes that x any y are time-aligned.
%
% IMPORTANT: SIIB may not be reliable for stimuli with short durations
% (e.g., less than 20 seconds). Note that longer stimuli can be created by
% concatenating short stimuli together.
%
% IMPORTANT: SIIB uses a knn-mutual information estimator. There are two
% options for using the knn-mutual information estimator:
%
% 1) Use my MATLAB implementation. This implementation is slow and was
% not used for the evaluation in [1] or [2]. To use this option set
% knn_flag=true.
%
% 2) Use a C implementation made publicly available by the inventors of the
% knn estimator. This implementation runs faster and was used for the
% evaluation in [1] and [2]. To use this option set knn_flag=false. You may
% also need to recompile the C code for your machine. Ensure that the
% correct MATLAB command is used to execute the C-code: unix(./MIxnyn) or
% dos(MIxnyn.exe). See
% http://www.ucl.ac.uk/ion/departments/sobell/Research/RLemon/MILCA/MILCA
% for more details.
%
%--------------------------------------------------------------------------
% UPDATES
% 23/8/2017: Included missing C source code
% 3/10/2017: Included error message for signals with small durations
%
%
%--------------------------------------------------------------------------

if nargin <= 3
    knn_flag = false; % see above
end
if nargin <= 4
    execpath = 'MI_kraskov/MIxnyn';
end

if length(x)~=length(y)
    error('x and y should have the same length');
end

% initialization
fs = 16000;                             % sample rate of acoustic signals
window_length = 400;                    % 25 ms analysis window
step_size = 400/2;                      % 50% overlap
delta_dB = 40;                          % VAD threshold
R = 1/(step_size/fs);                   % frames/second
y = y(:)/max(std(x),eps);               % received speech
x = x(:)/max(std(x),eps);               % clean speech

% resample signals to fs
if fs_signal ~= fs
  x = resample(x, fs, fs_signal);
  y = resample(y, fs, fs_signal);
end

% get |STFT|^2
x_hat = stft(x, window_length, step_size, window_length); % apply short-time DFT to speech
y_hat = stft(y, window_length, step_size, window_length);
x_hat       = ( abs( x_hat(:, 1:(window_length/2+1)) ).^2 )'; % single-sided spectrum, spectra as columns
y_hat       = ( abs( y_hat(:, 1:(window_length/2+1)) ).^2 )';

% VAD
vad_index_x = getVAD(x, window_length, step_size, delta_dB);
x_hat = x_hat(:,vad_index_x);
y_hat = y_hat(:,vad_index_x);

% if size(x_hat,2)/R < 20 % check that the duration (after removing silence) is at least 20 s
%     error('stimuli must have at least 20 seconds of speech');
% end

% ERB gammatone filterbank
mn = 100;   % minimum center frequency
mx = 6500;  % maximum center frequency
J = round(21.4*log10(1+0.00437*mx)-21.4*log10(1+0.00437*mn)); % number of filters
[G,f_erb] = gammatone(fs, window_length, J, mn, mx);
X = log(G.^2*x_hat + eps); % equation (2) in [1]
Y = log(G.^2*y_hat + eps);

% forward temporal masking (see Rhebergen et al., 2006)
T0=1;
Tf = floor(0.2*R); % 200 ms
for j=1:J
    E_tf = min(X(j,:)); % 'hearing threshold' replacement (dB)
    Xfmf = zeros(1,size(X,2))-inf; % initialize forward masking function
    Yfmf = zeros(1,size(X,2))-inf;

    % overlap max (similar to overlap add)
    for i=1:size(X,2)
        ii = min( i:i+Tf-1, size(X,2)); % frame indices
        frame = X(j,ii);
        frame = frame(T0) - ( log((T0:Tf)/T0) ./ log(Tf/T0) )*(frame(T0)-E_tf); % forward masking function [Rhebergen et al., 2006]
        Xfmf(ii) = max(Xfmf(ii),frame); % max between clean signal and masking function

        frame = Y(j,ii);
        frame = frame(T0) - ( log((T0:Tf)/T0) ./ log(Tf/T0) )*(frame(T0)-E_tf);
        Yfmf(ii) = max(Yfmf(ii),frame);
    end

    X(j,:) = Xfmf;
    Y(j,:) = Yfmf;
end

% remove mean (for KLT)
X=X-repmat(mean(X,2),1,size(X,2));
Y=Y-repmat(mean(Y,2),1,size(Y,2));

% stack spectra
K = 15; % number of stacked vectors
temp  = X(:)';
X  = temp( bsxfun(@plus, [0:J*K-1]', 1:J:length(temp)-J*K) ); % equation (12) in [1]
temp  = Y(:)';
Y  = temp( bsxfun(@plus, [0:J*K-1]', 1:J:length(temp)-J*K) );

% KLT
[U,D] = eig(cov(X'));
X = U'*X;
Y = U'*Y;

% estimate MI (assuming no time-freq dependencies)
g = 150;
k = max(2, ceil(size(X,2)/g) ); % number of nearest neighbours (Kraskov recommends k=2-6 but really it depends on the amount of data available and bias vs variance tradeoff)
I_channels = zeros(1,size(X,1));
for j=1:size(X,1)
    I_channels(j) =  I_kras(X(j,:), Y(j,:), k, knn_flag, execpath);
end

% speech production channel
rho_p = 0.75;
Imx = -0.5*log2(1-rho_p^2);

% compute SIIB
I = (R/K)*sum( min(Imx, I_channels) ); % bit/s
I = max(0,I);



%%
function x_stft = stft(x, N, K, N_fft)
% short-time Fourier transform of x. The rows and columns of x_stft
% denote the frame-index and dft-bin index, respectively.
x = x(:);
frames = 1:K:(length(x)-N);
x_stft = zeros(length(frames), N_fft);
w = hann(N,'periodic');

for i = 1:length(frames)
  ii = frames(i):(frames(i)+N-1);
  x_stft(i, :) = fft(w.*x(ii), N_fft);
end

%%
function vad_ind = getVAD(x, N, K, range)
% returns the indices of voice active frames
frames = 1:K:(length(x)-N);
w = hann(N,'periodic');

x_dB = zeros(size(frames));
for i = 1:length(frames)
  ii = frames(i):(frames(i)+N-1); % indices of current frame
  x_dB(i) = 10*log10( mean((w.*x(ii)).^2) + eps ); % compute the power (dB) of each frame
end

x_dB_sort = sort(x_dB);
max_x = x_dB_sort(round(length(x_dB)*0.999)); % find the 99.9 percentile
vad_ind = find(x_dB>max_x - range);

%%
function  [A, cf] = gammatone(fs, N_fft, numBands, cf_min, cf_max)
% gammatone filterbank
erbminmax 	= 21.4*log10(4.37*([cf_min cf_max]./1000) + 1);        % convert to erbs
cf_erb      = linspace(erbminmax(1), erbminmax(2), numBands);      % linspace M filters on ERB-scale
cf          = (10.^(cf_erb./21.4)-1)./4.37*1000;                   % obtain center frequency in Hz
cf=cf(:);

order = 4;
a = factorial(order-1)^2/(pi*factorial(2*order-2)*2^-(2*order-2)); % Normalisation factor that ensures the gammatone filter has the correct ERB [Holdsworth & Patterson 1988].
b = a * 24.7.*(4.37.*cf./1000+1); % bandwidth

% frequency vector (Hz)
f = linspace(0, fs, N_fft+1);
f = f(1:(N_fft/2+1));

% filter bank
A = zeros(numBands, length(f));
for i=1:numBands
    temp = 1./(b(i)^2+(f-cf(i)).^2).^(order/2);    % gammatone magnitude response
    A(i,:) = temp/max(temp);                       % normalise the maximum value
end
cf=cf(:);
A(A<0.001) = 0;

%%
function I = I_kras(x,y,k,knn_flag, execpath)
% this function estimates the mutual information (in bits) of x and y using a non-parametric
% nearest neighbour estimator ['Estimating Mutual Information", Kraskov et al., 2004]
% Note that this function relies on C-code distributed by Kraskov et al.

x=x-mean(x); % make sure the sequences are scaled 'reasonably'
y=y-mean(y);
x=x/(std(x)+eps);
y=y/(std(y)+eps);

x=x+(1e-10)*randn(size(x)); % small amount of noise to prevent 'singularities'
y=y+(1e-10)*randn(size(x));

if knn_flag     % use my MATLAB implementation (not used in [1] or [2])
    N=length(x); % x and y must be [1xn] or [nx1] (this implementation assumes univariate data)
    nx = zeros(N,1);
    ny = zeros(N,1);

    for i=1:N
        dx = abs(x(i) - x); % distance from x(i) to x(j) where i/=j
        dy = abs(y(i) - y);
        dx(i) = [];
        dy(i) = [];

        dz = max(dx,dy);
        s = sort(dz);
        e = s(k); % distance to the k'th nearest neighbour

        nx(i) = sum(dx<e); % number of x(j) points with distance from x(i) less than e(i)
        ny(i) = sum(dy<e);
    end
    I = psi(k) - mean(psi(nx+1)+psi(ny+1)) + psi(N); % mutual info in nats (Eq. 8 in Kraskov)
    I = I/log(2); % nats to bits


else            % use Kraskov et al. implementation (requires C-code)
    [Ndx,N]=size(x);
    [Ndy,~]=size(y);

    % save data to disk
    xy=[x;y]';
    save xydata.txt xy -ascii

    % execute C-code (unix or windows)
    [status, output]=unix( [execpath, ' xydata.txt ',num2str(Ndx),' ',num2str(Ndy),' ',num2str(N),' ',num2str(k)] ); % unix
    %[status, output]=dos( ['MIxnyn.exe xydata.txt ',num2str(Ndx),' ',num2str(Ndy),' ',num2str(N),' ',num2str(k)] ); % windows
    if status
        display(output)
        error('Error: C-code did not execute \n Try compiling MIxnyn.C, or set knn_flag=true to use the MATLAB implementation.',[])
    else
        I=str2num(output)/log(2); % nats to bits
    end
end
