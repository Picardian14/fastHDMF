function filt_bold = filter_bold(bold,flp,fhp,tr)
% filters the bold signal iin TXN matrix
% k is the order of the filter
[T,N] = size(bold);
fnq=1/(2*tr);                 % Nyquist frequency
Wn=[flp/fnq fhp/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter
% filtering and plotting
filt_bold = zeros(T,N);
nzeros = 40;
aux_filt = detrend(bold);
aux_filt = cat(1,zeros(nzeros,N),aux_filt,zeros(nzeros,N));
for n = 1:N        
%     aux_filt(isinf(aux_filt(:,n)),n) = 3.*std(aux_filt(:,n));
    aux_filt2 = filtfilt(bfilt,afilt,aux_filt(:,n));   % Zero phase filter the data
    filt_bold(:,n)= zscore(aux_filt2(nzeros+1:end-nzeros));
end
