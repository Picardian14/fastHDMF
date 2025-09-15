function fcd = compute_fcd(data,wsize,overlap,isubdiag)
% computes the fcd for a T x N multivariate dataset using a window of
% 'wsize' points and an overlap of 'overlap' points.
[T,~]=size(data);

win_start = 0:wsize-overlap:T-wsize-1;
nwins = length(win_start);
fcd = zeros(length(isubdiag),nwins);

for i=1:nwins
    tmp = data(win_start(i)+1:win_start(i)+wsize+1,:);
%     tmp = data(i+1:overlap+i,:);
    cormat = corrcoef(tmp);
    fcd(:,i) = cormat(isubdiag);
    
end

