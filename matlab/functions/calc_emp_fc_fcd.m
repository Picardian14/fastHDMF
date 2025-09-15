% filepath: /home/ivan.mindlin/Minimal_HDMF/functions/calc_emp_fc_fcd.m
function [emp_fc, emp_fcd] = calc_emp_fc_fcd(data, params)
    % data: 3D matrix with dimensions (nodes x time x subjects)
    % params: structure containing necessary parameters, e.g.:
    %   - burnout, TR, flp, fhi, wsize, overlap, N (number of nodes)
    
    N = size(data,1);
    NSUB = size(data,3);
    Isubdiag = find(tril(ones(N), -1));
    
    % Preallocate based on data size
    for nsub = 1:NSUB
        % Extract post-burnout data
        Wdata(:,:,nsub) = data(:, params.burnout:end, nsub);
        % Filter the bold signal
        WdataF(:,:,nsub) = permute(filter_bold(Wdata(:,:,nsub)', params.flp, params.fhi, params.TR), [2 1]);
        % Compute correlation matrices
        WFCdata(nsub,:,:) = corrcoef(squeeze(Wdata(:,:,nsub))');
        WFCdataF(nsub,:,:) = corrcoef(squeeze(WdataF(:,:,nsub))');
        % Compute FCD for this subject using compute_fcd
        tmp_time_fc = compute_fcd(WdataF(:,:,nsub)', params.wsize, params.overlap, Isubdiag);
        emp_fcd(nsub,:,:) = corrcoef(tmp_time_fc);
    end
    
    % Average FC across subjects from filtered data
    WFCdataF = permute(WFCdataF, [2,3,1]);
    emp_fc = mean(WFCdataF,3);
end