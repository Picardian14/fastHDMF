function run_fit(varargin)
% RUN_FIT Unified entry point for all dynamic/static FC/FCD fitting tasks.
% This script replaces AAL_/Bayesian_ specific driver scripts by exposing
% a parameterized interface usable locally or via SLURM.
%
% Usage (MATLAB interactive):
%   run_fit('parcellation','AAL','dataset','coma24', ...)
%
% Usage (command line / -batch):
%   matlab -nodisplay -batch "run_fit('parcellation','AAL','dataset','coma24','fit_mode','dyn_fc')"
%
% Key parameters (name-value pairs):
%   parcellation : 'AAL' | 'HCP' (drives SC loading & node count trimming)
%   dataset      : 'coma24' | 'awake' (empirical BOLD set & TR)
%   fit_mode     : 'dyn_fc' | 'dyn_fcd' | 'dyn_both' | 'stat_fc' | 'stat_fcd'
%   out_root     : (default '../results') base results folder
%   obj_rate     : target firing rate (default 3.44)
%   hours        : wall time budget for BO (affects MaxTime) default 12 (4 for stat_*)
%   resume       : true/false attempt to resume from checkpoint file
%   with_plasticity : true/false (default dyn_* true, stat_* false)
%   with_decay      : true/false (default dyn_* true, stat_* false)
%   TR           : override repetition time (if omitted uses dataset default)
%   burnout      : override burnout (default 7 for dyn & stat after preprocessing)
%   wsize        : window size (default 30)
%   overlap      : window overlap (default 29)
%   cpus         : parallel workers to open (default 24 for dyn, 0 for stat to avoid overhead)
%
% Output file naming:
%   <out_root>/<fit_mode>/results_<PARCELLATION>_<FITMODE>.mat
%
% Backwards compatibility:
%   The previous driver scripts can be reduced to simple wrappers calling run_fit.
%
% NOTE: Minimizes duplication; dynamic_fitting vs static_fitting selected by fit_mode.

try
    addpath ../dynamic_fic_dmf_Cpp ../results/ functions/ ../slurm/ ../data/

    % ------------------- Parse inputs -------------------
    p = inputParser;
    addParameter(p,'parcellation','AAL');
    addParameter(p,'dataset','coma24');
    addParameter(p,'fit_mode','dyn_fc');
    addParameter(p,'out_root','../results');
    addParameter(p,'obj_rate',3.44);
    addParameter(p,'hours',[]); % auto
    addParameter(p,'resume',false);
    addParameter(p,'with_plasticity',[]);
    addParameter(p,'with_decay',[]);
    addParameter(p,'TR',[]);
    addParameter(p,'burnout',[]);
    addParameter(p,'wsize',30);
    addParameter(p,'overlap',29);
    addParameter(p,'cpus',[]);
    parse(p,varargin{:});
    args = p.Results;

    fit_mode = args.fit_mode; % convenience
    is_dynamic = startsWith(fit_mode,'dyn');
    is_static  = startsWith(fit_mode,'stat');

    if isempty(args.hours)
        if is_static
            args.hours = 4;
        else
            args.hours = 12;
        end
    end

    if isempty(args.with_plasticity)
        args.with_plasticity = is_dynamic; % previous scripts: dyn true, stat false
    end
    if isempty(args.with_decay)
        args.with_decay = is_dynamic; % same logic
    end

    if isempty(args.cpus)
        if is_dynamic
            args.cpus = 24; else; args.cpus = 0; end
    end

    % ------------------- Load structural connectivity -------------------
    switch upper(args.parcellation)
        case 'AAL'
            load '../data/ts_coma24_AAL_symm_withSC.mat' SC timeseries_CNT24_symm; %#ok<LOAD>
            C = SC;
            if isempty(args.TR); defaultTR = 2.4; else; defaultTR = args.TR; end
            bold_source = 'timeseries_CNT24_symm';
            raw_timeseries = timeseries_CNT24_symm; %#ok<NODEF>
        case 'HCP'
            load '../data/DTI_fiber_consensus_HCP.mat' connectivity; %#ok<LOAD>
            C = connectivity(1:200,1:200);
            if isempty(args.TR); defaultTR = 0.72; else; defaultTR = args.TR; end
            if strcmpi(args.dataset,'awake') || strcmpi(args.dataset,'hcp')
                load '../data/BOLD_timeseries_Awake.mat' BOLD_timeseries_Awake; %#ok<LOAD>
                raw_timeseries = cellfun(@(x) x(1:200,:), BOLD_timeseries_Awake,'UniformOutput',false);
                bold_source = 'BOLD_timeseries_Awake';
            else
                error('Unsupported dataset %s for parcellation HCP', args.dataset);
            end
        otherwise
            error('Unknown parcellation %s', args.parcellation);
    end

    C = 0.2.*C./max(C(:));
    params = dyn_fic_DefaultParams('C',C);

    % Window / filtering defaults
    params.flp = 0.008; 
    params.fhi = 0.09; 
    params.wsize = args.wsize;
    params.overlap = args.overlap;
    params.with_plasticity = args.with_plasticity;
    params.with_decay = args.with_decay;

    % Fit flags
    params.fit_fc = false; params.fit_fcd = false; params.fit_both = false;
    switch fit_mode
        case 'dyn_fc';  params.fit_fc = true;
        case 'dyn_fcd'; params.fit_fcd = true;
        case 'dyn_both'; params.fit_both = true; % both fc and fcd
        case 'stat_fc';  params.fit_fc = true;
        case 'stat_fcd'; params.fit_fcd = true;
        otherwise; error('Unknown fit_mode %s', fit_mode);
    end

    params.return_bold = true;
    params.return_rate = false;
    params.return_fic  = false;
    params.obj_rate = args.obj_rate;

    % Burnout defaulting (scripts often set initial then override to 7)
    if isempty(args.burnout); params.burnout = 7; else; params.burnout = args.burnout; end

    % ------------------- Convert cell time series -> 3D array -------------------
    params.NSUB = length(raw_timeseries);
    params.N = length(C);
    data = zeros(params.N, 0, params.NSUB); % will size after first
    for nsub = 1:params.NSUB
        ts = raw_timeseries{nsub};
        if size(ts,1) ~= params.N
            error('Timeseries %d has %d nodes expected %d', nsub, size(ts,1), params.N);
        end
        if nsub == 1
            data = zeros(params.N, size(ts,2), params.NSUB);
        end
        data(:,:,nsub) = ts;
    end

    params.TR = defaultTR; %#ok<NASGU>
    params.T = size(data,2);
    params.TMAX = params.T - params.burnout;

    Isubdiag = find(tril(ones(params.N),-1)); %#ok<NASGU>

    % Precompute filtered data & empirical stats
    for nsub = 1:params.NSUB
        Wdata(:,:,nsub) = data(:, params.burnout:end, nsub); %#ok<AGROW>
        WdataF(:,:,nsub) = permute(filter_bold(Wdata(:, :,nsub)', params.flp, params.fhi, defaultTR), [2 1 3]); %#ok<AGROW>
        WFCdata(nsub,:,:) = corrcoef(squeeze(Wdata(:,:,nsub))'); %#ok<AGROW>
        WFCdataF(nsub,:,:) = corrcoef(squeeze(WdataF(:,:,nsub))'); %#ok<AGROW>
        if any(strcmp(fit_mode,{'dyn_fcd','dyn_both','stat_fcd'}))
            tmp_time_fc = compute_fcd(WdataF(:,:,nsub)', params.wsize, params.overlap, find(tril(ones(params.N),-1)));
            emp_fcd(nsub,:,:) = corrcoef(tmp_time_fc); %#ok<AGROW>
        end
    end
    WFCdataF = permute(WFCdataF,[2 3 1]);
    emp_fc = mean(WFCdataF,3);

    % ------------------- Ranges -------------------
    if is_dynamic
        LR_range = [1 10000];
        G_range  = [0.1 16];
    else
        ALPHA_range = [0.01 0.99];
        G_range     = [0.1 16];
    end

    % Steps / windows
    params.win_start = 0:params.wsize-params.overlap:params.TMAX-params.wsize-1;
    params.nwins = length(params.win_start);
    if is_dynamic
        params.nb_steps = fix((params.T)*defaultTR)/params.dtt; % follow dyn scripts
    else
        params.nb_steps = fix((params.T)*defaultTR)/params.dtt; % same formula
    end

    % ------------------- BO options -------------------
    checkpoint_dir = fullfile(args.out_root, fit_mode);
    if ~exist(checkpoint_dir,'dir'); mkdir(checkpoint_dir); end
    checkpoint_file = fullfile(checkpoint_dir, ['results_' upper(args.parcellation) '_' fit_mode '.mat']);

    bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',args.cpus>0,...
        'MinWorkerUtilization',max(4,min(args.cpus,8)), ...
        'AcquisitionFunctionName','expected-improvement-plus', ...
        'MaxObjectiveEvaluations',1e16, ...
        'ParallelMethod','clipped-model-prediction', ...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',args.hours*3600, ...
        'OutputFcn', @saveToFile, ...
        'SaveFileName', checkpoint_file, ...
        'PlotFcn', {@plotObjectiveModel,@plotMinObjective}};
    if args.resume && exist(checkpoint_file,'file')
        bo_opts = [bo_opts {'Resume', checkpoint_file}]; %#ok<AGROW>
    end

    % ------------------- Run fitting -------------------
    results = [];
    pool_started = false;
    try
        if args.cpus > 0
            parpool('local', args.cpus);
            pool_started = true;
        end
        if is_dynamic
            if exist('emp_fcd','var') && (params.fit_fcd || params.fit_both)
                target = emp_fcd; %#ok<NASGU>
            else
                target = emp_fc; %#ok<NASGU>
            end
            results = dynamic_fitting(G_range, exist('LR_range','var')*LR_range + (~exist('LR_range','var'))*[] , params, bo_opts, eval('target')); %#ok<EVAL>
        else
            if params.fit_fcd
                target = emp_fcd; %#ok<NASGU>
                results = static_fitting(G_range, ALPHA_range, params, bo_opts, target);
            else
                target = emp_fc; %#ok<NASGU>
                results = static_fitting(G_range, ALPHA_range, params, bo_opts, target);
            end
        end
    catch inner
        fprintf(2,'Error during fitting: %s\n', inner.message);
        fprintf(2,'%s\n', getReport(inner,'extended'));
        rethrow(inner);
    finally %#ok<*NOILL>
        if pool_started
            delete(gcp('nocreate'));
        end
    end

    % ------------------- Save -------------------
    outfile = checkpoint_file; % same naming
    save(outfile,'results','args','fit_mode','emp_fc','-v7.3');
    if exist('emp_fcd','var')
        save(outfile,'emp_fcd','-append');
    end
    fprintf('Saved results to %s\n', outfile);
catch ME
    fprintf(2,'RUN_FIT failed: %s\n', ME.message);
    fprintf(2,'%s\n', getReport(ME,'extended'));
    exit(1);
end
exit(0);
end
