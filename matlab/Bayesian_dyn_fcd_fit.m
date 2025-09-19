try
    clear all;
    close all;
    addpath ../dynamic_fic_dmf_Cpp ../results/ functions/ ../slurm/ .../data/

    folder_name = '../results/dyn_fcd';
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end

    sub_experiment_name = "HCP_NVC";

    % Resume from checkpoint option
    resume_from_checkpoint = false; % Set to true to resume from existing checkpoint

    % Load Data
    load ../data/DTI_fiber_consensus_HCP.mat
    C = connectivity(1:200,1:200);
    C = 0.2.*C./max(C(:));
    params = dyn_fic_DefaultParams('C',C);
    % Fitting params
    params.fit_fc = false;
    params.fit_fcd = true;
    % type of fic calculation
    params.with_plasticity=true;
    params.with_decay=true;
    % Setting model parameters
    params.return_rate=false;
    params.return_fic=false;
    params.return_bold=true;

    params.obj_rate = 3.44;

    % basic model parameters
    params.burnout =  2;
    params.flp = 0.008; 
    params.fhi = 0.09; 
    params.wsize = 30; 
    params.overlap = 29; 
    params.N=length(params.C);

    load('../data/BOLD_timeseries_Awake.mat')
    % Save in data the timeseries
    params.NSUB=length(BOLD_timeseries_Awake);
    indexsub=1:params.NSUB;
    for nsub=indexsub
        data(:, :, nsub)=BOLD_timeseries_Awake{nsub}(1:200,:);
    end
    Isubdiag = find(tril(ones(params.N),-1));

    params.burnout = 7;
    params.T = size(data,2);
    params.TMAX = params.T - params.burnout;

    indexsub=1:params.NSUB;
    for nsub=indexsub
        nsub;    
        Wdata(:,:,nsub)=data(:, params.burnout:end, nsub) ; 
        WdataF(:,:,nsub) = permute(filter_bold(Wdata(:, :,nsub)', params.flp, params.fhi, params.TR), [2 1 3]);
        WFCdata(nsub,:,:)=corrcoef(squeeze(Wdata(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
        WFCdataF(nsub,:,:)=corrcoef(squeeze(WdataF(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
        tmp_time_fc = compute_fcd(WdataF(:,:,nsub)',params.wsize, params.overlap,Isubdiag);
        emp_fcd(nsub, :, :) = corrcoef(tmp_time_fc);
    end

    WFCdata = permute(WFCdata, [2,3,1]);
    WFCdataF = permute(WFCdataF, [2,3,1]);
    emp_fc = mean(WFCdataF,3);
    NHOURS = 12;
    % bayesian model params
    checkpoint_file = "../results/dyn_fcd/results_"+sub_experiment_name+".mat";
    
    % Configure bayesian optimization options
    if resume_from_checkpoint && exist(checkpoint_file, 'file')
        % Resume from existing checkpoint
        bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,... %% Will be determinsitic so we do not estimate error
                'MinWorkerUtilization',8,...
                'AcquisitionFunctionName','expected-improvement-plus',...
                'MaxObjectiveEvaluations',1e16,...
                'ParallelMethod','clipped-model-prediction',...
                'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',NHOURS*3600,...  
                'OutputFcn', @saveToFile,...
                'SaveFileName', checkpoint_file,...
                'PlotFcn', {@plotObjectiveModel,@plotMinObjective},...
                'Resume', checkpoint_file};
    else
        % Start fresh optimization
        bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,... %% Will be determinsitic so we do not estimate error
                'MinWorkerUtilization',8,...
                'AcquisitionFunctionName','expected-improvement-plus',...
                'MaxObjectiveEvaluations',1e16,...
                'ParallelMethod','clipped-model-prediction',...
                'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',NHOURS*3600,...  
                'OutputFcn', @saveToFile,...
                'SaveFileName', checkpoint_file,...
                'PlotFcn', {@plotObjectiveModel,@plotMinObjective}};
    end





    %params.win_start = np.arange(0, TMAX - wsize - 1, wsize - overlap)
    params.win_start = 0:params.wsize-params.overlap:params.TMAX-params.wsize-1;
    params.nwins = length(params.win_start);
    %int((data.shape[-1]-burnout)*params['TR']/params['dtt'])

    params.nb_steps = fix((params.T)*params.TR)/params.dtt; % Generate the same amount of time points ant then remove the transient period
    LR_range = [1 10000];
    G_range = [0.1 16];
    % seed fixed for a training
    %params.seed = sub_experiment_name;
    % training

    %%
    parpool(24); % Open a parallel pool with 24 workers
    results = dynamic_fitting(G_range,LR_range,params,bo_opts, emp_fcd);
    delete(gcp('nocreate'));
    close all;
    % save results
    filename = sprintf('%s/%s.mat', folder_name, sub_experiment_name); % Create filename
    save(filename, 'results'); % Save results in a .mat file
catch ME
    disp(ME.message)
end
