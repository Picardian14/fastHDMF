try

    clear all;
    close all;
    addpath ../dynamic_fic_dmf_Cpp ../results/ functions/ ../slurm/ ../data/

    folder_name = '../results/dyn_fcd';
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end

    sub_experiment_name = "AAL_NVC_dyn_fcd";

    % Load Data
    load ../data/ts_coma24_AAL_symm_withSC.mat
    C = SC;
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
    params.burnout =  8;
    params.flp = 0.008; 
    params.fhi = 0.09; 
    params.wsize = 30; 
    params.overlap = 29; 
    params.N=length(params.C);
    % Save in data the timeseries
    params.NSUB=length(timeseries_CNT24_symm);
    indexsub=1:params.NSUB;
    for nsub=indexsub
        data(:, :, nsub)=timeseries_CNT24_symm{nsub};
    end
    Isubdiag = find(tril(ones(params.N),-1));

    params.T = size(data,2);
    params.TMAX = params.T - params.burnout;
    params.TR = 2.4;


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
    bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',false,... %% Will be determinsitic so we do not estimate error
            'MinWorkerUtilization',8,...
            'AcquisitionFunctionName','expected-improvement-plus',...
            'MaxObjectiveEvaluations',1e16,...
            'ParallelMethod','clipped-model-prediction',...
            'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',NHOURS*3600,...  
            'OutputFcn', @saveToFile,...
            'SaveFileName', checkpoint_file,...
            'PlotFcn', {@plotObjectiveModel,@plotMinObjective}};
            %'OutputFcn',@stoppingCriteria,...        %% We leave it running without criteria





    %params.win_start = np.arange(0, TMAX - wsize - 1, wsize - overlap)
    params.win_start = 0:params.wsize-params.overlap:params.TMAX-params.wsize-1;
    params.nwins = length(params.win_start);
    %int((data.shape[-1]-burnout)*params['TR']/params['dtt'])

    params.nb_steps = ceil((params.T)*params.TR)/params.dtt; % Generate the same amount of time points ant then remove the transient period
    LR_range = [1 10000];
    G_range = [0.1 16];
    % seed fixed for a training
    %params.seed = sub_experiment_name;
    % training

    %%
    %parpool('local',24); % Open a pool with 24 workers
    results = dynamic_fitting(G_range,LR_range,params,bo_opts, emp_fcd);
    %delete(gcp('nocreate')); % Close the pool
    close all;
    % save results
    filename = sprintf('%s/%s.mat', folder_name, sub_experiment_name); % Create filename
    save(filename, 'results'); % Save results in a .mat file
catch ME
    disp('An error occurred:');
    disp(ME.message);
    disp(ME.getReport());
end