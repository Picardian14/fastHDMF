#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=AAL_NVC_stat_fc
#SBATCH --mail-type=END
#SBATCH --mail-user=
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --chdir=/network/iss/home/ivan.mindlin/Repos/fastHDMF/matlab
#SBATCH --output=../slurm/outputs/AAL_NVC_stat_fc.out
#SBATCH --error=../slurm/errors/AAL_NVC_stat_fc.err

ml matlab/R2022b
ml glib
matlab -nodisplay<<-EOF

clear all;
close all;
addpath ../dynamic_fic_dmf_Cpp ../results/ functions/ ../slurm/ ../data/


folder_name = '../results/stat_fc';
if ~exist(folder_name, 'dir')
    mkdir(folder_name);
end

sub_experiment_name = "AAL_NVC_stat_fc";
% Load Data
load ../data/ts_coma24_AAL_symm_withSC.mat
C = SC;
C = 0.2.*C./max(C(:));
stren = sum(C);
params = dyn_fic_DefaultParams('C',C);
% fitting options
params.fit_fc = true;
params.fit_fcd = false;
% type of fic calculation
params.with_plasticity=false;
params.with_decay=false;

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
%params.seed = sub_experiment_name;

% Save in data the timeseries
params.NSUB=length(timeseries_CNT24_symm);
indexsub=1:params.NSUB;
for nsub=indexsub
    data(:, :, nsub)=timeseries_CNT24_symm{nsub};
end
Isubdiag = find(tril(ones(params.N),-1));
params.burnout = 7;
params.T = size(data,2);
params.TMAX = params.T - params.burnout;
params.TR = 2.4

indexsub=1:params.NSUB;
for nsub=indexsub
    nsub;    
    Wdata(:,:,nsub)=data(:, params.burnout:end, nsub) ; 
    WdataF(:,:,nsub) = permute(filter_bold(Wdata(:, :,nsub)', params.flp, params.fhi, params.TR), [2 1 3]);    
    WFCdataF(nsub,:,:)=corrcoef(squeeze(WdataF(:,:,nsub))'); % toma las correlaciones de todos los nodos entre sí para cada sujeto
end
WFCdataF = permute(WFCdataF, [2,3,1]);
emp_fc = mean(WFCdataF,3);
NHOURS = 4;
% bayesian model params
checkpoint_file = "Results/stat_fc/results_"+sub_experiment_name+".mat";
bo_opts = {'IsObjectiveDeterministic',false,'UseParallel',true,... %% Will be determinsitic so we do not estimate error
        'MinWorkerUtilization',8,...
        'AcquisitionFunctionName','expected-improvement-plus',...
        'MaxObjectiveEvaluations',1e16,...
        'ParallelMethod','clipped-model-prediction',...
        'GPActiveSetSize',300,'ExplorationRatio',0.5,'MaxTime',NHOURS*3600,...  
        'OutputFcn', @saveToFile,...
        'SaveFileName', checkpoint_file,...
        'PlotFcn', {@plotObjectiveModel,@plotMinObjective}};
        %'OutputFcn',@stoppingCriteria,...        %% We leave it running without criteria



params.win_start = 0:params.wsize-params.overlap:params.TMAX-params.wsize-1;
params.nwins = length(params.win_start);
params.nb_steps = fix((params.T)*params.TR)/params.dtt; % Generate the same amount of time points ant then remove the transient period
ALPHA_range = [0.01 0.99];
G_range = [0.1 16];
params.ALPHA_range = ALPHA_range;
params.G_range = G_range;
% training

%%
results = static_fitting(G_range,ALPHA_range,params,bo_opts, emp_fc);
close all;
% save results
filename = sprintf('%s/%s.mat', folder_name, sub_experiment_name); % Create filename
save(filename, 'results'); % Save results in a .mat file

EOF
