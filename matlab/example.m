% Remember to run `mex dynamic_fic_dmf_Cpp/dyn_fic_DMF.cpp`

addpath ../data ../dynamic_fic_dmf_Cpp
% Load connectivity matrix and set parameters
load('../data/DTI_fiber_consensus_HCP.mat', 'connectivity'); 
C = connectivity(1:200, 1:200);
C = 0.2 * C / max(C(:));
N = size(C, 1);
params = dyn_fic_DefaultParams('C',C);
params.C = C;

% Homeostatic parameters depend on the objective rate
params.obj_rate = 3.44;
% Pre-computed a,b coefficients for objective rate 3.44
if params.obj_rate == 3.44
    coeffs = load('../data/fit_res_3-44.mat'); 
    a = coeffs.fit_res(2);
    b = coeffs.fit_res(1);
else
    error('Need to calculate a,b coefficients for the desired objective rate');
end

% To explore the effect of the decay set these as desired
params.with_decay = true;
params.with_plasticity = true;
LR = 10;
% Calculate the decay using the a,b coefficients
DECAY = exp(a + log(LR) * b);
% Set heterogeneous vectors or scalars
params.taoj_vector = ones(N,1) * DECAY;
params.lr_vector = ones(N,1) * LR;

% Set the global coupling strength
G_VAL = 2.1;
params.G = G_VAL;
% Initialize FIC values using the linear solution from fastDMF paper
params.J = 0.75 * params.G * sum(params.C, 1)' + 1; 

% Short simulation
nb_steps = 50000;
params.seed = 1;

% Assume you want everything returned
params.return_bold = true;
params.return_fic = true;
params.return_rate = true;


[rates_dyn, rates_inh_dyn, bold_dyn, fic_t_dyn] = dyn_fic_DMF(params, nb_steps); 