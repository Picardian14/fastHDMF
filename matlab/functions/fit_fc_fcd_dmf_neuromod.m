function [opt_fc_error,opt_fcd_ks,opt_pars,results] = fit_fc_fcd_dmf_neuromod(T,emp_fc,emp_fcd,G,nm_e,nm_i,dmf_pars,opts)
%
% Function to find optimal DMF with neuromodulation parameters to fit either the FC or the FCD
% of bold signals. Assumes dmf_pars.J is the right one.
% T = seconds to simulate
% tr = scalar, TR of simulated BOLD signals
% emp_fc = N x N empirical FC
% emp_fcd = N x N empirical FCD. If empty, only works with FCD.
% G = 2 x 1 with upper and lower bounds on G or empty
% nm_e = 2 x 1 with upper and lower bounds Excitatory neuromodulation
% nm_i = 2 x 1 with upper and lower bounds Inhibitory neuromodulation
% dmf_pars = structure generarted by dyn_fic_DefaultParams with DMF.
% Includes receptors to neuromodulation
% parameters
% opts = options for the fit and optimization.

% Setting DMF parameters
N = size(dmf_pars.C,1);
stren = sum(dmf_pars.C)./2;
isubfc = find(tril(ones(N),-1));
nsteps = T.*(1000); % number of DMF timepoints


opt_vars = [];
if length(G(:))==2 % if 2x1, optimized within bounds, otherwise dont optimize, use fix value provided in dmf_pars
    gvals = optimizableVariable('G',[G(1) G(2)]);
    opt_vars = [gvals];
end

if length(nm_e(:))==2 % if 2x1, optimized within bounds, otherwise dont optimize
    nmevals = optimizableVariable('nm_e',[nm_e(1) nm_e(2)]);
    opt_vars = [opt_vars nmevals];
end

if length(nm_i(:))==2 % if 2x1, optimized within bounds, otherwise dont optimize
    nmivals = optimizableVariable('nm_i',[nm_i(1) nm_i(2)]);
    opt_vars = [opt_vars nmivals];
end

% BAYES OPTIMIZATION
if isempty(emp_fcd) % only optimizes FC
    results = bayesopt(@aux_dmf_fit_fc,opt_vars,opts{:});
    opt_fc_error.results = results.MinEstimatedObjective;
    opt_fcd_ks = [];
    opt_pars =results.XAtMinEstimatedObjective;
    
else % OPTIMIZES FCD and returns FC GOF
    
    results = bayesopt(@aux_dmf_fit_fcd,opt_vars,opts{:});
    
    opt_fcd_ks = results.MinEstimatedObjective;
    opt_pars =results.XAtMinEstimatedObjective;
    [~,min_id] = min(results.EstimatedObjectiveMinimumTrace);
    opt_fc_error = results.UserDataTrace{min_id};
end


    function [fc_error] = aux_dmf_fit_fc(g_nm_ei)
        % Unpacking parameters
        thispars = unpack_parameters(g_nm_ei);
        
        % Simulating
        bold = dyn_fic_DMF(thispars, nsteps,'bold'); % runs simulation
        bold = bold(:,dmf_pars.burnout:end); % remove initial transient
        bold(isnan(bold))=0;
        bold(isinf(bold(:)))=max(bold(~isinf(bold(:))));
        if isempty(bold)            
            fc_error = nan;
            return
        end
        % Filtering and computing FC
        filt_bold = filter_bold(bold',dmf_pars.flp,dmf_pars.fhi,dmf_pars.TR);
        sim_fc = corrcoef(filt_bold);
        
        % Computing error: 1-Corrrelation between FC's
        fc_error= 1-corr2(sim_fc(isubfc),emp_fc(isubfc));
        
    end

    function [fcd_ks,const,fc_error] = aux_dmf_fit_fcd(g_nm_ei)
        const = [];
        % Unpacking parameters
        thispars = unpack_parameters(g_nm_ei);
        
        % Simulating
        bold = dyn_fic_DMF(thispars, nsteps,'bold'); % runs simulation
        bold = bold(:,dmf_pars.burnout:end); % remove initial transient
        bold(isnan(bold))=0;
        bold(isinf(bold(:)))=max(bold(~isinf(bold(:))));
        % Filtering and computing FC
        filt_bold = filter_bold(bold',dmf_pars.flp,dmf_pars.fhi,dmf_pars.TR);
        sim_fc = corrcoef(filt_bold);
        % Computing FC error: 1-Corrrelation between FC's
        fc_error= 1-corr2(sim_fc(isubfc),emp_fc(isubfc));
        % FCD
        sim_fcd = compute_fcd(filt_bold,dmf_pars.wsize,dmf_pars.overlap,isubfc);
        sim_fcd(isnan(sim_fcd))=0;
        sim_fcd = corrcoef(sim_fcd);
        if isempty(sim_fcd)
            fcd_ks = nan;
            fc_error = nan;
            return
        end
        [~,~,fcd_ks] = kstest2(sim_fcd(:),emp_fcd(:));
    end


    function thispars = unpack_parameters(g_nm_ei)
        % Unpacking parameters
        thispars = dmf_pars;
%         thispars.J = 1.5*thispars.G*stren' + 1;
        if ismember('G',g_nm_ei.Properties.VariableNames)
            thispars.G = g_nm_ei.G;
            thispars.J = 0.75*thispars .G*stren' + 1; % updates it
        end
        if ismember('nm_e',g_nm_ei.Properties.VariableNames)
            thispars.wgaine = g_nm_ei.nm_e;
        else
            thispars.wgaine = nm_e;
        end
        if ismember('nm_i',g_nm_ei.Properties.VariableNames)
            thispars.wgaini = g_nm_ei.nm_i;
        else
            thispars.wgaini = nm_i;
        end
        
        % optimizes both parameters with the same value
        if isempty(nm_i) && ~isempty(nm_i)
            thispars.wgaini = g_nm_ei.nm_e;
        end
    end
end