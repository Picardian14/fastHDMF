function [results] = findHomeostasis(decay, lr,G_span,dmf_pars,opts)
%FINDHOMEOSTASIS find a decay and lr that where target obj_rate is met


% if 2x1, optimized within bounds, otherwise dont optimize, use fix value provided in dmf_pars
% fitting variables could include G and OBJ_RATE in the future
opt_vars = [];
stren = sum(dmf_pars.C)./2;
if length(decay(:))==2 
    decayvals = optimizableVariable('decay',[decay(1) decay(2)], 'Transform','log');
    opt_vars = [decayvals];
end

if length(lr(:))==2 
    lr = optimizableVariable('lr',[lr(1) lr(2)], 'Transform','log');
    opt_vars = [opt_vars lr];
end

results = bayesopt(@aux_dyn_fic_dmf,opt_vars,opts{:});


    function [homeostatic_fittness, const, outdata] = aux_dyn_fic_dmf(decay_lr_params)
        const = [];
        dmf_pars.taoj = decay_lr_params.decay;
        dmf_pars.lrj = decay_lr_params.lr;
        homeostatic_fittness_list = zeros(length(G_span),1);
        corr_list = zeros(length(G_span),1);        
        fr_list = zeros(length(G_span),length(dmf_pars.C));
        plow_ptot_list = zeros(length(G_span),1);
        for idx=1:length(G_span)
            thispars = dmf_pars;
            thispars.G = G_span(idx);
            thispars.J = 0.75*thispars.G*stren' + 1; % updates it
            thispars.return_bold = false;
            [rates, rates_inh,~,~] = dyn_fic_DMF(thispars, thispars.nb_steps);
            rates = rates(:, (thispars.burnout*1000):end);
            reg_fr = mean(rates,2);
            fr_list(idx,:) = reg_fr;
            rates_fc = corrcoef(rates');
            corr_list(idx) = corr2(rates_fc-eye(length(dmf_pars.C)),dmf_pars.C);
            homeostatic_fittness_list(idx) = abs(thispars.obj_rate - mean(reg_fr));
        end
        mean_fit = squeeze(mean(homeostatic_fittness_list));
        std_fit = squeeze(std(homeostatic_fittness_list));
        homeostatic_fittness = mean_fit + 0.35 * mean_fit * std_fit;
        outdata = {homeostatic_fittness_list, corr_list,fr_list};
    end
end

