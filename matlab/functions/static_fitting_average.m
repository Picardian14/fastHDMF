function [results] = static_fitting(G, alpha, dmf_pars, opts, observable)
    % Static fitting (no plasticity/decay) with multiple stochastic simulations.
    % Aggregation strategy (mirrors dynamic version):
    %   - Average FC across runs.
    %   - Pool lower triangle entries of FCD correlation matrices across runs (distribution).
    %   - Compute objective based on chosen fitting flags.

    if (dmf_pars.with_decay || dmf_pars.with_plasticity)
        error("Error: Parameters not set to static");
    end

    opt_vars = [];
    stren = sum(dmf_pars.C);
    if numel(G)==2
        Gvals = optimizableVariable('G',[G(1) G(2)]);
        opt_vars = [opt_vars Gvals];
    end
    if numel(alpha)==2
        alpha = optimizableVariable('alpha',[alpha(1) alpha(2)]);
        opt_vars = [opt_vars alpha];
    end

    results = bayesopt(@aux_dyn_fic_dmf,opt_vars,opts{:});

    function [out_error, const, outdata] = aux_dyn_fic_dmf(G_alpha_params)
        const = [];
        thispars = dmf_pars;
        thispars.G = G_alpha_params.G;
        thispars.alpha = G_alpha_params.alpha;
        thispars.lr_vector = zeros(thispars.N,1);
        thispars.taoj_vector = zeros(thispars.N,1);
        thispars.J = thispars.alpha * thispars.G * stren' + 1;

        isubfc_idx = find(tril(ones(thispars.N), -1));
        fc_sum = zeros(thispars.N, thispars.N);
        fcd_pool = [];
        mean_rates_runs = zeros(thispars.NSUB, thispars.N);

        for run = 1:thispars.NSUB
            [rates, ~, bold, ~] = dyn_fic_DMF(thispars, thispars.nb_steps);
            rates = rates(:, (thispars.burnout*thispars.TR/thispars.dtt):end);
            bold = bold(:, thispars.burnout:end);
            if isempty(bold)
                out_error = NaN; outdata = {[]}; return;
            end
            bold(~isfinite(bold)) = 0;

            filt_bold = filter_bold(bold', thispars.flp, thispars.fhi, thispars.TR);
            sim_fc = corrcoef(filt_bold);
            sim_fc(~isfinite(sim_fc)) = 0;
            fc_sum = fc_sum + sim_fc;

            sim_fcd_mat = compute_fcd(filt_bold, thispars.wsize, thispars.overlap, isubfc_idx);
            if ~isempty(sim_fcd_mat)
                sim_fcd_mat(~isfinite(sim_fcd_mat)) = 0;
                sim_fcd_corr = corrcoef(sim_fcd_mat);
                sim_fcd_corr(~isfinite(sim_fcd_corr)) = 0;
                fcd_pool = [fcd_pool; sim_fcd_corr(isubfc_idx)]; %#ok<AGROW>
            end

            mean_rates_runs(run, :) = mean(rates,2)';
        end

        if thispars.fit_fc || thispars.fit_both
            fc_obs = observable;
            if size(fc_obs,1) ~= thispars.N || size(fc_obs,2) ~= thispars.N
                error('Observable FC dimensions mismatch');
            end
            avg_fc = fc_sum / thispars.NSUB;
            fc_term = 1 - corr(avg_fc(isubfc_idx), fc_obs(isubfc_idx));
        else
            fc_term = 0;
        end

        if thispars.fit_fcd || thispars.fit_both
            if isempty(fcd_pool)
                out_error = NaN; outdata = {[]}; return;
            end
            obs_fcd_dist = observable(:); % assume passed distribution when fitting FCD
            obs_fcd_dist = obs_fcd_dist(~isnan(obs_fcd_dist));
            try
                [~,~,ks] = kstest2(fcd_pool(:), obs_fcd_dist(:));
            catch
                ks = 1;
            end
        else
            ks = 0;
        end

        if thispars.fit_fc && ~thispars.fit_fcd
            out_error = fc_term;
        elseif thispars.fit_fcd && ~thispars.fit_fc
            out_error = ks;
        elseif thispars.fit_both
            out_error = 0.5*fc_term + 0.5*ks;
        else
            out_error = NaN;
        end

        mean_rates = mean(mean_rates_runs,1)';
        outdata = {mean_rates};
    end
end
    
    