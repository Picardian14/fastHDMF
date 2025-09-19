function [results] = dynamic_fitting(G, lr, dmf_pars, opts, observable)
    % Dynamic fitting (plasticity + adaptation) with multiple stochastic simulations.
    % Aggregation strategy:
    %   - Run model NSUB times (assuming internal stochasticity or noise seeds).
    %   - For each run compute FC matrix (corr of filtered BOLD) and add to running sum.
    %   - For FCD: compute windowed FC time series, then correlation matrix of those windows; collect LOWER triangle
    %     entries across runs into one pooled distribution (fcd_pool); DO NOT AVERAGE these matrices.
    %   - Objective:
    %       fit_fc: 1 - corr( avg_fc_lower_triangle , observable_lower_triangle )
    %       fit_fcd: ks statistic between pooled simulated distribution and observed distribution
    %       fit_both: 0.5 * fc_term + 0.5 * ks

    if ~(dmf_pars.with_decay & dmf_pars.with_plasticity)
        error("Error: Parameters not set to dynamic");
    end

    data = load("../data/fit_res_3-44.mat");
    a = data.fit_res(2);
    b = data.fit_res(1);

    opt_vars = [];
    stren = sum(dmf_pars.C);
    if numel(G)==2
        Gvals = optimizableVariable('G',[G(1) G(2)]);
        opt_vars = [opt_vars Gvals];
    end
    if numel(lr)==2
        lr = optimizableVariable('lr',[lr(1) lr(2)], 'Transform','log');
        opt_vars = [opt_vars lr];
    end

    results = bayesopt(@aux_dyn_fic_dmf,opt_vars,opts{:});

    function [out_error, const, outdata] = aux_dyn_fic_dmf(G_lr_params)
        const = [];
        thispars = dmf_pars;
        thispars.G = G_lr_params.G;
        thispars.lrj = G_lr_params.lr;
        thispars.taoj = exp(a + log(thispars.lrj) * b);
        thispars.lr_vector = thispars.lrj * ones(thispars.N,1);
        thispars.taoj_vector = thispars.taoj * ones(thispars.N,1);
        thispars.J = 0.75 * thispars.G * stren' + 1; % coupling matrix update

        isubfc_idx = find(tril(ones(thispars.N), -1));

        % Accumulators
        fc_sum = zeros(thispars.N, thispars.N);
        fcd_pool = [];
        mean_rates_runs = zeros(thispars.NSUB, thispars.N);

        for run = 1:thispars.NSUB
            % NOTE: If you want deterministic different seeds, set them here (e.g., rng(run + some_offset)).
            % (Optional) set different noise seeds if model supports it; assume handled inside dyn_fic_DMF
            [rates, ~, bold, ~] = dyn_fic_DMF(thispars, thispars.nb_steps);

            % Remove transient
            rates = rates(:, (thispars.burnout*thispars.TR/thispars.dtt):end);
            bold = bold(:, thispars.burnout:end);

            if isempty(bold)
                out_error = NaN; return; %#ok<NASGU>
            end
            bold(~isfinite(bold)) = 0;

            filt_bold = filter_bold(bold', thispars.flp, thispars.fhi, thispars.TR);
            sim_fc = corrcoef(filt_bold);
            if any(~isfinite(sim_fc(:)))
                sim_fc(~isfinite(sim_fc)) = 0;
            end
            fc_sum = fc_sum + sim_fc;

            % FCD windows and correlation-of-windows matrix
            sim_fcd_mat = compute_fcd(filt_bold, thispars.wsize, thispars.overlap, isubfc_idx);
            if isempty(sim_fcd_mat)
                continue; % skip this run; produces no FCD contribution
            end
            sim_fcd_mat(~isfinite(sim_fcd_mat)) = 0;
            sim_fcd_corr = corrcoef(sim_fcd_mat);
            sim_fcd_corr(~isfinite(sim_fcd_corr)) = 0;

            % Pool lower triangle entries as distribution samples (avoid diagonal/self & redundancy)
            fcd_pool = [fcd_pool; sim_fcd_corr(isubfc_idx)]; %#ok<AGROW>

            mean_rates_runs(run, :) = mean(rates,2)';
        end

        if isempty(fcd_pool)
            out_error = NaN; outdata = {[]}; return;
        end

        % Average FC across runs
        avg_fc = fc_sum / thispars.NSUB;

        % Observable handling: assume same shape; for FC we compare triangle; for FCD we kstest2 pooled distribution
        ks = NaN;
        if thispars.fit_fc || thispars.fit_both
            fc_obs = observable; % assume observable provided is an FC matrix when fit_fc true
            if size(fc_obs,1) ~= thispars.N || size(fc_obs,2) ~= thispars.N
                error('Observable FC dimensions mismatch');
            end
            fc_term = 1 - corr(avg_fc(isubfc_idx), fc_obs(isubfc_idx));
        else
            fc_term = 0;
        end

        if thispars.fit_fcd || thispars.fit_both
            % If observable is FCD distribution/matrix: accept either matrix or precomputed lower-tri vector
            if ismatrix(observable) && all(size(observable) == [thispars.N thispars.N]) && ~(thispars.fit_fc && ~thispars.fit_fcd)
                % ambiguous; assume FC not FCD. Require user to pass explicit FCD distribution when fitting FCD.
                % Fallback: treat observable as FC if ks fails.
                obs_fcd_dist = observable(isubfc_idx); % fallback assumption
            else
                obs_fcd_dist = observable(:); % treat as vector distribution
            end
            obs_fcd_dist = obs_fcd_dist(~isnan(obs_fcd_dist));
            try
                [~,~,ks] = kstest2(fcd_pool(:), obs_fcd_dist(:));
            catch
                ks = 1; % worst case
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
            out_error = NaN; % no target specified
        end

        mean_rates = mean(mean_rates_runs,1)';
        outdata = {mean_rates};
    end
end
    
    