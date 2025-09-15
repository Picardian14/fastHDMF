function [results] = dynamic_fitting(G, lr,dmf_pars,opts, observable)
    %FINDHOMEOSTASIS find a G and lr that where target obj_rate is met
    
    % If parameters are not set to static, fail
    if ~(dmf_pars.with_decay & dmf_pars.with_plasticity)
        error("Error: Parameters not set to dynamic");                
    end

    data = load("data/fit_res_3-44.mat");
    a = data.fit_res(2);
    b = data.fit_res(1);
    % if 2x1, optimized within bounds, otherwise dont optimize, use fix value provided in dmf_pars
    % fitting variables could include G and OBJ_RATE in the future
    opt_vars = [];
    stren = sum(dmf_pars.C);
    if length(G(:))==2 
        Gvals = optimizableVariable('G',[G(1) G(2)]);
        opt_vars = [Gvals];
    end
    
    if length(lr(:))==2 
        lr = optimizableVariable('lr',[lr(1) lr(2)], 'Transform','log');
        opt_vars = [opt_vars lr];
    end
    
    results = bayesopt(@aux_dyn_fic_dmf,opt_vars,opts{:});

    function [out_error, const, outdata] = aux_dyn_fic_dmf(G_lr_params)
        const = [];
        % set arguments for this step
        thispars = dmf_pars;                
        thispars.G = G_lr_params.G;
        ones_vector = ones(thispars.N,1);
        thispars.lr_vector = ones_vector*G_lr_params.lr;                          
        taoj = exp(a+log(thispars.lrj)*b);
        thispars.taoj_vector = ones_vector*taoj;
        % save a safe copy to send to dyn_fic function
        thispars.J = 0.75*thispars.G*stren' + 1; % updates it
        %seed_range = thispars.NSUB*thispars.seed:thispars.NSUB*(thispars.seed+1);
        all_sim_fc = zeros(thispars.NSUB,thispars.N, thispars.N);
        all_sim_fcd = zeros(thispars.NSUB, thispars.nwins,thispars.nwins);
        all_rates = zeros(thispars.NSUB,thispars.N);
        isubfc = find(tril(ones(thispars.N),-1));
        %for idx=1:length(seed_range)
            %thispars.seed = seed_range(idx);            
            [rates, rates_inh, bold, fic_t] = dyn_fic_DMF(thispars, thispars.nb_steps);
            % takeout transient simulation
            rates = rates(:, (thispars.burnout*thispars.TR/thispars.dtt):end);
            %all_rates(idx, :) = mean(rates, 2);
            bold = bold(:,thispars.burnout:end); % remove initial transient
            bold(isnan(bold))=0;
            bold(isinf(bold(:)))=max(bold(~isinf(bold(:))));
            if isempty(bold)      
                disp("G: "+params.G+" LR: "+params.lrj+" Gave empty bold");
                out_error = nan;
                return
            end
            % Filtering and computing FC
            filt_bold = filter_bold(bold',thispars.flp,thispars.fhi,thispars.TR);
            % compute out_errors
            

            if thispars.fit_fc
                sim_fc = corrcoef(filt_bold);
                %all_sim_fc(idx, :, :) = sim_fc;
            elseif thispars.fit_fcd                
                sim_fcd = compute_fcd(filt_bold,thispars.wsize,thispars.overlap,isubfc);
                sim_fcd(isnan(sim_fcd))=0;
                sim_fcd = corrcoef(sim_fcd);
                if (size(observable, 2)~=size(sim_fcd,2))
                     error("not same size FCD")
                end
                if isempty(sim_fcd)                    
                    sim_fcd = zeros(size(sim_fcd));
                    return
                end
                %all_sim_fcd(idx, :, :) = sim_fcd;
            else
                disp('error: target observable not set')
                out_error=nan; 
            end
        %end
        if thispars.fit_fc
            %mean_fc = mean(all_sim_fc, 1);
            % SOLO ESTOY COMPARANDO CON 1 FC
            out_error = 1-corr2(sim_fc(isubfc),observable(isubfc));
        elseif thispars.fit_fcd             
            % SOLO ESTOY COMPARANDO CON 1 FC      
            try
                [~,~,out_error] = kstest2(sim_fcd(:),observable(:));
            catch 
                disp("G: "+ thispars.G);
                disp("LR: "+ thispars.lrj);
                if isempty(sim_fcd)                    
                    disp("FCD Was empty ");
                end
                out_error=1;
            end
        else
            disp('error: target observable not set')
            out_error=nan;      
        end
        %mean_rates = mean(all_rates,1);
        mean_rates = mean(rates,2);
        outdata = {mean_rates};
    end
    end
    
    