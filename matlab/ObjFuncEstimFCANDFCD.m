close all; clear; clc;

num_points_y = 100;
num_points_x = 60;
G_space = linspace(0.1, 16, num_points_x);
topPct  = 0.1;

%% Begin: Set folder for figures
figFolder = 'AAL_NVC';
if ~exist(fullfile('../plots', figFolder), 'dir')
    mkdir(fullfile('../plots', figFolder));
end
%% End: Set folder for figures

% ---- Dynamic (lr, log y)
compareFCandFCD( ...
    '../results/dyn_fcd/AAL_NVC_dyn_fcd.mat', ...
    '../results/dyn_fc/AAL_NVC_dyn_fc.mat', ...
    'lr', logspace(0, 4, num_points_y), 'Learning Rate', 'log', ...
    G_space, topPct, 'Dynamic AAL', figFolder);

% ---- Static (alpha, linear y)
compareFCandFCD( ...
    '../results/dyn_fcd/HCP_NVC_dyn_fcd.mat', ...
    '../results/dyn_fc/HCP_NVC_dyn_fc.mat', ...
    'lr', logspace(0, 4, num_points_y), 'Learning Rate', 'log', ...
    G_space, topPct, 'Dynamic HCP', figFolder);

% ======================================================================
function compareFCandFCD(fileFCD, fileFC, yVar, ySpace, yLabel, yScaleType, G_space, topPct, tagTitle, figFolder)
    [FCD, XminFCD, XminEstFCD] = getSurfaceAndMask(fileFCD, yVar, ySpace, G_space, topPct);
    [FC,  XminFC,  XminEstFC ] = getSurfaceAndMask(fileFC,  yVar, ySpace, G_space, topPct);

    %% FCD Figure
    fprintf('%s FCD: min obj=%.6g at G=%.6g, %s=%.6g\n', tagTitle, min(FCD.Z(:)), XminFCD.G, yVar, XminFCD.(yVar));    
    figure; ax = axes; hold(ax,'on');
    contourf(ax, G_space, ySpace, FCD.Z', 30, 'LineStyle','none');
    finishAxis(ax, yScaleType, 'G', yLabel, sprintf('%s — FCD', tagTitle));
    contour(ax, G_space, ySpace, FCD.mask', [1 1], 'w', 'LineWidth', 1.5);
    plot(ax, XminFCD.G,      XminFCD.(yVar),      'rx', 'MarkerSize',10, 'LineWidth',2, 'DisplayName','XAtMin');
    plot(ax, XminEstFCD.G,   XminEstFCD.(yVar),   'bo', 'MarkerSize',10, 'LineWidth',2, 'DisplayName','XAtMinEst');
    legend(ax,'show','Location','best'); hold(ax,'off');
    % Save FCD figure
    saveas(gcf, fullfile('../plots', figFolder, sprintf('%s_FCD_contour.png', tagTitle)));
    saveas(gcf, fullfile('../plots', figFolder, sprintf('%s_FCD_contour.fig', tagTitle)));

    %% FC Figure
    fprintf('%s FC: min obj=%.6g at G=%.6g, %s=%.6g\n', tagTitle, min(FC.Z(:)), XminFC.G, yVar, XminFC.(yVar));
    figure; ax = axes; hold(ax,'on');
    contourf(ax, G_space, ySpace, FC.Z', 30, 'LineStyle','none');
    finishAxis(ax, yScaleType, 'G', yLabel, sprintf('%s — FC', tagTitle));
    contour(ax, G_space, ySpace, FC.mask', [1 1], 'w', 'LineWidth', 1.5);
    plot(ax, XminFC.G,       XminFC.(yVar),       'rx', 'MarkerSize',10, 'LineWidth',2, 'DisplayName','XAtMin');
    plot(ax, XminEstFC.G,    XminEstFC.(yVar),    'bo', 'MarkerSize',10, 'LineWidth',2, 'DisplayName','XAtMinEst');
    legend(ax,'show','Location','best'); hold(ax,'off');
    % Save FC figure
    saveas(gcf, fullfile('../plots', figFolder, sprintf('%s_FC_contour.png', tagTitle)));
    saveas(gcf, fullfile('../plots', figFolder, sprintf('%s_FC_contour.fig', tagTitle)));

    %% Overlap Figure
    bothMask = FCD.mask & FC.mask;
    figure; ax = axes; hold(ax,'on');
    contourf(ax, G_space, ySpace, bothMask' + FC.mask' + FCD.mask', -0.5:1:3.5, 'LineStyle','none');
    finishAxis(ax, yScaleType, 'G', yLabel, sprintf('%s — Overlap (top %g%%)', tagTitle, 100*topPct));
    colormap(ax, [1 1 1; 0.85 0.35 0.35; 0.35 0.35 0.85; 0.25 0.7 0.25]);
    caxis(ax, [-0.5 3.5]);
    cb = colorbar(ax); cb.Ticks = 0:3; cb.TickLabels = {'None','FCD only','FC only','Overlap'};
    contour(ax, G_space, ySpace, FCD.mask', [1 1], 'r', 'LineWidth', 1.0);
    contour(ax, G_space, ySpace, FC.mask',  [1 1], 'b', 'LineWidth', 1.0);
    if any(bothMask(:))
        Z = FCD.Z; Z(~bothMask) = Inf;
        [minVal, idx] = min(Z(:));
        [iG, jY] = ind2sub(size(Z), idx);
        gStar = G_space(iG); yStar = ySpace(jY);
        plot(ax, gStar, yStar, 'kd', 'MarkerSize',9, 'LineWidth',1.5, 'MarkerFaceColor','w', 'DisplayName','Overlap-min (FCD)');
        legend(ax,'show','Location','best');
        fprintf('%s overlap-min (FCD): G=%.6g, %s=%.6g, FCD=%.6g\n', tagTitle, gStar, yVar, yStar, minVal);
    else
        fprintf('%s: no FC∩FCD overlap at top %g%%.\n', tagTitle, 100*topPct);
    end
    hold(ax,'off');
    % Save Overlap figure
    saveas(gcf, fullfile('../plots', figFolder, sprintf('%s_Overlap_contour.png', tagTitle)));
    saveas(gcf, fullfile('../plots', figFolder, sprintf('%s_Overlap_contour.fig', tagTitle)));
end

function finishAxis(ax, yScaleType, xlab, ylab, ttl)
    set(ax,'YDir','normal');
    if strcmpi(yScaleType,'log'), set(ax,'YScale','log'); end
    xlabel(ax, xlab); ylabel(ax, ylab); title(ax, ttl);
    grid(ax,'on'); colorbar(ax);
end

function [S, Xmin, XminEst] = getSurfaceAndMask(resultFile, yVar, ySpace, G_space, topPct)
    data = load(resultFile);
    if     isfield(data,'results'),          bo = data.results;
    elseif isfield(data,'BayesoptResults'),  bo = data.BayesoptResults;
    else,  error('Unexpected structure in %s', resultFile);
    end
    Xmin    = bo.XAtMinObjective;
    XminEst = bo.XAtMinEstimatedObjective;

    [Y, G] = meshgrid(ySpace, G_space);
    grid_points = table(Y(:), G(:), 'VariableNames', {yVar,'G'});
    o = predictObjective(bo, grid_points);
    Z = reshape(o, numel(G_space), numel(ySpace));

    sv  = sort(Z(:));
    thr = sv(max(1, round(topPct * numel(sv))));
    S.Z = Z;
    S.mask = (Z <= thr);
end
