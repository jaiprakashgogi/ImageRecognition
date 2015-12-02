function render_hist( histogram_values )
%RENDER_HIST Summary of this function goes here
%   Detailed explanation goes here

    clf;
    hold on;
    br = bar(histogram_values(1, :), 1, 'r', 'EdgeColor', 'none');
    bg = bar(histogram_values(2, :), 1, 'g', 'EdgeColor', 'none');
    bb = bar(histogram_values(3, :), 1, 'b', 'EdgeColor', 'none');
    
    h = findobj(gca,'Type','patch');
    set(h, 'facealpha', 0.25);
    
    alpha(get(br, 'child'), 0.25);
    alpha(get(bg, 'child'), 0.25);
    alpha(get(bb, 'child'), 0.25);
end

