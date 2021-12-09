function show_3d(ax,vol,my_title, max_val)
    if nargin<3
      max_val=200;
    end

    hold(ax,'on');
    x_spacing = 0.0188;
    y_spacing = 0.0243;
    z_spacing  = 0.025;
    [ nx, ny, nz ] = size(vol);
    vol3d('cdata', vol,...
          'ydata', linspace(0, nx * x_spacing, nx - 1),...
          'xdata', linspace(0, ny * y_spacing, ny - 1),...
          'zdata', linspace(15*z_spacing, (nz+15) * z_spacing, nz - 1)); 
    view(3);
    grid on;
    axis equal;
%     caxis( [ 0, max(vol(:)) ] );
    caxis( [ 0, max_val ] );
    axis vis3d;
    xlabel('X [km]', 'FontSize', 24);
    ylabel('Y [km]', 'FontSize', 24);
    zlabel('Z [km]', 'FontSize', 24);
    colorbar;
    title(my_title, 'fontsize',40,'interpreter','latex' );
end

