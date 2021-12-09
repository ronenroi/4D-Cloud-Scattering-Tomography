function show_3d(ax,vol,my_title, max_val)
  if nargin<3
      max_val=200;
    end

    hold(ax,'on');
    xy_spacing = 0.01;
    z_spacing  = 0.03333;
    [ nx, ny, nz ] = size(vol);
    vol3d('cdata', vol,...
          'ydata', linspace(0, nx * xy_spacing, nx - 1),...
          'xdata', linspace(0, ny * xy_spacing, ny - 1),...
          'zdata', linspace(15*z_spacing, (nz+15) * z_spacing, nz - 1)); 
    view(3);
    grid on;
    axis equal;
%     caxis( [ 0, max(vol(:)) ] );
    caxis( [ 0, max_val ] );
    axis vis3d;
    xlabel('X [km]');
    ylabel('Y [km]');
    zlabel('Z [km]');
    colorbar;
    title(my_title, 'fontsize',26,'interpreter','latex' );
end

