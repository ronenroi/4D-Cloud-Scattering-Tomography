function show_3d_slices(ax,cloud,my_title, max_val)
xy_spacing = 0.01;
z_spacing  = 0.0333;

% cloud = permute(cloud,[2,1,3]);
cmax = ceil( max_val );
    
[ nx, ny, nz ] = size(cloud);
x = 0 : 1 : nx-1;
y = 0 : 1 : ny-1;
z = 0 : 1 : nz-1;

xticks_v = linspace(0, nx * xy_spacing, nx - 1);
yticks_v = linspace(0, ny * xy_spacing, ny - 1);
zticks_v = 1 : nz : nz * z_spacing;

[ X, Y, Z ] = meshgrid(y,x, z);

% figure;
% axes1 = axes('Parent',gcf);
hold(ax,'on');
xslice = 20;
yslice = 30;
zslice = 5;
h = slice(X, Y, Z, cloud, xslice, yslice, zslice);



axis on;
 set(gca,'DataAspectRatio',[1 1 0.33333])
light;
axis tight
% view(ax,[-127.800935304773 32.4989717400132]);
view(3);
camproj perspective
camlight('headlight');
grid on;
colormap(jet(64));
c = colorbar('vertical');
c.Label.String = '[1/km]';
caxis( [ 0, cmax ] );
xlabel('Y [km]');
ylabel('X [km]');
zlabel('Z [km]');
title(my_title);
shading interp;
xticklabels({'0', '0.1','0.2'})
xticks([1, 10, 20])
shading interp;
yticklabels({'0', '0.2', '0.4'})
yticks([1, 20, 40])
zticklabels({'0.5', '0.6', '0.7', '0.8', '0.9', '1'})
zticks([1, 4, 7, 10,13,16])

end