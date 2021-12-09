clear;
close all;
clc;


% user = 'tamarl'; %work computer
user = 'tamar'; %laptop
addpath(genpath(strcat('C:\Users\', user, '\Google Drive\University\Master\Research\clouds\jpl\res')));
% addpath(genpath('no multiscale'));

shdom = load('SHDOM\result.mat');
beta_shdom = shdom.extinction;
our = load('multiscale\monoz\smooth\19_10_10_17_15\stages 8 lambda1 1 lambda2 1 w air and ocean 9 sensors 1024 photons and prior weight changes w scale SC mask beta0 2 iter_.mat');
beta_gt = our.beta_gt;
beta = our.beta;

xy_spacing = 0.02;
z_spacing  = 0.04;

cmax = ceil( max( [max(beta_gt(:)), max(beta(:)), max(beta_shdom(:))] ) );
    
[ nx, ny, nz ] = size(beta_gt);
x = 1 : 1 : nx;
y = 1 : 1 : ny;
z = 1 : 1 : nz;

xticks_v = 1 : nx : nx * xy_spacing;
yticks_v = 1 : ny : ny * xy_spacing;
zticks_v = 1 : nz : nz * z_spacing;

[ X, Y, Z ] = meshgrid(x, y, z);

figure;
axes1 = axes('Parent',gcf);
hold(axes1,'on');
xslice = 29;
yslice = 13;%23;%10
zslice = 14;%15;
h = slice(X, Y, Z, beta_gt, xslice, yslice, zslice);


axis on;

light;
axis tight
view(axes1,[-127.800935304773 32.4989717400132]);
camproj perspective
camlight('headlight');
grid on;
colormap(jet(64));
c = colorbar('vertical');
c.Label.String = '[1/km]';
caxis( [ 0, cmax ] );
xlabel('X [km]');
ylabel('Y [km]');
zlabel('Z [km]');
title('Ground truth \beta', 'fontsize', 20);
shading interp;
xticklabels({'0.2', '0.4', '0.6', '0.76'})
xticks([10, 20, 30, 38])
yticklabels({'0.2', '0.4', '0.6', '0.76'})
xticks([10, 20, 30, 38])
zticklabels({'0.4', '0.8', '1.2', '15.2'})
zticks([10, 20, 30, 38])
set(gca,'FontSize',16);


figure;
axes1 = axes('Parent',gcf);
hold(axes1,'on');

h = slice(X, Y, Z, beta, xslice, yslice, zslice);

axis on;

light;
axis tight
view(axes1,[-127.800935304773 32.4989717400132]);
camproj perspective
camlight('headlight');
grid on;
colormap(jet(64));
c = colorbar('vertical');
c.Label.String = '[1/km]';
caxis( [ 0, cmax ] );
xlabel('X [km]');
ylabel('Y [km]');
zlabel('Z [km]');
title('Our Output \beta', 'fontsize', 20);
shading interp;
xticklabels({'0.2', '0.4', '0.6', '0.76'})
xticks([10, 20, 30, 38])
yticklabels({'0.2', '0.4', '0.6', '0.76'})
xticks([10, 20, 30, 38])
zticklabels({'0.4', '0.8', '1.2', '15.2'})
zticks([10, 20, 30, 38])
set(gca,'FontSize',16);


figure;
axes1 = axes('Parent',gcf);
hold(axes1,'on');

h = slice(X, Y, Z, beta_shdom, xslice, yslice, zslice);

axis on;

light;
axis tight
view(axes1,[-127.800935304773 32.4989717400132]);
camproj perspective
camlight('headlight');
grid on;
colormap(jet(64));
c = colorbar('vertical');
c.Label.String = '[1/km]';
caxis( [ 0, cmax ] );
xlabel('X [km]');
ylabel('Y [km]');
zlabel('Z [km]');
title('SHDOM Output \beta', 'fontsize', 20);
shading interp;
xticklabels({'0.2', '0.4', '0.6', '0.76'})
xticks([10, 20, 30, 38])
yticklabels({'0.2', '0.4', '0.6', '0.76'})
xticks([10, 20, 30, 38])
zticklabels({'0.4', '0.8', '1.2', '15.2'})
zticks([10, 20, 30, 38])
set(gca,'FontSize',16);

