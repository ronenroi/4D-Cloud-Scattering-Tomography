clc;
close all;
GV = [-140,40];
% load lwc recs.

ind = 3;
gt = load('GT_high_res_cloud.mat');
gt = gt.extinction;
gt = squeeze(gt(ind,:,:,:));
recon = load('log_name-18-Oct-2020-17:56:32_sim_eshkol_reg_weighted_extinction.mat');
recon = recon.estimated_extinction;
recon = recon(:,:,:,ind);
[nx , ny, nz] = size(recon);
delta_x = 10;
delta_y = 10;
delta_z = 34.1;

Lx = delta_x*nx;
H = delta_z*(nz-15);
Ly = delta_y*ny;
Zbottom = 17*delta_z ;
H = 600;
svmin = [-0.5 * Lx, -0.5 * Ly,  Zbottom ];
svmax = [ 0.5 * Lx,  0.5 * Ly,  Zbottom + H];

x= linspace(svmin(1)+(delta_x/2),svmax(1)-(delta_x/2),nx);
y= linspace(svmin(2)+(delta_y/2),svmax(2)-(delta_y/2),ny);
z= linspace(svmin(3),svmax(3),nz);

x=linspace(0,Lx,nx);
y=linspace(0,Ly,ny);


[X,Y,Z] = meshgrid(y,x,z);



figure;
mask_gt = gt>0;
gt = gt(mask_gt);
recon = recon(mask_gt);
Z=Z(mask_gt);
N = numel(gt);
N = sum(mask_gt(:));
k = round(0.4*N);
p = randperm(N,k);

subplot(1,1,1);

scatter(gt(p),recon(p),[],Z(p),'filled')
xlabel('Ground Truch Extinction');
ylabel('Estimated Extinction');
title('Extinction scattter plot');
hold on;colorbar;
plot([0,100],[0,100],'r--');axis tight;axis equal;
max_ = max(max(gt(:)), max(recon(:)));
min_ = min(min(gt(:)), min(recon(:)));
xlim([min_,max_]);
ylim([min_,max_]);


%%
clc;
close all;
    figure;

GV = [-140,40];
% load lwc recs.
    gt = load('GT_high_res_cloud.mat');
    gt_all = gt.extinction;
        recon = load('log_name-18-Oct-2020-17:56:32_sim_eshkol_reg_weighted_extinction.mat');
    recon_all = recon.estimated_extinction;
for ind=1:9


    gt = squeeze(gt_all(ind,:,:,:));
    recon = recon_all(:,:,:,ind);
    [nx , ny, nz] = size(recon);
    delta_x = 10;
    delta_y = 10;
    delta_z = 34.1;

    Lx = delta_x*nx;
    H = delta_z*(nz-15);
    Ly = delta_y*ny;
    Zbottom = 17*delta_z ;
    H = 600;
    svmin = [-0.5 * Lx, -0.5 * Ly,  Zbottom ];
    svmax = [ 0.5 * Lx,  0.5 * Ly,  Zbottom + H];

    x= linspace(svmin(1)+(delta_x/2),svmax(1)-(delta_x/2),nx);
    y= linspace(svmin(2)+(delta_y/2),svmax(2)-(delta_y/2),ny);
    z= linspace(svmin(3),svmax(3),nz);

    x=linspace(0,Lx,nx);
    y=linspace(0,Ly,ny);


    [X,Y,Z] = meshgrid(y,x,z);



    mask_gt = gt>0;
    gt = gt(mask_gt);
    recon = recon(mask_gt);
    Z=Z(mask_gt);
    N = numel(gt);
    N = sum(mask_gt(:));
    k = round(0.04*N);
    p = randperm(N,k);

    subplot(1,1,1);
hold on
    scatter(gt(p),recon(p),[],recon(p)*0+ind,'filled')
    xlabel('Ground Truth Extinction');
    ylabel('Estimated Extinction');
    title('Extinction scattter plot');
    hold on;
%     colorbar;
    plot([0,120],[0,120],'r--');axis tight;axis equal;
    max_ = max(max(gt(:)), max(recon(:)));
    min_ = min(min(gt(:)), min(recon(:)));
    xlim([min_,max_]);
    ylim([min_,max_]);
    xlim([0,120]);
    ylim([0,120]);

end






