close all
clear all
%%
load('3sat/GT_3D_extinction.mat')
vol = squeeze(gt_extinction(4,:,:,15:25));

load('3sat/FINAL_3D_extinction20.mat')
vol1 = squeeze(estimated_extinction(:,:,15:25,4));

load('2sat/FINAL_3D_extinction20.mat')
vol2 = squeeze(estimated_extinction(:,:,15:25,4));


load('base_line/FINAL_3D_extinction.mat')
vol4 = squeeze(estimated_extinction(:,:,15:25,1));



%% 3d inter
% [nx,ny,nz]= size(vol);
% x=0:0.01:0.01*(ny-1);
% y = 0:0.01:0.01*(nx-1);
% Z = 0:0.033333:0.033333*(nz-1);
% [X,Y,Z] = meshgrid(x,y,Z);
% [Xq,Yq,Zq] = meshgrid(x,y,0:0.01:0.033333*(nz-1));
% 
% vol = interp3(X,Y,Z,vol,Xq,Yq,Zq);
figure
ax_mask = subplot(1,1,1);
show_3d(ax_mask,vol,"Ground-truth extinction", 140);
set(gca,'linewidth',2)
fig.Renderer='Painters'; 

% vol1 = interp3(X,Y,Z,vol1,Xq,Yq,Zq);
figure
ax_mask = subplot(1,1,1);
show_3d(ax_mask,vol1,"Recoverd extinction - 3 Satellites", 140);
set(gca,'linewidth',2)
fig.Renderer='Painters'; 


% vol2 = interp3(X,Y,Z,vol2,Xq,Yq,Zq);
figure
ax_mask = subplot(1,1,1);
show_3d(ax_mask,vol2,"Recoverd extinction - 2 Satellites", 140);
set(gca,'linewidth',2)
fig.Renderer='Painters'; 


% vol4 = interp3(X,Y,Z,vol4,Xq,Yq,Zq);
figure
ax_mask = subplot(1,1,1);
show_3d(ax_mask,vol4,"Recoverd extinction - Baseline", 140);
set(gca,'linewidth',2)
fig.Renderer='Painters'; 

%%
    eps1 = abs(vol-vol1);
%     eps1(eps1==Inf | isnan(eps1))=0;
    eps2 = abs(vol-vol2);
%     eps2(eps2==Inf | isnan(eps2))=0;
%     eps3(eps3==Inf | isnan(eps3))=0;
    eps4 = abs(vol-vol4);
%     eps4(eps4==Inf | isnan(eps4))=0;
    m1 = max(eps1(:));
    m2 = max(eps2(:));
    m=max([m1,m2]);
figure
    ax_mask = subplot(1,1,1);
    show_3d(ax_mask,eps1,"Recovery Error - 3 Satellites", m);
%     colorbar('off');
    figure
    ax_mask = subplot(1,1,1);
    show_3d(ax_mask,eps2,"Recovery Error - 2 Satellites", m);
%     colorbar('off');
    
%     colorbar('off');
    figure
    ax_mask = subplot(1,1,1);
        show_3d(ax_mask,eps4,"Recovery Error - Base-line", m);
%     colorbar('off');
% pos = get(gca, 'Position'); % gives x left, y bottom, width, height
% x = pos(1); y = pos(2); w = pos(3); h = pos(4);
% colorbar;
% set(gca, 'Position',  [x, y, w*1.1, h])

% %%
% figure
%     ax_mask = subplot(1,3,1);
%     show_3d_slices(ax_mask,eps1,"Error 3 Satellites - sigma=40[sec]", 5);
%     colorbar('off');
%     ax_mask = subplot(1,3,2);
%     show_3d_slices(ax_mask,eps2,"Error 2 Satellites - sigma=40[sec]", 5);
%     colorbar('off');
%     ax_mask = subplot(1,3,3);
%     show_3d_slices(ax_mask,eps3,"Error Single platform - sigma=40[sec]", 5);
%     colorbar('off');
    %%
    figure
    ax_mask = subplot(1,1,1);
    show_3d_slices(ax_mask,vol,'', max(vol(:))*1.1);
set(gca,'linewidth',2)
fig.Renderer='Painters'; 
figure
    ax_mask = subplot(1,1,1);
    show_3d_slices(ax_mask,vol1,'', max(vol(:))*1.1);
%     colorbar('off');
set(gca,'linewidth',2)
fig.Renderer='Painters'; 
figure
    ax_mask = subplot(1,1,1);
    show_3d_slices(ax_mask,vol2,'', max(vol(:))*1.1);
    set(gca,'linewidth',2)
fig.Renderer='Painters'; 
%     colorbar('off');

%     colorbar('off');
figure
    ax_mask = subplot(1,1,1);
    show_3d_slices(ax_mask,vol4,'', max(vol(:))*1.1);
    set(gca,'linewidth',2)
fig.Renderer='Painters'; 
%     colorbar('off');
    %%
figure
    ax_mask = subplot(1,1,1);
    show_3d_slices(ax_mask,eps1,'', 80);
        set(gca,'linewidth',2)
fig.Renderer='Painters'; 
%     colorbar('off');
figure
    ax_mask = subplot(1,1,1);
    show_3d_slices(ax_mask,eps2,'', 80);
        set(gca,'linewidth',2)
fig.Renderer='Painters'; 
%     colorbar('off');

%     colorbar('off');
figure
    ax_mask = subplot(1,1,1);
    show_3d_slices(ax_mask,eps4,'', 80);
        set(gca,'linewidth',2)
fig.Renderer='Painters'; 
%     colorbar('off');
%% scatter plot
mask = vol>0;
mask1 = vol1>0;
mask2 = vol2>0;
mask4 = vol4>0;
M = max(vol(:));
ind1 = mask1+mask;
ind2 = mask2+mask;
ind4 = mask4+mask;
n = 2000;


figure
% subplot(222)
vec1 = vol1(ind1>0);
vec = vol(ind1>0);
p = randperm(numel(vec));
p = p(1:n);
scatter(vec(p),vec1(p))
hold on 
plot([0,150],[0,150],'r--','LineWidth',2);axis tight;axis equal;
xlim([0 150])
ylim([0 150])
xlabel('$ ${\boldmath $\beta$}$^{\rm true}$','Interpreter','latex','FontSize',20)
ylabel('  { {\boldmath ${\hat \beta}$}} ','Interpreter','latex','FontSize',20)
title('3 Satellites', 'fontsize',16,'interpreter','latex' );
set(gca,'linewidth',2)

figure
% subplot(223)
vec2 = vol2(ind2>0);
vec = vol(ind2>0);
p = randperm(numel(vec));
p = p(1:n);
scatter(vec(p),vec2(p))
hold on 
plot([0,150],[0,150],'r--','LineWidth',2);axis tight;axis equal;
xlim([0 150])
ylim([0 150])
xlabel('$ ${\boldmath $\beta$}$^{\rm true}$','Interpreter','latex','FontSize',20)
ylabel('  { {\boldmath ${\hat \beta}$}} ','Interpreter','latex','FontSize',20)
title('2 Satellites', 'fontsize',16,'interpreter','latex' );

set(gca,'linewidth',2)

figure
% subplot(221)
vec4 = vol4(ind4>0);
vec = vol(ind4>0);
p = randperm(numel(vec));
p = p(1:n);
scatter(vec(p),vec4(p))
hold on 
plot([0,150],[0,150],'r--','LineWidth',2);axis tight;axis equal;
xlim([0 150])
ylim([0 150])
xlabel('$ ${\boldmath $\beta$}$^{\rm true}$','Interpreter','latex','FontSize',20)
ylabel('  { {\boldmath ${\hat \beta}$}} ','Interpreter','latex','FontSize',20)
title('Baseline', 'fontsize',16,'interpreter','latex' );

set(gca,'linewidth',2)


