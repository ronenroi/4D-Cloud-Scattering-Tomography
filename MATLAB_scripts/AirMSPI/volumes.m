close all
clear all
%%
load('FINAL_3D_extinction.mat')
vol1 = squeeze(estimated_extinction(:,:,:,11));



%% 3d inter
% [nx,ny,nz]= size(vol);
% x=0:0.01:0.01*(ny-1);
% y = 0:0.01:0.01*(nx-1);
% Z = 0:0.033333:0.033333*(nz-1);
% [X,Y,Z] = meshgrid(x,y,Z);
% [Xq,Yq,Zq] = meshgrid(x,y,0:0.01:0.033333*(nz-1));
% 
% vol = interp3(X,Y,Z,vol,Xq,Yq,Zq);


% vol1 = interp3(X,Y,Z,vol1,Xq,Yq,Zq);
figure
ax_mask = subplot(1,1,1);
show_3d(ax_mask,vol1,"Recoverd extinction", 20);
set(gca,'linewidth',2)
fig.Renderer='Painters'; 


%% scatter plot
n = 200;
load('sigma_60_3.mat')

figure
input_image = input_image(:);
interp_image = interp_image(:);
p = randperm(numel(input_image));
p = p(1:n);
scatter(input_image(p),interp_image(p))
hold on 
plot([0,0.1365],[0,0.1365],'r--','LineWidth',2);axis tight;axis equal;
xlim([0 0.1365])
ylim([0 0.1365])
xlabel('  Captured image $+54^\circ$ view','Interpreter','latex','FontSize',20)
ylabel('  Re-projected  $+54^\circ$ view','Interpreter','latex','FontSize',20)

load('static_3.mat')
estimated_image = estimated_image(:);
scatter(input_image(p),estimated_image(p),'+');

set(gca,'linewidth',2)

load('sigma_60_10.mat')

figure
input_image = input_image(:);
interp_image = interp_image(:);
p = randperm(numel(input_image));
p = p(1:n);
scatter(input_image(p),interp_image(p))
hold on 
plot([0,0.1365],[0,0.1365],'r--','LineWidth',2);axis tight;axis equal;
xlim([0 0.08])
ylim([0 0.08])
xlabel('  Captured image nadir view','Interpreter','latex','FontSize',20)
ylabel('  Re-projected  nadir view','Interpreter','latex','FontSize',20)
load('static_10.mat')
estimated_image = estimated_image(:);
scatter(input_image(p),estimated_image(p),'+');
load('sigma_60_17.mat')

set(gca,'linewidth',2)
figure
input_image = input_image(:);
interp_image = interp_image(:);
p = randperm(numel(input_image));
p = p(1:n);
scatter(input_image(p),interp_image(p))
hold on 
plot([0,0.1365],[0,0.1365],'r--','LineWidth',2);axis tight;axis equal;
xlim([0 0.08])
ylim([0 0.08])
xlabel('  Captured image $-54^\circ$ view','Interpreter','latex','FontSize',20)
ylabel('  Re-projected  $-54^\circ$ view','Interpreter','latex','FontSize',20)
load('static_17.mat')
estimated_image = estimated_image(:);
scatter(input_image(p),estimated_image(p),'+');

set(gca,'linewidth',2)






