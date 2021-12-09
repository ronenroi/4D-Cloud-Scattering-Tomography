%% airmspi volume video
close all
clear
v = VideoWriter('CVPR_AirMSPI_video.avi', 'Motion JPEG AVI');
v.FrameRate = 20;

load('../../AirMSPI/FINAL_3D_extinction.mat')
vol = estimated_extinction;

t = [4, 30 52 74 95 113 133 150 169 187 204 222 240 257 274 291 310 330 350 372 395];
t = t - min(t);
dt = 1;
Nt = round((t(end)+10)/dt);
size_vol = size(vol);
size_vol1 = size_vol(1:3);
index = (1:prod(size_vol(1:3)))';
flat_cloud = reshape(estimated_extinction,prod(size_vol(1:3)),size_vol(4));
[indexi,ti] = meshgrid(index, t);

time = linspace(t(1),t(end)+10,Nt+1);
%%
figure('units','pixels','position',[0 0 1920 1080]) 
set(gca,'linewidth',20)
% fig.Renderer='Painters';

open(v);
for kathy=1:size(time,2)
    clf
curr_time = time(kathy);
    if curr_time< t(end)
        [indexq,tq] = meshgrid(index, curr_time);
        flat_cloudi = interp2(indexi,ti,flat_cloud',indexq,tq)'; 
%         size_vol(4) = length(curr_time);
        curr_cloud = reshape(flat_cloudi,size_vol1);
    else
        flat_cloudi=flat_cloud(:,end);
        curr_cloud = reshape(flat_cloudi,size_vol1);

    end
        ax_mask = subplot(1,1,1);
        hold on

    show_3d(ax_mask,curr_cloud,"Recoverd extinction - AirMSPI data", 20);
tex = annotation('textbox', [0.539017808958447,0.749748743718599,0,0], 'string', ['Time=' ...
 num2str(round(curr_time)) '[sec]']);
tex.FontSize = 24;
tex.FontSize = 30;
tex.FitBoxToText = 'on';
tex.Margin = 0
tex.EdgeColor = 'r';
tex.LineWidth = 2;
tex.VerticalAlignment = 'middle';
tex.HorizontalAlignment = 'cente';
    ax = gca;
    ax.YAxis.FontSize = 24;
    ax.XAxis.FontSize = 24;
ax.ZAxis.FontSize = 24;
c = colorbar;
c.Label.String = 'Extinction [1/km]';
c.FontSize=20;
c.Label.Position = [1.142857131503877,21.319368353019364,0];
c.Label.Rotation = 0;
    drawnow;


    frame = getframe(gcf);
    writeVideo(v,frame);
    
end
close(v);

%%
% figure
% ax_mask = subplot(1,1,1);
% show_3d(ax_mask,vol(:,:,:,1),"Recoverd extinction", 20);
% set(gca,'linewidth',2)
% fig.Renderer='Painters';
% 
% open(v);
% set(gca,'nextplot','replacechildren');
% for i=2:21
%     clf
%     ax_mask = subplot(1,1,1);
%     
%     show_3d(ax_mask,vol(:,:,:,i),"Recoverd extinction", 20);
%     frame = getframe(gcf);
%     writeVideo(v,frame);
%     
% end
% close(v);
% % %%
% % v = VideoWriter('Eshkol_sim.avi');
% % v.FrameRate = 1;
% % estimated_extinction=reff;
% % size_vol = size(estimated_extinction);
% % dx=1;
% % dy=1;
% % dz=1;
% % open(v);   close all
% % figure
% % for i=1:size_vol(4)
% %     
% %     clf
% %     vol3d('cdata', estimated_extinction(:,:,:,i), 'xdata', [0 dx*size_vol(1)], 'ydata', [0 dy*size_vol(2)], 'zdata', [0 dz*size_vol(3)]);
% %     colormap(bone(256))
% %     alphamap([0 linspace(0.1, 0, 255)]);
% %     axis equal off
% %     set(gcf, 'color', 'w');
% %     set(gca,'nextplot','replacechildren');
% %     
% %     view(66,37);
% %     frame = getframe(gcf);
% %     
% %     writeVideo(v,frame);
% %     
% % end
% % close(v);
% % %%
% % close all
% % size_vol = size(estimated_extinction);
% % flat_cloud = reshape(estimated_extinction,prod(size_vol(1:3)),size_vol(4));
% % % flat_cloud(min(flat_cloud,[],2)>0.1,:)=[];
% % flat_cloud=flat_cloud(min(flat_cloud(60:end),[],2)>0.01,:);
% % ind = randperm(size(flat_cloud,1));
% % i=1:5;
% % plot(flat_cloud(i,:)');
% % hold on
% % plot(60:10:126,flat_cloud(i,60:10:end)','ro-');
% % 
% % 
