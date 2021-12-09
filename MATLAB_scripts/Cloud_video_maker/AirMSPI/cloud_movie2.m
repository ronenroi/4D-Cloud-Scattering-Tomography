%%
% t=[  1.79174059 , 67.5839426,  130.4524017 , 192.81670056, 198.27509336...
%  256.3873515  313.4972616  368.33325488 423.74883288];
% t=[  1.79174059 ,  130.4524017 , 192.81670056, 198.27509336...
%  256.3873515  313.4972616  368.33325488 423.74883288];
t=0:20:20*20;
t(4)=[];

dx = x(2,1)-x(1,1);
dy = y(2,1)-y(1,1);
dz = z(2,1)-z(1,1);
OptionZ.FrameRate=15;OptionZ.Duration=5.5;OptionZ.Periodic=true;
OptionZ.FrameRate=10;OptionZ.Duration=10;OptionZ.Periodic=true;
CaptureFigVid([-250,50;-250,50], 'StillCamera_21_new_views2',OptionZ,estimated_extinction,dx,dy,dz,t)
% CaptureFigVid([-20,10;-110,10;-190,80;-290,10;-380,10], 'WellMadeVid',OptionZ,estimated_extinction,dx,dy,dz,t)
% CaptureFigVid([-190,80;-290,10], 'MovingCamera_Eshkol_dt5',OptionZ,estimated_extinction,dx,dy,dz,t)
%% show gt
t=0:5:32*5;
a = full(extinction(:,79:(79+4*8)));
cloud = reshape(a,320,502,502,33);
cloud = permute(cloud,[3,2,1,4]);
cloud1 = cloud(357:400,340:370,1:150,:);
dx = 10/1000;
dy = 10/1000;
dz = 10/1000;
OptionZ.FrameRate=10;OptionZ.Duration=10;OptionZ.Periodic=true;
CaptureFigVid([-250,50;-250,50], 'StillCamera_Eshkol_dt5_gt',OptionZ,cloud1,dx,dy,dz,t)
%% show gt
OptionZ.FrameRate=10;OptionZ.Duration=30;OptionZ.Periodic=true;
ViewZ = [-250,50;-250,50];
figure();clf;
daObj=VideoWriter('Opening_cloud1'); %my preferred format
if isfield(OptionZ,'FrameRate')
    daObj.FrameRate=OptionZ.FrameRate;
end
% Durration (if frame rate not set, based on default)
if isfield(OptionZ,'Duration') %space out view angles
    temp_n=round(OptionZ.Duration*daObj.FrameRate); % number frames
    temp_p=(temp_n-1)/(size(ViewZ,1)-1); % length of each interval
    ViewZ_new=zeros(temp_n,2);
    % space view angles, if needed
    for inis=1:(size(ViewZ,1)-1)
        ViewZ_new(round(temp_p*(inis-1)+1):round(temp_p*inis+1),:)=...
            [linspace(ViewZ(inis,1),ViewZ(inis+1,1),...
             round(temp_p*inis)-round(temp_p*(inis-1))+1).',...
             linspace(ViewZ(inis,2),ViewZ(inis+1,2),...
             round(temp_p*inis)-round(temp_p*(inis-1))+1).'];
    end
    ViewZ=ViewZ_new;
end
open(daObj);

for i=1:126
a = full(extinction(:,i));
cloud = reshape(a,320,502,502);
cloud = permute(cloud,[3,2,1]);
% cloud1 = cloud(357:400,340:370,1:150);
dx = 10/1000;
dy = 10/1000;
dz = 10/1000;
OptionZ.FrameRate=10;OptionZ.Duration=10;OptionZ.Periodic=true;
volshow(cloud,cloud_config,'ScaleFactors',[3,3,3]) 
drawnow;
    writeVideo(daObj,getframe(gcf));
end
close(daObj);