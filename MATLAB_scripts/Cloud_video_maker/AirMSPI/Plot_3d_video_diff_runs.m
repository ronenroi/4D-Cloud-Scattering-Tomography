

clear
% close all
clc
%%
scrsz=get(0,'ScreenSize');
tic


zen1=10;
zen2=10;
az1=-60;
az2=-60;

accurlwca=0;
aff=0;
ZBf=54;             % calc or determine cloub base level. if ZBf==0 takes to layers above the minimum layer with Qc>0.01g/kg else, Zb=ZBf
large3D=1;   % for com3D seperated to two files
RGBflag=false;
toplim=2000;
dtstep=60;
dt=0.5;
timef=dt/60;
timeN=2400:dtstep:9000;% snapshot number
LWCth=0.01;   % LWC threshold for cloud
Coli=156;        % snapshor cutting
Colf=356;
Ri=Coli;%65; % For taking only small part of the matrix in the y direction
Rf=Colf;%115;
DXY=10; 
dz=10;
InvZ=[150 150 150 100];

if ZBf==1
    bl=ZBf;
else
    bl=60;
end


% file='BOMEX_1CLD_512x512x320_500CCNlow_10m_7dT0.1';
files={'BOMEX_1CLD_512x320_500CCNblowB_10m_dT01_Tr3','BOMEX_1CLD_512x320_500CCNblowB_10m_dT001_Tr3','BOMEX_512x320_500CCNblowB_10m_dT01_Tr3_R250','BOMEX_512x320_500CCNblowB_10m_dT01_Tr3_InvL'};
titles={'Reference cloud','Weaker pertubation','Smaller cloud','Lower inversion'};
% file='BOMEX_1CLD_256x256x320_500CCNblowInv_10m_1sec';

PATH=['/home/labs/koren/eshkole/SAM/SAM_SBM/OUT_3D'];
% PATH='/data1/koren-public/WARM_simulations/SAM_OUTPUT/BIN/';

vidio2=VideoWriter(['Few_Sim_compare.avi']);
vid.Quality=100;
vidio2.FrameRate=1;
open(vidio2);

PCU='512';

a=2; % arrors steps
eps=0.622;  % Rd/Rv
g=9.8;  % gravity (m/s)
Rd=287; % Gas constant for dry air
Rvv=461; % Gas constant for water vapor  
cp=1000; % heat capacity J/kg
L= 2.5e6; % Latent heat g/kg





% h2= figure('Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)]);

%%

for bn=1:length(timeN)
Tsteplength=10-length(num2str(timeN(bn)));
fileN='0000000000';
fileN(Tsteplength+1:end)=num2str(timeN(bn));

h2= figure('Position',[scrsz(1) scrsz(2) scrsz(3) scrsz(4)]);


for ff=1:length(files)
% path=[PATH '/' file];

if ff==1 && bn>find(timeN==6720)
    continue
elseif ff>2 && bn>find(timeN==7200)
    continue
end

file=files{ff};
disp(file)
disp(fileN)
path=[PATH '/' file ];





    if large3D==1
[~, outstrct]=read_nc_file_struct([path '/' file '_' PCU '_' fileN '_1.nc']);
    else
[~, outstrct]=read_nc_file_struct([path '/' file '_' PCU '_' fileN '.nc']);
    end
    

Qp=outstrct.QP;   % g/kg
Qp=Qp(Ri:Rf,Coli:Colf,:);

Tr1=outstrct.TR01;   % g/kg
Tr1=Tr1(Ri:Rf,Coli:Colf,:);

% Tr3=outstrct.TR03;   % g/kg
% Tr3=Tr3(Ri:Rf,Coli:Colf,:);

if large3D==1
    [~, outstrct]=read_nc_file_struct([path '/' file '_' PCU '_' fileN '.nc']);
end    
    
tmp=outstrct.QN;   % g/kg
Qn=tmp(Ri:Rf,Coli:Colf,:);
% [~, outstrct]=read_nc_file_struct([path '/' file '.nc']);
% T=(outstrct.TABS(:,timeS));
% p=outstrct.p; 
% rho=p*100./(Rd*T);


LWCtot=Qn+Qp;   % g/kg
CldMtrx=LWCtot>LWCth;


if bn==1
sz=size(Qn);
SZ=size(tmp);
ms=SZ(1);%size(LWC,1);
ns=SZ(2);%size(LWC,2);
z=outstrct.z;


zlimtop=find(abs(z-toplim)==min(abs(z-toplim)));
zlimtop=zlimtop(1);
zlimbot=1;

x = 1:ns;  % size of the matrix
y = 1:ms;
z2 = 1:length(z);

xs=1:sz(2);
ys=1:sz(1);
zs=1:zlimtop;

end

%%
% Qv=outstrct.QV;   % g/kg
% Qv=Qv(Ri:Rf,Coli:Colf,:);
% T=outstrct.TABS;
% T=T(Ri:Rf,Coli:Colf,:);
% PP=outstrct.PP;   % g/kg
% PP=PP(Ri:Rf,Coli:Colf,:);
% P=outstrct.p; 
% 
% % u=outstrct.U; 
% % u=u(Ri:Rf,Coli:Colf,:);
% w=outstrct.W; 
% w=w(Ri:Rf,Coli:Colf,:);
% 
% % Density calculation
% p=repmat(permute(P*100,[2 3 1]),sz(1),sz(2),1)+PP;
% % Tdens=T.*((1+((Qv*1e-3)/eps))./(1+(Qv+Qn+Qp)*1e-3));
% RHOd=p./(Rd*T);
% 
% % cloud base calculation
% if ZBf==0
% [~,~,CldZs]=ind2sub(sz,find(CldMtrx));
% Zb=min(CldZs)+2;  % cloud base level
% else
% Zb=ZBf;
% end

%%

% load([path '/Bin_files/rho.mat']) %density

% LWC=LWC.*tmp; % g/kg -> g/m^3
% WV=Qv.*tmp; % g/kg -> g/m^3
% RR=RR.*tmp; % g/kg -> g/m^3
% LWCtot=LWCtot.*tmp; % g/kg -> g/m^3


%%  RH calculation

if accurlwca==true


% mixR=Qv*1e-3./(1-Qv*1e-3);   % mixing ratio
% E=(repmat(permute(p,[2 3 1]),Rf-Ri+1,Colf-Coli+1,1)+PP*(1e-2)).*mixR./(mixR+eps); % WV spesific prussure
% es=6.1094*exp(17.625*(T-273)./(T-273+243.04));
% RH=E./es*100;
% S=RH-100;
varname='SSW';    
    
    load([path '/Bin_files/' file '_' fileN '_' varname '_VALS.mat'],'VALS')
    load([path '/Bin_files/' file '_' fileN '_' varname '_IDX.mat'],'IDX')
    

   
    tmp=zeros(ms,ns,zs);
    if ~isnan(IDX(1))
        tmp(IDX)=VALS;
    end
    RH=(tmp(Ri:Rf,Coli:Colf,:,:)+100);
   
    
end


%%

%% adiabatic LWC
if aff~=0
if sum(CldMtrx(:))~=0 % no cloud so no AF
%% adiabatic LWC

[Corew,THw] = findCORE(w,CldMtrx,95);

coreprf=Corew;  % which core to consider for the adiabatic profile

rhov=(1e-3)*Qv.*RHOd;
rhoa=p./(Rd.*T);

A1= g./T.*((L/(cp*Rvv.*T))-1/Rd);
A2= 1/rhov + (L.^2)./(Rvv*cp.*(T.^2).*rhoa);

if accurlwca==1
S=(RH-100)/100;
S(~coreprf)=nan;
Score=squeeze(nanmean(nanmean(S)));
% dSdz= diff(Score)./diff(z);
% dSdz=[dSdz; 0];
end

tmp=A1;
tmp(~coreprf)=nan;
A1z=squeeze(nanmean(nanmean(tmp)));

tmp=A2;
tmp(~coreprf)=nan;
A2z=squeeze(nanmean(nanmean(tmp)));

% dLWCdz=(A1z(1:end-1)-dSdz)./A2z(1:end-1);
dLWCdz=(A1z)./A2z;


% LWC0=Qn(:,:,Zb,t).*RHO(:,:,Zb,t)*1e-3; % kg/kg
% LWC0(~CldMtrx(:,:,Zb))=nan;
% LWC0=nanmean(LWC0(:));
if accurlwca==1
S0=S(:,:,Zb);
S0=nanmean(S0(:));
end

if aff==2
Qt0=Qn(:,:,Zb)+Qp(:,:,Zb)+Qv(:,:,Zb);
Qt0(CldMtrx(:,:,Zb)==0)=nan;
Qt0=nanmean(Qt0(:));
qte = (Qn+Qp+Qv);
qte(CldMtrx)=nan;
qte(D>100)=nan;   % Qt_e is the environment that mixes (i.e. 100m close to the cloud AND NOT THE FAR FIELD) 
qte=nanmean(nanmean(qte));

% Qtf=((Qn+Qp+Qv)-qte)./(repmat(Qt0,ns,ms,zs)-qte);
Qtf=((Qn+Qp+Qv))./(repmat(Qt0,ns,ms,zs));

Qtf(~CldMtrx)=nan;
end

tmp=nan([sz(3) 1]);
tmp(Zb:end)=dLWCdz(Zb:end);
% saLWCa=LWC0+tmp.*(z(1:end)-z(Zb));

if accurlwca==true
C=log((S0+1)./(Score+1))./A2z;
else
    C=0;
end
LWCa = tmp.*(z(1:end)-z(Zb))+C ;% + LWC0; % + permute(max(max(Qn*1e-3)),[3 2 1]); %  
    LWCf=(1e-3)*(Qn+Qp).*RHOd./repmat(permute(LWCa,[3 2 1]),sz(1),sz(2),1);
    LWCf(~CldMtrx)=nan;




tmp=sum(LWCf(~isnan(LWCf))>1.5);
if tmp > 0.1*numel(LWCf(~isnan(LWCf)))
    disp('LWCf>1.5 > 10% of the data')
else
LWCf(LWCf>2)=nan;
end


if aff==1
    %tmp=(max(max(w,[],1),[],2)); AF=w./repmat(tmp,512,512,1);
    AF=LWCf;
elseif aff==2
    AF=Qtf;
end


end

end
%%


if sum(CldMtrx(:))~=0
CORE= Tr1>=0.9; % 3D core mask
CORE(~CldMtrx)=false;
else
CORE=nan(sz);
end

alpha=0.3; % transperancy of cloud 


subplot(2,2,ff)
plot3DCld(CldMtrx(ys,xs,zs),CORE(ys,xs,zs),x(xs),y(ys),z2(zs),[az1 zen1],alpha)

pch=patch([1 1 length(xs) length(xs)],[1 length(ys) length(ys) 1],[InvZ(ff) InvZ(ff) InvZ(ff) InvZ(ff)],'red');
set(pch,'FaceAlpha',0.2)

% pch=patch([1 1 length(xs) length(xs)],[1 length(ys) length(ys) 1],bl*ones(1,4),'green');
% set(pch,'FaceAlpha',0.1)

title(titles{ff})

if ff==2
tt=text(10,10,[num2str(timeN(bn)*timef) ' min']);
set(tt,'FontSize',30,'Units','normalized','Position',[-1 1 1.35],'FontWeight','Bold');
end

xlim([1 200])
ylim([1 200])
zlim([1 250])
xticks(40:20:260)
xticklabels({'400','','','','1200','','','','2000','','2400',''})
yticks(40:20:260)
yticklabels({'400','','','','1200','','','','2000','','2400',''})
zticks(40:20:260)
zticklabels({'400','','800','','1200','','1600','','2000','','2400',''})

set(gca,'LineWidth',2,'FontSize',25)

clear CldMtrx CORE Qn Qp

pause(5)

end

frame2=getframe(h2);

try
writeVideo(vidio2,frame2);
catch ME
end

disp([num2str(timeN(bn)*timef) ' min'])

pause(5)

close(h2);
pause(3)

end

toc

close(vidio2);







