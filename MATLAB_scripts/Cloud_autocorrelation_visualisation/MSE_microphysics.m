%% LWC REFF MSE + Correlation
load('matlab_lwc_R100.mat')
load('matlab_reff_R100.mat')
load('matlab_veff_R100.mat')
load('matlab_ext_R100.mat')

%%
lwc = r100_lwc;
reff = r100_reff;
veff = r100_veff;
mask = lwc>0.01;
mask = sum(mask,2)>0;
flat_ext = lwc(mask,2:end);
% mask = sum(flat_ext,2)>0;
C = fftshift(fft(full(flat_ext),[],2))/159;
PowerLWC = abs(C).^2;%.*conj(C);
PowerLWC = sum(PowerLWC,1);
T=5;
E=sum(PowerLWC);
PowerLWC=PowerLWC/E/(1/T);
%%
% mean_lwc=[];
% mean_reff=[];
% mean_veff=[];
% 
% for i=1:size(lwc,2)
%    lwc_i = lwc(:,i);
%    lwc_i = permute(reshape(full(lwc_i),320,514,514),[3,2,1]);
%    reff_i = reff(:,i);
%    reff_i = permute(reshape(full(reff_i),320,514,514),[3,2,1]);
%    veff_i = veff(:,i);
%    veff_i = permute(reshape(full(veff_i),320,514,514),[3,2,1]);
%    mask2 = lwc_i<0.01;
%    lwc_i(mask2)=nan;
%    reff_i(mask2)=nan;
%    veff_i(mask2)=nan;
% 
%    mean_lwc(:,i) = squeeze(mean(lwc_i,[1,2],'omitnan'));
%    mean_reff(:,i) = squeeze(mean(reff_i,[1,2],'omitnan'));
%    mean_veff(:,i) = squeeze(mean(veff_i,[1,2],'omitnan'));
% 
% end
% save('mean_microphysics','mean_lwc','mean_reff','mean_veff')

%%
mean_reff(isnan(mean_reff))=0;
mean_lwc(isnan(mean_lwc))=0;
mask = mean_lwc>0.01;
mask = sum(mask,2)>0;
flat_ext = mean_reff(mask,2:end);
% mask = sum(flat_ext,2)>0;
C = fftshift(fft(full(flat_ext),[],2))/159;
PowerREFF = abs(C).^2;%.*conj(C);
PowerREFF = sum(PowerREFF,1);
E=sum(PowerREFF);
PowerREFF=PowerREFF./E./(1/T);
%%
mean_veff(isnan(mean_veff))=0;
mean_lwc(isnan(mean_lwc))=0;
mask = mean_lwc>0.01;
mask = sum(mask,2)>0;
flat_ext = mean_veff(mask,2:end);
% mask = sum(flat_ext,2)>0;
C = fftshift(fft(full(flat_ext),[],2))/159;
PowerVEFF = abs(C).^2;%.*conj(C);
PowerVEFF = sum(PowerVEFF,1);
E=sum(PowerVEFF);
PowerVEFF=PowerVEFF./E./(1/T);
%%
Fs=1/5;
F = (-79:79)*Fs/159;
f = (-79:0.1:79)*Fs/159;
PowerLWCi = interp1(F,PowerLWC,f,'cubic');
PowerREFFi = interp1(F,PowerREFF,f,'cubic');
PowerVEFFi = interp1(F,PowerVEFF,f,'cubic');

figure;
plot(f,(PowerLWCi),'LineWidth',2)
hold on
xlabel('Frequency [Hz]');
ylabel('Normalized Power');
set(gca,'linewidth',2)
grid on
plot(f,(PowerREFFi),'LineWidth',2)
grid on
plot(f,(PowerVEFFi),'LineWidth',2)
% plot(F,S1(:,13),'LineWidth',2)
legend('LWC','Vertical r_{eff}')
xlim([-0.02 0.02])

%%
PowerLWCi2 = PowerLWCi.^2;
center = 791;
ii=1;
close all
T_vec = [1,10,20,30,40,80,160,320];
MSE_lwc=[]
for T=T_vec
cir_Lambda = PowerLWCi;
cir_Lambda2 = PowerLWCi2;
w0=2*pi/T;
m = floor(pi/w0/5);
dn = ceil(790/m);
for i=1:m
    if i==m
cir_Lambda = cir_Lambda+((circshift(PowerLWCi,dn*i) + circshift(PowerLWCi,-dn*i)))/2;
cir_Lambda2 = cir_Lambda2+((circshift(PowerLWCi2,dn*i) + circshift(PowerLWCi2,-dn*i)))/2;
    else
        cir_Lambda = cir_Lambda+(circshift(PowerLWCi,dn*i) + circshift(PowerLWCi,-dn*i));
cir_Lambda2 = cir_Lambda2+(circshift(PowerLWCi2,dn*i) + circshift(PowerLWCi2,-dn*i));
    end
end
len = floor(length(f)/2/T);
F = round(center-len:center+len);
divider = cir_Lambda2./cir_Lambda;
divider(isnan(divider))=0;
figure
plot(f,divider)
hold on
plot(f,PowerLWCi)
figure
plot(F,PowerLWCi(F) - divider(F))
MSE_lwc(ii)=sum(PowerLWCi(F) - divider(F))/w0;
ii=ii+1;
end
close all
figure
hold on
plot(T_vec,MSE_lwc)
xlabel('Sampeling interval T[sec]')
ylabel('MSE Eq.10')
%%
PowerREFFi2 = PowerREFFi.^2;
center = 791;
ii=1;
close all
T_vec = [1,10,20,30,40,80,160,320];
MSE_reff=[]
for T=T_vec

    cir_Lambda = PowerREFFi;
cir_Lambda2 = PowerREFFi2;
w0=2*pi/T;
m = floor(pi/w0/5);
dn = ceil(790/m);
for i=1:m
    if i==m
cir_Lambda = cir_Lambda+((circshift(PowerREFFi,dn*i) + circshift(PowerREFFi,-dn*i)))/2;
cir_Lambda2 = cir_Lambda2+((circshift(PowerREFFi2,dn*i) + circshift(PowerREFFi2,-dn*i)))/2;
    else
        cir_Lambda = cir_Lambda+(circshift(PowerREFFi,dn*i) + circshift(PowerREFFi,-dn*i));
cir_Lambda2 = cir_Lambda2+(circshift(PowerREFFi2,dn*i) + circshift(PowerREFFi2,-dn*i));
    end
end
len = floor(length(f)/2/T);
F = round(center-len:center+len);
divider = cir_Lambda2./cir_Lambda;
divider(isnan(divider))=0;
figure
plot(f,divider)
hold on
plot(f,PowerREFFi)
figure
plot(F,PowerREFFi(F) - divider(F))
MSE_reff(ii)=sum(PowerREFFi(F) - divider(F))/w0;
ii=ii+1;
end
close all
figure
hold on
plot(T_vec,MSE_reff)
xlabel('Sampeling interval T[sec]')
ylabel('MSE Eq.10')


%%
T_vec = [1,10,20,30,40,80,160,320];
figure
hold on
plot([0 T_vec],[0 MSE_reff],'LineWidth',2)
hold on
plot([ 0 T_vec],[0 MSE_lwc],'LineWidth',2)

xlabel('Sampeling \ interval\ T\ $[{\rm sec}]$', 'Interpreter','latex')
ylabel('MSE ', 'Interpreter','latex')
xlim([0,100])
set(gca,'linewidth',2)
ylim([0 30])
grid on
%%
%%
mask = lwc>0.01;
mask = sum(mask,2)>1;
flat_ext = full(lwc(mask,:));
c_ww=[]
i=randperm(size(flat_ext,1));
for ri=1:500
    ii=i(ri)
    [c_ww_i,lags] = xcorr(flat_ext(ii,:),159,'normalized');
    c_ww(ri,:) = c_ww_i;
end
% [c_ww,lags] = xcorr(flat_ext(1:L,:)',89,'coeff');
ac_lwc = mean(c_ww,1);
figure;plot(lags*5,mean(c_ww,1))
xlabel('Time lag[sec]');
ylabel('Mean auto-correlation LWC');
%%
mean_reff(isnan(mean_reff))=0;
c_ww=[]
flat_ext = mean_reff;
for ri=1:size(flat_ext)
    [c_ww_i,lags] = xcorr(flat_ext(ri,:),159,'normalized');
    c_ww(ri,:) = c_ww_i;
end
% c_ww = c_ww./c_ww_0;
ac_reff = mean(c_ww,1,'omitnan')';
figure;plot(lags*5,ac_reff)
xlabel('Time lag[sec]');
ylabel('Auto-correlation reff');
%%
figure;plot(lags*5,ac_lwc,'LineWidth',2)
hold on
plot(lags*5,ac_reff,'LineWidth',2)
hold on

xlabel('Time lag [sec]', 'Interpreter','latex');
ylabel('Auto-correlation', 'Interpreter','latex');
grid on
plot([-500 500],[0.5 0.5],'LineWidth',2,'LineStyle','--','Color','k')
xlim([-500 500])
set(gca,'linewidth',2)