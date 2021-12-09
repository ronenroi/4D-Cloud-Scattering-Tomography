clear
folder = '/media/roironen/8AAE21F5AE21DB09/clouds wiz/lwc';
% folder='/home/roironen/pyshdom/matlab scripts';
sad = dir( fullfile( folder, 'cloud*.mat' ) );
len = length( sad );
out = struct( 'name',cell(1,len), 'data',[] );
ii=1;
for jj = 1:len
    %     data = load(fullfile( folder, sad(jj).name ),'time');
    data = load(fullfile( folder, ['cloud', num2str(jj-1+37),'.mat'] ));
    
    %     ext = permute(reshape(full(data.extinction),320,502,502),[3,2,1]);
    lwc_i = full(data.lwc(:));
    reff_i = full(data.reff(:));
    
    %         time(ii) = data.time;
    %     ii = ii + 1;
    
    if ~exist('mask')
        %         prev_time = data.time;
        mask = lwc_i>0.1;
        lwc(:,ii) = lwc_i;
        reff(:,ii) = reff_i;
        ii = ii + 1;
    else
        %         if data.time - prev_time == 5
        mask = mask | (lwc_i>0);
        %         prev_time = data.time;
        lwc(:,ii) = lwc_i;
        reff(:,ii) = reff_i;
        %     time(ii) = data.time;
        %     data.time
        ii = ii + 1;
        %         end
    end
    lwc = sparse(lwc);
    reff = sparse(reff);
    mask = sparse(mask);
    %     extinction(:,:,:,jj) = ext;
    %     extinction=extinction(mask,:);

end
%%
clear
folder = '/media/roironen/8AAE21F5AE21DB09/clouds wiz/lwc';
% folder='/home/roironen/pyshdom/matlab scripts';
sad = dir( fullfile( folder, 'cloud*.mat' ) );
len = length( sad );
out = struct( 'name',cell(1,len), 'data',[] );
ii=1;
for jj = 1:len
    %     data = load(fullfile( folder, sad(jj).name ),'time');
    data = load(fullfile( folder, ['cloud', num2str(jj-1+37),'.mat'] ),'veff');
    
    %     ext = permute(reshape(full(data.extinction),320,502,502),[3,2,1]);
    veff_i = full(data.veff(:));
    
    veff(:,jj) = sparse(veff_i);

%     save('matlab_veff','veff','-v7.3');    

end
%     save('matlab_lwc','lwc','mask','-v7.3');
% extinction = extinction(mask,:);
%%
load('matlab_lwc_R100.mat')
load('matlab_reff_R100.mat')
load('matlab_veff_R100.mat')
%%
lwc = r100_lwc;
reff = r100_reff;
veff = r100_veff;
mask = lwc>0.01;
mask = sum(mask,2)>0;
flat_ext = lwc(mask,:);
%%
close all
for sigma = [20,30,40]
g=normpdf(-100:5:100,0,sigma)';
g=g/sum(g);
gg=zeros(41,1);
gg(1:4:end)=g(1:4:end);
figure
stem(gg)
for i=1:4:1000
    b = full(flat_ext(i,:))';
    a = conv(b,gg,'same'); 
    err(i) = sum((a-b).^2);
end
gg=zeros(41,1);
gg(2:4:end)=g(2:4:end);
figure
stem(gg)
hold on
plot(g)
for i=2:4:1000
    b = full(flat_ext(i,:))';
    a = conv(b,gg,'same'); 
    err(i) = sum((a-b).^2);
end
gg=zeros(41,1);
gg(3:4:end)=g(3:4:end);
figure
stem(gg)
hold on
plot(g)
for i=3:4:1000
    b = full(flat_ext(i,:))';
    a = conv(b,gg,'same'); 
    err(i) = sum((a-b).^2);
end
gg=zeros(41,1);
gg(4:4:end)=g(4:4:end);
figure
stem(gg)
hold on
plot(g)
for i=4:4:1000
    b = full(flat_ext(i,:))';
    a = conv(b,gg,'same'); 
    err(i) = sum((a-b).^2);
end
figure 
plot(err)
mean(err)
end
%%
close all
mask = lwc>0.01;
mask = sum(mask,2)>0;
flat_ext = lwc(mask,2:end);
% mask = sum(flat_ext,2)>0;
C = fftshift(fft(full(flat_ext),[],2))/89;


T = 5;             % Sampling period
Fs = 1/T;
L = 89;             % Length of signal
t = (0:L-1)*T;        % Time vector
F = (-44:44)*Fs/90;

flat_ext = lwc(mask,2:4:end);
C20 = fftshift(fft(full(flat_ext),[],2))/89;
F20 = (-44:4:44)*Fs/90;
ii=0
MSE=[]
for sigma=[1,10 20 30 40 50]
    ii=ii+1;
g=normpdf(F,0,1/sigma)';
g=g/sum(g);
figure
plot(g)
for i=1:10000
C20_1(i,:) = interp1(F20,C20(i,:),F).*g';
end
MSE(ii) = mean(mean(abs((C20_1 - C(1:10000,:)).^2)));
end
close all
plot(MSE)
figure
plot(F,abs(C))
hold on

plot(F,Lambda_LWC_10)
hold on
plot(F,Lambda_LWC_20)
hold on
plot(F,Lambda_LWC_40)
%%
close all
mask = lwc>0.01;
mask = sum(mask,2)>0;
flat_ext = lwc(mask,2:end);
% mask = sum(flat_ext,2)>0;
C = fft(full(flat_ext),[],2);
P = abs(C).^2;%.*conj(C);
A = sum(P,1);
T = 5;             % Sampling period
Fs = 1/T;
L = 89;             % Length of signal
t = (0:L-1)*T;        % Time vector
Lambda_LWC = fftshift(abs(A/L));
n=sum(Lambda_LWC);
Lambda_LWC=Lambda_LWC/n;
F = (-44:44)*Fs/90;

flat_ext = lwc(mask,1:2:end);
C = fft(full(flat_ext),[],2);
P = abs(C).^2;%.*conj(C);
A = sum(P,1);

L = 45;             % Length of signal
Lambda_LWC_10 = fftshift(abs(A/L))*2;
F10 = (-44:2:44)*Fs/90;
Lambda_LWC_10 = interp1(F10,Lambda_LWC_10,F)/n;

flat_ext = lwc(mask,1:4:end);
C = fft(full(flat_ext),[],2);
P = abs(C).^2;%.*conj(C);
A = sum(P,1);

L = 45/2;             % Length of signal
Lambda_LWC_20 = fftshift(abs(A/L))*4;
F20 = (-44:4:44)*Fs/90;
Lambda_LWC_20 = interp1(F20,Lambda_LWC_20,F)/n;


flat_ext = lwc(mask,1:8:end);
C = fft(full(flat_ext),[],2);
P = abs(C).^2;%.*conj(C);
A = sum(P,1);

L = 90/8;             % Length of signal
Lambda_LWC_40 = fftshift(abs(A/L))*8;
F40 = (-44:8:44)*Fs/90-0.0089;
Lambda_LWC_40 = interp1(F40,Lambda_LWC_40,F)/n;
Lambda_LWC_40(89)=Lambda_LWC_40(1);
Lambda_LWC_40(88)=Lambda_LWC_40(2);
Lambda_LWC_40(87)=Lambda_LWC_40(3);
Lambda_LWC_40(86)=Lambda_LWC_40(4);
Lambda_LWC_40(85)=Lambda_LWC_40(5);
figure
plot(F,Lambda_LWC)
hold on

plot(F,Lambda_LWC_10)
hold on
plot(F,Lambda_LWC_20)
hold on
plot(F,Lambda_LWC_40)

%%

Fi = (-44:0.1:44)*Fs/90;

Lambda_LWCi = interp1(F,Lambda_LWC,Fi);
Lambda_LWC2 = Lambda_LWCi.^2;

cir_Lambda_LWC_T2 = Lambda_LWCi+(circshift(Lambda_LWCi,440) + circshift(Lambda_LWCi,-440))/2;

cir_Lambda_LWC2_T2 = Lambda_LWC2+(circshift(Lambda_LWC2,440) + circshift(Lambda_LWC2,-440))/2;
center = ceil(length(Fi)/2);
len = floor(length(Fi)/2)/2;
F2 = center-len:center+len;
figure
plot(Fi,cir_Lambda_LWC2_T2./cir_Lambda_LWC_T2)
hold on
plot(Fi,Lambda_LWCi)
MSE=[];
MSE(1)=0;
MSE(2)=sum(Lambda_LWCi(F2) - cir_Lambda_LWC2_T2(F2)./cir_Lambda_LWC_T2(F2))
%
cir_Lambda_LWC_T4 = Lambda_LWCi+(circshift(Lambda_LWCi,440) + circshift(Lambda_LWCi,-440))/2+(circshift(Lambda_LWCi,220) + circshift(Lambda_LWCi,-220));

cir_Lambda_LWC2_T4 = Lambda_LWC2+(circshift(Lambda_LWC2,440) + circshift(Lambda_LWC2,-440))/2+(circshift(Lambda_LWC2,220) + circshift(Lambda_LWC2,-220));
len = floor(length(Fi)/4)/2;
F4 = center-len:center+len;
figure
plot(Fi,cir_Lambda_LWC2_T4./cir_Lambda_LWC_T4)
hold on
plot(Fi,Lambda_LWCi)
MSE(3)=sum(Lambda_LWCi(F4) - cir_Lambda_LWC2_T4(F4)./cir_Lambda_LWC_T4(F4))
%
df = round(880/6);
cir_Lambda_LWC_T6 = Lambda_LWCi+(circshift(Lambda_LWCi,df) + circshift(Lambda_LWCi,-df))+(circshift(Lambda_LWCi,df*2) + circshift(Lambda_LWCi,-df*2))+(circshift(Lambda_LWCi,df*3) + circshift(Lambda_LWCi,-df*3))/2;
cir_Lambda_LWC2_T6 = Lambda_LWC2+(circshift(Lambda_LWC2,df) + circshift(Lambda_LWC2,-df))+(circshift(Lambda_LWC2,df*2) + circshift(Lambda_LWC2,-df*2))+(circshift(Lambda_LWC2,df*3) + circshift(Lambda_LWC2,-df*3))/2;
len = floor(length(Fi)/6)/2;
F6 = center-len:center+len;
figure
plot(Fi,cir_Lambda_LWC2_T6./cir_Lambda_LWC_T6)
hold on
plot(Fi,Lambda_LWCi)
MSE(4)=sum(Lambda_LWCi(F6) - cir_Lambda_LWC2_T6(F6)./cir_Lambda_LWC_T6(F6))
%
cir_Lambda_LWC_T8 = Lambda_LWCi+(circshift(Lambda_LWCi,440) + circshift(Lambda_LWCi,-440))/2+(circshift(Lambda_LWCi,220) + circshift(Lambda_LWCi,-220))+(circshift(Lambda_LWCi,110) + circshift(Lambda_LWCi,-110))+(circshift(Lambda_LWCi,330) + circshift(Lambda_LWCi,-330));
cir_Lambda_LWC2_T8 = Lambda_LWC2+(circshift(Lambda_LWC2,440) + circshift(Lambda_LWC2,-440))/2+(circshift(Lambda_LWC2,220) + circshift(Lambda_LWC2,-220))+(circshift(Lambda_LWC2,110) + circshift(Lambda_LWC2,-110))+(circshift(Lambda_LWC2,330) + circshift(Lambda_LWC2,-330));
len = floor(length(Fi)/8)/2;
F8 = center-len:center+len;
figure
plot(Fi,cir_Lambda_LWC2_T8./cir_Lambda_LWC_T8)
hold on
plot(Fi,Lambda_LWCi)
MSE(5)=sum(Lambda_LWCi(F8) - cir_Lambda_LWC2_T8(F8)./cir_Lambda_LWC_T8(F8))
%
df = round(880/16);
cir_Lambda_LWC_T16 = Lambda_LWCi;
cir_Lambda_LWC2_T16 = Lambda_LWC2;
for i=1:7
cir_Lambda_LWC_T16 = cir_Lambda_LWC_T16+(circshift(Lambda_LWCi,df*i) + circshift(Lambda_LWCi,-df*i));
cir_Lambda_LWC2_T16 = cir_Lambda_LWC2_T16+(circshift(Lambda_LWC2,df*i) + circshift(Lambda_LWC2,-df*i));
end
i=i+1;
cir_Lambda_LWC_T16 = cir_Lambda_LWC_T16 +(circshift(Lambda_LWCi,df*i) + circshift(Lambda_LWCi,-df*i))/2;
cir_Lambda_LWC2_T16 = cir_Lambda_LWC2_T16 +(circshift(Lambda_LWC2,df*i) + circshift(Lambda_LWC2,-df*i))/2;
len = floor(length(Fi)/16+1)/2;
F16 = center-len:center+len;
figure
plot(Fi,cir_Lambda_LWC2_T16./cir_Lambda_LWC_T16)
hold on
plot(Fi,Lambda_LWCi)
MSE(6)=sum(Lambda_LWCi(F16) - cir_Lambda_LWC2_T16(F16)./cir_Lambda_LWC_T16(F16))
%
df = round(880/32);
cir_Lambda_LWC_T32 = Lambda_LWCi;
cir_Lambda_LWC2_T32 = Lambda_LWC2;
for i=1:14
cir_Lambda_LWC_T32 = cir_Lambda_LWC_T32+(circshift(Lambda_LWCi,df*i) + circshift(Lambda_LWCi,-df*i));
cir_Lambda_LWC2_T32 = cir_Lambda_LWC2_T32+(circshift(Lambda_LWC2,df*i) + circshift(Lambda_LWC2,-df*i));
end
i=i+1;
cir_Lambda_LWC_T32 = cir_Lambda_LWC_T32 +(circshift(Lambda_LWCi,df*i) + circshift(Lambda_LWCi,-df*i))/2;
cir_Lambda_LWC2_T32 = cir_Lambda_LWC2_T32 +(circshift(Lambda_LWC2,df*i) + circshift(Lambda_LWC2,-df*i))/2;
len = floor(length(Fi)/32+1)/2;
F32 = center-len:center+len;
figure
plot(Fi,cir_Lambda_LWC2_T32./cir_Lambda_LWC_T32)
hold on
plot(Fi,Lambda_LWCi)
MSE(7)=sum(Lambda_LWCi(F32) - cir_Lambda_LWC2_T32(F32)./cir_Lambda_LWC_T32(F32))
%%
T=[2,4,6,8,16,32]*5;
figure
plot(T,MSE(2:end))
xlabel('Sampeling interval T[sec]')
ylabel('MSE Eq.10')
%%

Lambda_LWC1 = Lambda_LWC*10^0;
Lambda_LWC_101 =Lambda_LWC_10 *10^0;
Lambda_LWC_201 =Lambda_LWC_20 *10^0;
Lambda_LWC_401 =Lambda_LWC_40 *10^0;

MSE(1) = sum( (Lambda_LWC1)-(Lambda_LWC1./Lambda_LWC1 .* Lambda_LWC1));
MSE(2) = sum( (Lambda_LWC1)-(Lambda_LWC1.* Lambda_LWC_101)./(Lambda_LWC1 .* Lambda_LWC_101.^2));
MSE(3) = sum( (Lambda_LWC1))-sum((Lambda_LWC1.* Lambda_LWC_201).^2 ./ (Lambda_LWC1 .* Lambda_LWC_201.^2));
MSE(4) = (sum( (Lambda_LWC1))-sum((Lambda_LWC1.* Lambda_LWC_401).^2 ./ (Lambda_LWC1 .* Lambda_LWC_401.^2)));
figure
plot([1,2,4,8],MSE/10^1)

%%
mask = lwc>0.01;
mask = sum(mask,2)>0;
flat_ext = lwc(mask,:);
% mask = sum(flat_ext,2)>0;
C = fft(full(flat_ext),[],2);
P = abs(C).^2;%.*conj(C);
max(imag(P(:)))
A = sum(P,1);

E=sum(A);
A=A/E;
der_flat_ext = diff(flat_ext,1,2);
der_C = fft(full(der_flat_ext),[],2);
der_P = abs(der_C).^2;%.*conj(C);
max(imag(der_P(:)))
der_A = sum(der_P,1);
der_E=sum(der_A);
der_E/E

T = 5;             % Sampling period
Fs = 1/T;
L = 90;             % Length of signal
t = (0:L-1)*T;        % Time vector
LWC_F = abs(A/L);

% P1 = P2(1:L/2+1)';
% P1(2:end-1) = 2*P1(2:end-1);
% f = (Fs*(0:(L/2))/L)';
% figure
% plot(f,P1)
% hold on
% title('Single-Sided Amplitude Spectrum of cloud(x,t)')
% xlabel('f (Hz)')
% ylabel('|A(f)|')
% plot(fittedmodel)
%%
mean_reff=[];
for i=1:size(lwc,2)
   lwc_i = lwc(:,i);
   lwc_i = reshape(full(lwc_i),502,502,320);
   reff_i = reff(:,i);
   reff_i = reshape(full(reff_i),502,502,320);
   veff_i = veff(:,i);
   veff_i = reshape(full(veff_i),502,502,320);
   
   mask2 = lwc_i<0.01;
   lwc_i(mask2)=nan;
   reff_i(mask2)=nan;
   veff_i(mask2)=nan;

   mean_lwc(:,i) = squeeze(mean(lwc_i,[1,2],'omitnan'));
   mean_reff(:,i) = squeeze(mean(reff_i,[1,2],'omitnan'));
   mean_veff(:,i) = squeeze(mean(veff_i,[1,2],'omitnan'));


end
%%
mean_reff(isnan(mean_reff))=0;
mean_lwc(isnan(mean_lwc))=0;
mask = mean_lwc>0.01;
mask = sum(mask,2)>0;
flat_ext = mean_reff(mask,:);
% mask = sum(flat_ext,2)>0;
C = fft(full(flat_ext),[],2);
P = abs(C).^2;%.*conj(C);
max(imag(P(:)))
A = sum(P,1);

E=sum(A);
A=A/E;
der_flat_ext = diff(flat_ext,1,2);
der_C = fft(full(der_flat_ext),[],2);
der_P = abs(der_C).^2;%.*conj(C);
max(imag(der_P(:)))
der_A = sum(der_P,1);
der_E=sum(der_A);
der_E/E

T = 5;             % Sampling period
Fs = 1/T;
L = 90;             % Length of signal
REFF_F = abs(A/L);
%%
mean_veff(isnan(mean_veff))=0;
flat_ext = mean_veff(mask,:);
% mask = sum(flat_ext,2)>0;
C = fft(full(flat_ext),[],2);
P = abs(C).^2;%.*conj(C);
max(imag(P(:)))
A = sum(P,1);

E=sum(A);
A=A/E;
der_flat_ext = diff(flat_ext,1,2);
der_C = fft(full(der_flat_ext),[],2);
der_P = abs(der_C).^2;%.*conj(C);
max(imag(der_P(:)))
der_A = sum(der_P,1);
der_E=sum(der_A);
der_E/E

T = 5;             % Sampling period
Fs = 1/T;
L = 90;             % Length of signal
VEFF_F = abs(A/L);

%%
F = (-44.5:44.5)*Fs/90;
figure;
plot(F,fftshift(LWC_F),'LineWidth',2)
hold on
xlabel('Frequency [Hz]');
ylabel('Normalized Power');
set(gca,'linewidth',2)
grid on
plot(F,fftshift(REFF_F),'LineWidth',2)
grid on
plot(F,fftshift(VEFF_F),'LineWidth',2)
% plot(F,S1(:,13),'LineWidth',2)
legend('LWC','Vertical r_{eff}','Vertical v_{eff}')
xlim([-0.02 0.02])
%%
F = (-44.5:44.5)*Fs/90;
figure;
semilogy(F,fftshift(LWC_F),'LineWidth',2)
hold on
xlabel('Frequency [Hz]');
ylabel('Normalized Power');
set(gca,'linewidth',2)
grid on
semilogy(F,fftshift(REFF_F),'LineWidth',2)
grid on
semilogy(F,fftshift(VEFF_F),'LineWidth',2)
% plot(F,S1(:,13),'LineWidth',2)
legend('LWC','Vertical r_{eff}','Vertical v_{eff}')
%%
mask = lwc>0.01;
mask = sum(mask,2)>1;
flat_ext = full(lwc(mask,:));
c_ww=[]
i=randperm(size(flat_ext,1));
for ri=1:500
    ii=i(ri)
    [c_ww_i,lags] = xcorr(flat_ext(ii,:),89,'normalized');
    c_ww(ri,:) = c_ww_i;
end
% [c_ww,lags] = xcorr(flat_ext(1:L,:)',89,'coeff');
ac_lwc = mean(c_ww,1);
figure;plot(lags*5,mean(c_ww,1))
xlabel('Time lag[sec]');
ylabel('Mean auto-correlation LWC');
%%
mean_reff(isnan(mean_reff))=0;
mean_lwc(isnan(mean_lwc))=0;
mask = mean(mean_lwc,2)>0.001;
c_ww=[]
flat_ext=[];
flat_ext = mean_reff(mask,:);
% flat_ext= flat_ext(mean_lwc>0.01);
for ri=1:size(flat_ext)
    [c_ww_i,lags] = xcorr(flat_ext(ri,:),89,'normalized');
    c_ww(ri,:) = c_ww_i;
end
ac_reff =mean(c_ww,1,'omitnan')';
% c_ww = c_ww./c_ww_0;
figure;plot(lags*5,ac_reff')
xlabel('Time lag[sec]');
ylabel('Mean auto-correlation reff');
%%
mean_veff(isnan(mean_veff))=0;
c_ww=[]
flat_ext = mean_veff;
for ri=1:size(flat_ext)
    [c_ww_i,lags] = xcorr(flat_ext(ri,:),89,'normalized');
    c_ww(ri,:) = c_ww_i;
end
% c_ww = c_ww./c_ww_0;
figure;plot(lags*5,mean(c_ww,1,'omitnan')')
xlabel('Time lag[sec]');
ylabel('Mean auto-correlation veff');
%%
close all
figure
plot(lags*5,ac_reff-ac_lwc','LineWidth',2)
hold on
plot(lags*5,ac_lwc,'LineWidth',2)
plot([-500 500],[0.5 0.5],'LineWidth',2,'LineStyle','--')

xlabel('Time lag[sec]');
ylabel('Mean auto-correlation');
set(gca,'linewidth',2)


%%
load('matlab_lwc.mat')
load('matlab_reff.mat')


% flat_ext = reshape(reff(mask,:),[],52);
mask = lwc>0.01;
mask = sum(mask,2)>0;
flat_ext = reff(mask,:);
% mask = sum(flat_ext,2)>0;
C = fft(full(flat_ext),[],2);
P = abs(C).^2;%.*conj(C);
max(imag(P(:)))
A = sum(P,1);
E=sum(A);
der_flat_ext = diff(flat_ext,1,2);
der_C = fft(full(der_flat_ext),[],2);
der_P = abs(der_C).^2;%.*conj(C);
max(imag(der_P(:)))
der_A = sum(der_P,1);
der_E=sum(der_A);
der_E/E

T = 5;             % Sampling period
Fs = 1/T;
L = 52;             % Length of signal
t = (0:L-1)*T;        % Time vector
P2 = abs(A/L);
P1 = P2(1:L/2+1)';
P1(2:end-1) = 2*P1(2:end-1);
f = (Fs*(0:(L/2))/L)';
%%
%%
for i=1:size(lwc,2)
   lwc_i = lwc(:,i);
   lwc_i = reshape(full(lwc_i),502,502,320);
   reff_i = reff(:,i);
   reff_i = reshape(full(reff_i),502,502,320);
   lwc_i= lwc_i(357:400,340:370,60:90);
   m_lwc1(:,i) = lwc_i(:);
   reff_i = reff_i(357:400,340:370,60:90);
   m_reff1(:,i) = reff_i(:);

end
% flat_ext = reshape(reff(mask,:),[],52);
mask = m_lwc1>0.1;
mask = sum(mask,2)>0;
flat_ext = m_reff1(mask,:);

% mask = sum(flat_ext,2)>0;
C = fft(full(flat_ext),[],2);
P = abs(C).^2;%.*conj(C);
max(imag(P(:)))
A = sum(P,1);
E=sum(A);
der_flat_ext = diff(flat_ext,1,2);
der_C = fft(full(der_flat_ext),[],2);
der_P = abs(der_C).^2;%.*conj(C);
max(imag(der_P(:)))
der_A = sum(der_P,1);
der_E=sum(der_A);
der_E/E

T = 5;             % Sampling period
Fs = 1/T;
L = 90;             % Length of signal
t = (0:L-1)*T;        % Time vector
P2 = abs(A/L);
P1 = P2(1:L/2+1)';
P1(2:end-1) = 2*P1(2:end-1);
f = (Fs*(0:(L/2))/L)';
%%

% fo = fitoptions('Method','NonlinearLeastSquares',...
%                'Lower',[Inf,-10],...
%                'Upper',[10,0],...
%                'StartPoint',[1 -0.1]);
% ft = fittype('a*(x)^n');
% [curve2,gof2] = fit(f(2:end),P1(2:end),ft);
%
%


figure
plot(f,P1)
hold on
% plot(fittedmodel)

title('Single-Sided Amplitude Spectrum of cloud(x,t)')
xlabel('f (Hz)')
ylabel('|A(f)|')
figure
loglog(f,P1)
title('Single-Sided Amplitude Spectrum of cloud(x,t)')
xlabel('f (Hz)')
ylabel('|A(f)|')
%%
figure
[S,F,T] = stft(full(flat_ext)',Fs,'Window',kaiser(6,5),'OverlapLength',5,'FFTLength',6);
% [S,F,T] = stft(full(flat_ext)',Fs,'Window',hamming(4),'OverlapLength',3,'FFTLength',50);
a=T/5;
norm = flat_ext(:,a);
norm = sum(norm>0);

S=abs(S).^2;
S1=sum(S,3);
surf(F,T,abs(S1)')
ax = gca;
% ax.Title.String = ['STFT cloud(x,t)'];
ax.XLabel.String = 'Frequency (Hz)';
ax.YLabel.String = 'Time (seconds)';
ax.ZLabel.String = 'Power [km^{-2}]';
view(3)
set(gca,'linewidth',2)
fig.Renderer='Painters';
%
% figure
% imagesc(T,F,S1)
% ax.Title.String = ['STFT cloud(x,t)'];
% ax.XLabel.String = 'Frequency (Hz)';
% ax.YLabel.String = 'Time (seconds)';

%%
S1 = abs(S1);
figure;
plot(F,S1(:,1),'LineWidth',2)
hold on
xlabel('Frequency [Hz]');
ylabel('Power [km^{-2}]');
set(gca,'linewidth',2)
grid on
plot(F,S1(:,3),'LineWidth',2)
% plot(F,S1(:,13),'LineWidth',2)
legend('Time 0','Time 300[sec]','Time 600[sec]')


%%
for i=1:size(lwc,2)
   lwc_i = lwc(:,i);
   lwc_i = reshape(full(lwc_i),502,502,320);
   reff_i = reff(:,i);
   reff_i = reshape(full(reff_i),502,502,320);
   mask2 = lwc_i<0.01;
   lwc_i(mask2)=nan;
   reff_i(mask2)=nan;
   mean_lwc(:,i) = squeeze(mean(lwc_i,[1,2],'omitnan'));
   mean_reff(:,i) = squeeze(mean(reff_i,[1,2],'omitnan'));

end
% reff = reshape(reff,502,502,320*size(reff,2));
% lwc = reshape(lwc,502*502,320*size(lwc,2));
% 
% mask2 = lwc>0.01;
% reff1=reff;
% for i=1:size(mask2,2)
%     reff1(~mask2(:,i),i)=nan;
% end
% mean_reff = squeeze(mean(reff,[1,2],'omitnan'));
colorma = colormap('hot');
colorma=colorma(1:2:180,:);
figure
hold on
for i=1:90
plot(0:10:3199,mean_reff(:,i),'Color',colorma(i,:))
end
ylabel('reff[micron]')
xlabel('Altitude[m]')
figure
hold on
for i=1:90
plot(0:10:3199,mean_lwc(:,i),'Color',colorma(i,:))
end
ylabel('LWC')
xlabel('Altitude[m]')

%%

%%
for i=1:size(lwc,2)
   lwc_i = lwc(:,i);
   lwc_i = reshape(full(lwc_i),502,502,320);
   reff_i = reff(:,i);
   reff_i = reshape(full(reff_i),502,502,320);
   mask2 = lwc_i<0.01;
   lwc_i(mask2)=nan;
   reff_i(mask2)=nan;
   mean_lwc(:,i) = squeeze(mean(lwc_i(357:400,340:370,:),[1,2],'omitnan'));
   mean_reff(:,i) = squeeze(mean(reff_i(357:400,340:370,:),[1,2],'omitnan'));
   std_lwc(:,i) = squeeze(std(lwc_i(357:400,340:370,:),0,[1,2],'omitnan')) ;
   std_reff(:,i) = squeeze(std(reff_i(357:400,340:370,:),0,[1,2],'omitnan'));

end
% reff = reshape(reff,502,502,320*size(reff,2));
% lwc = reshape(lwc,502*502,320*size(lwc,2));
% 
% mask2 = lwc>0.01;
% reff1=reff;
% for i=1:size(mask2,2)
%     reff1(~mask2(:,i),i)=nan;
% end
% mean_reff = squeeze(mean(reff,[1,2],'omitnan'));
%%
colorma = colormap('hot');
colorma=colorma(1:10:220,:);
figure
for i=1:2:20
plot(0:10:3199,mean_reff(:,i+63),'Color',colorma(i,:))
hold on
end
ylabel('reff[micron]')
xlabel('Altitude[m]')
z = ((0:10:3199)-530)/1000;
z(z<0)=0;
rrr = 7.1 * (z.^(1/3))+5;
rrr(z>0.5)=5;
plot(0:10:3199,rrr)

figure
hold on
for i=1:2:20
plot(0:10:3199,mean_lwc(:,i+63),'Color',colorma(i,:))
end
ylabel('LWC')
xlabel('Altitude[m]')
rrr = 0.7 * (z)+0.01;
rrr(z>0.4)=0.01;
plot(0:10:3199,rrr)
%%
colorma = colormap('hot');
colorma=colorma(1:10:220,:);
figure
hold on
for i=1:2:20
plot(0:10:3199,std_reff(:,i+50),'Color',colorma(i,:))
end
ylabel('reff[micron]')
xlabel('Altitude[m]')
figure
hold on
for i=1:2:20
plot(0:10:3199,std_lwc(:,i+50),'Color',colorma(i,:))
end
ylabel('LWC')
xlabel('Altitude[m]')