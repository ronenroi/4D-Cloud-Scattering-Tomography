clear
folder='/home/roironen/pyshdom/Develop_Dynamic_cloud';
folder='../../../wdata/roironen/R100';

sad = dir( fullfile( folder, 'R100_*.mat' ) );
len = length( sad );
out = struct( 'name',cell(1,len), 'data',[] );
ii=1;
for jj = 1:len
%     data = load(fullfile( folder, sad(jj).name ),'time');
    data = load(fullfile( folder, ['R100_LWC_REFF_VEFF_', num2str(2805+(jj-1)*5),'.0.mat'] ));

%     ext = permute(reshape(full(data.extinction),320,514,514),[3,2,1]);
    %ext = full(data.extinction(:));
    lwc = full(data.LWC(:));
    reff = full(data.REFF(:));
    veff = full(data.VEFF(:));
         time(ii) = data.time;
%     ii = ii + 1;

    %if ~exist('mask')
%         prev_time = data.time;
     %   mask = ext>0; 
      %  extinction(:,ii) = ext;
%         time(ii) = data.time;
%          data.time
       %     ii = ii + 1
%         extinction(:,ii) = ext;
    %else
%         if data.time - prev_time == 5
       % mask = mask | (ext>0);
%         prev_time = data.time;
        %extinction(:,ii) = ext;
%     time(ii) = data.time;
%     data.time
    r100_lwc(:,ii)=lwc;
    r100_reff(:,ii)=reff;
    r100_veff(:,ii)=veff;
     ii = ii + 1
%         end
    %end
r100_lwc = sparse(r100_lwc);
r100_reff = sparse(r100_reff);
r100_veff = sparse(r100_veff);
%mask = sparse(mask);
%     extinction(:,:,:,jj) = ext;
%     extinction=extinction(mask,:);

end
save('matlab_lwc_R100','r100_lwc','time','-v7.3');
save('matlab_reff_R100','r100_reff','time','-v7.3');
save('matlab_veff_R100','r100_veff','time','-v7.3');


% extinction = extinction(mask,:);
%%
% load('matlab_ext_final.mat')
flat_ext = reshape(extinction(mask,:),[],length(time));
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
L = length(time);             % Length of signal
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
% [S,F,T] = stft(full(flat_ext)',Fs,'Window',kaiser(12,5),'OverlapLength',5,'FFTLength',12);
[S,F,T] = stft(full(flat_ext)',Fs,'Window',hamming(12),'OverlapLength',3,'FFTLength',50);
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
plot(F,S1(:,7),'LineWidth',2)
plot(F,S1(:,13),'LineWidth',2)
legend('Time 0','Time 300[sec]','Time 600[sec]')