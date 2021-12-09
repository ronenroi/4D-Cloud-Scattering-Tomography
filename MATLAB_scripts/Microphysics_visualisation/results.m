clear all
close all
%% summary sat3x7 sat2x7  dt 10
%%3 sat
sigma = [0 10 20 30 40 100];%100=inf
eps = [ 0.5997 0.637 0.661 0.668 0.6888 0.704 0.751 ;
      0.573 0.582 0.609 0.611 0.641 0.655 0.727;
      0.569 0.569 0.585 0.587 0.6 0.636 0.7
      0.587 0.58 0.592 0.589 0.598 0.635 0.723;
      0.614 0.607 0.619 0.618 0.63 0.67 0.77;
      0.908 0.823 0.756 0.73 0.75 0.806 0.88
      ];
%%
N = ceil(size(eps,2)/2);
figure(1)
eps_base_line = ones(length(sigma)+2)*0.4777;
plot([-100 sigma 160],eps_base_line,'k','LineWidth',2)
hold on

%%
map = hsv2rgb( [0 0.25 1 
    0  0.5 1 
    0 0.75 1 
    0 1 1 ]);


figure(1)
colormap(map)
jetcustom =map;
for H = 1:7
    c = H.*(H<=N) + (N*2-H).*(H>N);
    plot(sigma,eps(:,H),  'Color',  jetcustom(c,:),'Marker','o','LineStyle','none')   
    hold on
end
hold on
plot(sigma,mean(eps,2),'--r','LineWidth',2)



%% summary sat2x7 dt 10
sigma = [0 10 20 30 40 100];%100=inf
eps = [0.935 0.798 0.761 0.735 0.761 0.815 0.857 ;
     0.843 0.76 0.7 0.688 0.7 0.752 0.823;
     0.71 0.722 0.69 0.67 0.68 0.72 0.767;
     0.758 0.706 0.679 0.665 0.678 0.72 0.782;
     0.7577 0.71 0.69 0.68 0.69 0.7234 0.782;
     0.8625 0.791 0.7416 0.717 0.744 0.8 0.854];
 
% delta = [ -0.194 -0.214 -0.2 -0.245 -0.237 -0.265 -0.286;
%      -0.2 -0.22 -0.197 -0.255 -0.248 -0.264 -0.278;
%      -0.1735 -0.2 -0.22 -0.234 -0.256 -0.277 -0.283;
%      -0.148 -0.195 -0.231 -0.239 -0.253 -0.266 -0.258;
%      -0.125 -0.19 -0.24 -0.252 -0.257 -0.258 -0.24;
%      -0.119 -0.1605 -0.2 -0.225 -0.244 -0.264 -0.279];%%
%%
N = ceil(size(eps,2)/2);

map = hsv2rgb( [0.3 0.25 1 
    0.3  0.5 1 
    0.3 0.75 1 
    0.3 1 1 ]);


figure(1)
colormap(map)
jetcustom =map;
for H = 1:7
    c = H.*(H<=N) + (N*2-H).*(H>N);
    plot(sigma,eps(:,H),  'Color',  jetcustom(c,:),'Marker','o','LineStyle','none')   
    hold on
end
hold on
plot(sigma,mean(eps,2),'color',[0 0.5 0],'LineWidth',2)
xlabel('$\sigma$ [sec]','interpreter','latex','FontSize', 18)
ylabel('$\varepsilon$','interpreter','latex','FontSize', 18)
limsy=get(gca,'YLim');
set(gca,'Ylim',[0 1]);
xlim([-5 105])
xticks([0 20 40 60 80 100])
xticklabels({'0' '20' '40' '60' '80' '\infty'})
ylim([0 1])
yticks([0 0.25 0.5 0.75 1 ])
yticklabels({'0' '0.25' '0.5' '0.75' '1' })
set(gca,'linewidth',2)
grid on
breakxaxis([90 95]);

