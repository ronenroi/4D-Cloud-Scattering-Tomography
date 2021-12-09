clear all
close all
%% summary sat3x7 sat2x7  dt 10
%%3 sat
sigma = [0 10 20 40 60 80 100];%100=inf
eps = [0.8 0.76 0.733 0.734 0.74 0.73  0.74;
    0.62 0.586 0.573 0.579 0.582 0.6 0.628;
    0.6 0.56 0.55 0.55 0.55 0.55 0.59;
    0.623 0.57 0.56 0.56 0.56 0.565 0.6;
    0.67 0.61 0.59 0.6 0.6 0.6 0.62;
    0.71 0.65 0.61 0.61 0.62 0.62 0.65;
    0.89 0.74 0.63 0.58 0.59 0.64 0.7;
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
eps = [0.996 0.98 0.98 0.9 0.8 0.75 0.74;
        0.85 0.8 0.78 0.77 0.73 0.7 0.7;
        0.77 0.7 0.65 0.63 0.62 0.63 0.67;
        0.77 0.7 0.64 0.63 0.63 0.65 0.7;
        0.81 0.735 0.67 0.66 0.67 0.695 0.74;
        0.865 0.78 0.72 0.7 0.7 0.73 0.77;
        0.966 0.82 0.7 0.67 0.69 0.75 0.81;
];
 
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

