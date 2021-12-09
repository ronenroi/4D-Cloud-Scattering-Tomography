clear all
close all
%% summary sat3x7 sat2x7  dt 10
%%3 sat
sigma = [0 10 20 40 60 80 100];%100=inf
eps = [0.52 0.51 0.5 0.52 0.55 0.6 0.61;
    0.49 0.46 0.45 0.46 0.49 0.53 0.57;
    0.49 0.44 0.43 0.45 0.47 0.52 0.56;
    0.55 0.47 0.46 0.48 0.51 0.58 0.71;
    0.63 0.54 0.52 0.54 0.55 0.61 0.71;
    0.65 0.56 0.54 0.55 0.56 0.61 0.7;
    0.87 0.72 0.61 0.56 0.57 0.64 0.71;
      ];
%%
N = ceil(size(eps,2)/2);
figure(1)
eps_base_line = ones(length(sigma)+2)*0.29;
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
eps = [0.86 0.77 0.75 0.78 0.73 0.72 0.71;
        0.77 0.68 0.66 0.64 0.63 0.66 0.71;
        0.72 0.64 0.57 0.56 0.57 0.62 0.7;
        0.73 0.64 0.59 0.58 0.59 0.64 0.74;
        0.77 0.68 0.63 0.62 0.63 0.68 0.78;
        0.83 0.73 0.68 0.67 0.67 0.71 0.8;
        0.96 0.82 0.71 0.67 0.69 0.75 0.83;
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

