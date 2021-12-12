clear all
close all
%% summary sat3x7 sat2x7 singlex21 dt 10
%%3 sat
sigma = [0 20 40 60 80 100];%100=inf;
eps = [0.53865075,0.55428600,0.52651268,0.54934585,0.56415474,0.64163125,0.62368757;0.50344914,0.45757079,0.44874477,0.47879946,0.49417084,0.52868724,0.60791922;0.58451676,0.52469444,0.51753187,0.53843588,0.55705214,0.60436618,0.72397512;0.63671386,0.56382543,0.55114925,0.56554115,0.57910275,0.62454313,0.72576612;0.69092327,0.60100824,0.58260888,0.58728218,0.59932566,0.64107108,0.73321915;0.94788027,0.79776412,0.67739737,0.63448733,0.63758397,0.68427277,0.75812471];
delta = [-0.023887804,-0.029144429,-0.021314003,-0.041635059,-0.024719190,-0.043703839,-0.056885615;-0.051756930,0.012782164,0.031227129,0.018730942,0.015476095,0.0068500554,-0.069898479;-0.13165672,-0.033473760,0.026720485,0.055383991,0.054652799,0.0043653972,-0.14710252;-0.14220800,-0.040537246,0.027794072,0.065930098,0.072653510,0.033797469,-0.085052043;-0.15766378,-0.056372374,0.015735967,0.056249131,0.069857448,0.045446858,-0.041722428;-0.26679546,-0.17075114,-0.090483539,-0.020611126,0.045590125,0.098113663,0.12768339];

%%
N = ceil(size(eps,2)/2);
figure(1)
eps_base_line = ones(length(sigma)+2)*0.365;
plot([-100 sigma 160],eps_base_line,'k','LineWidth',2)
hold on
figure(2)
delta_base_line = ones(length(sigma)+2)*-0.0161;
plot([-100 sigma 160],delta_base_line,'k','LineWidth',2)
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

%%
figure(2)
for H = 1:7
    c = H.*(H<=N) + (N*2-H).*(H>N);
    plot(sigma,delta(:,H),  'Color',  jetcustom(c,:),'Marker','o','LineStyle','none')   
    hold on
end

hold on
plot(sigma,mean(delta,2),'--r','LineWidth',2)


%%
sigma = [0 20 40 60 80 100];
eps = [1.3000925,0.94075465,0.84179211,0.93586421,0.86321491,0.77939528,0.75375974;0.77182144,0.67236108,0.60996234,0.60145402,0.60600346,0.65230972,0.77087593;0.76188648,0.66025466,0.61197990,0.59938455,0.62902427,0.69950205,0.81733084;0.84188503,0.73099583,0.67962575,0.66436923,0.68166035,0.74257720,0.85264170;0.93993425,0.82028747,0.76227355,0.74642378,0.75807667,0.81654167,0.93164033;1.0217564,0.87860268,0.77289373,0.73683226,0.74873024,0.79567248,0.87706530];
delta = [-0.22773567,-0.11443589,-0.12442271,-0.17418647,-0.17739035,-0.16237968,-0.13271469;-0.10888878,-0.011782553,0.0080950391,-0.022439895,-0.042296935,-0.067133099,-0.15836239;-0.18392207,-0.071825437,-0.0063423715,0.012767635,0.0043513495,-0.034878187,-0.13032390;-0.23750341,-0.11615866,-0.037966475,-0.00047004534,0.010898415,-0.010367798,-0.075809442;-0.27680066,-0.15350875,-0.074751943,-0.032164901,-0.013981523,-0.030011417,-0.083129302;-0.35056472,-0.24816926,-0.16259383,-0.088100970,-0.017522030,0.038474727,0.069999807];


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
%%
figure(2)
for H = 1:7
    c = H.*(H<=N) + (N*2-H).*(H>N);
    plot(sigma,delta(:,H),  'Color',  jetcustom(c,:),'Marker','o','LineStyle','none')   
    hold on
end

hold on
plot(sigma,mean(delta,2),'color',[0 0.5 0],'LineWidth',2)
limsy=get(gca,'YLim');
set(gca,'Ylim',[limsy(1) limsy(2)]);

xlabel('$\sigma$ [sec]','interpreter','latex','FontSize', 18)
ylabel('$\delta$','interpreter','latex','FontSize', 18)
xlim([-5 105])
xticks([0 20 40 60 80 100])
ylim([-0.2 0.2])
xticklabels({'0' '20' '40' '60' '80' '\infty'})
set(gca,'linewidth',2)
grid on
breakxaxis([90 95]);