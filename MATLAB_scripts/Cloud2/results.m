clear all
close all
%% summary sat3x7 sat2x7 singlex21 dt 10
%%3 sat
sigma = [0 20 40 60 80 100];%100=inf
eps = [0.54743409,0.57329822,0.58554029,0.60142469,0.64950126,0.64238775,0.68781751;0.51125777,0.50664449,0.53252620,0.54851192,0.56842995,0.60944265,0.72017872;0.55586296,0.52885389,0.52976388,0.53710294,0.57145953,0.61663580,0.71444082;0.58662635,0.55938739,0.55507267,0.56653202,0.60775280,0.66309148,0.76029253;0.62274641,0.59379560,0.58646023,0.59894532,0.64253527,0.69666326,0.78808516;0.84340483,0.75679666,0.68805337,0.67482859,0.72128719,0.79599977,0.87319416];
delta = [0.020947136,0.022921830,0.019938184,0.012522294,-0.0047142543,-0.0073907874,-0.024847897;0.012419914,0.037769191,0.034619194,0.016045192,0.016885217,0.0067476290,-0.050571274;-0.042113587,0.026392002,0.078207850,0.093380898,0.078765564,0.014111244,-0.087988526;-0.048982073,0.020538846,0.073954329,0.090224989,0.081556179,0.026861161,-0.069084853;-0.043419342,0.021257650,0.072025143,0.087123021,0.081749626,0.036318935,-0.040033728;-0.064705610,-0.028340098,0.0094531858,0.027690683,0.042889014,0.047585845,0.057647172];
%%
N = ceil(size(eps,2)/2);
figure(1)
eps_base_line = ones(length(sigma)+2)*0.3544;
plot([-100 sigma 160],eps_base_line,'k','LineWidth',2)
hold on
figure(2)
delta_base_line = ones(length(sigma)+2)*0.0205;
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
eps = [1.2866851,0.83195239,0.75403678,0.76871312,0.78352273,0.82567573,0.79593629;0.75502306,0.69113982,0.64514244,0.63895190,0.65898395,0.69394422,0.76168072;0.70903528,0.64031821,0.61414683,0.62708265,0.66560763,0.72436786,0.81887794;0.72021419,0.65710622,0.63308901,0.64886218,0.69082391,0.74486941,0.83698243;0.74574667,0.67973244,0.64954913,0.66504753,0.71074980,0.76492923,0.84832251;0.86645073,0.78570187,0.73208374,0.74046767,0.80156583,0.86706525,0.93525851];
delta = [0.0032470303,-0.010217149,0.0093823327,-0.0034305411,0.026054645,0.021202674,0.047630377;-0.025144335,0.029154917,0.048320856,0.031480696,0.028465059,0.028223759,0.019424727;-0.077980042,-0.0024865824,0.050651770,0.057901703,0.050467499,0.018370062,-0.026441034;-0.087969236,-0.013717513,0.044069648,0.061074022,0.060305547,0.032100040,-0.0028182196;-0.080556668,-0.016310135,0.037709396,0.055749081,0.059772976,0.041148268,0.019360926;-0.084295847,-0.047261227,-0.0087725567,0.0098005030,0.025278479,0.030061731,0.040308181];

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
xticklabels({'0' '20' '40' '60' '80' '\infty'})
set(gca,'linewidth',2)
grid on
breakxaxis([90 95]);
