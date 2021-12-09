%% Show figures
close all;
clear all
wavelength = {'355','380','445','470'...
    ,'555','660','865','935'};
wavelength = {'660'};


load('/home/roironen/Desktop/vadim/test/roi21_smallcloud.mat', 'roi21')
folder = '/home/roironen/Desktop/vadim/test';
sad = dir( fullfile( folder, '*.hdf' ) );
len = length( sad );
out = struct( 'name',cell(1,len), 'data',[] );
X = 1:3584;
Y = 1:2560;
image = zeros(3584,2560);
image = zeros(2400,4000);

thr = [0.0734 0.0665 0.062 .057 0.0511 0.046 0.0393 0.0346 0.0277 0.0241 0.0249 0.0237 0.038  0.034 ...
    0.04 0.041 0.045 0.0473 0.0465 0.0436 0.0443 ];
for jj = 2:2:len
    
    
    
    
    roi = roi21(jj,:);
    out(jj).name = sad(jj).name;
    SelectDataset = ['/HDFEOS/GRIDS/' wavelength{1} 'nm_band/Data Fields/I'];
    
    data= h5read(fullfile( folder, sad(jj).name ),SelectDataset);
    data(data == -999) = 0;
    img=data(roi(3):roi(4),roi(1):roi(2));
    
    SelectDataset = ['/HDFEOS/GRIDS/' wavelength{1} 'nm_band/Data Fields/View_zenith'];
    data= h5read(fullfile( folder, sad(jj).name ),SelectDataset);
    zenith=data(roi(3):roi(4),roi(1):roi(2));
    
    SelectDataset = ['/HDFEOS/GRIDS/' wavelength{1} 'nm_band/Data Fields/View_azimuth'];
    data= h5read(fullfile( folder, sad(jj).name ),SelectDataset);
    azimuth=data(roi(3):roi(4),roi(1):roi(2));
    
    %     SelectDataset = ['/HDFEOS/GRIDS/' wavelength{1} 'nm_band/Data Fields/XDim'];
    %     data= h5read(fullfile( folder, sad(jj).name ),SelectDataset);
    %     XDim=data;
    %
    %     SelectDataset = ['/HDFEOS/GRIDS/' wavelength{1} 'nm_band/Data Fields/YDim'];
    %     data= h5read(fullfile( folder, sad(jj).name ),SelectDataset);
    YDim=data;
    [x,y] = meshgrid(Y,X);
    x = x(roi(3):roi(4),roi(1):roi(2));
    y = y(roi(3):roi(4),roi(1):roi(2));
    theta = deg2rad(zenith);
    mu = cos(theta);
    phi = deg2rad(azimuth);
    
    Xcloud_base_pos = round(x + 80 * tan(theta) .* cos(phi));
    Ycloud_base_pos  = round(y + 80 * tan(theta) .* sin(phi));
    
    %     h=subplot(1,1,1);
    %     finalThreshold = isodataAlgorithm(img)
    img_bin = (img>thr(jj)).*(theta>=0).*(phi>=0);
    xx = (x + 2000 * tan(theta) .* cos(phi));
    [n,m]=size(xx);
        yy = (y + 2000 * tan(theta) .* sin(phi));
    airmspi_x(jj) =  xx(round(n/2),round(m/2));
    airmspi_y(jj) =  yy(round(n/2),round(m/2));

    com_x(jj) = sum(Xcloud_base_pos(:,:).*img.*img_bin,[1,2])/sum(img.*img_bin,'all');
    com_y(jj) = sum(Ycloud_base_pos(:,:).*img.*img_bin,[1,2])/sum(img.*img_bin,'all');
    figure(1)
    imshow(img_bin);
    
    %         hrect = imrect(h);
    % h = drawrectangle('Position',rect(jj,:),'StripeColor','r');
    
    %
    %         rect(jj,:) = wait(hrect);
    
    
    out(jj).img = img;
    %     out(jj).zenith = zenith;
    %     out(jj).azimuth = azimuth;
    out(jj).x = Xcloud_base_pos;
    out(jj).y = Ycloud_base_pos;
    img_bin = edge(img_bin);
    img_bin = imdilate(img_bin, strel('disk',2));

%     image(Ycloud_base_pos(:,1),Xcloud_base_pos(1,:)) =  img_bin*10*jj+ (1-img_bin).*image(Ycloud_base_pos(:,1),Xcloud_base_pos(1,:)) ;
    image(Ycloud_base_pos(1,1):Ycloud_base_pos(1,1)+n-1,Xcloud_base_pos(1,1):Xcloud_base_pos(1,1)+m-1) =  img_bin*10*jj+ (1-img_bin).*image(Ycloud_base_pos(1,1):Ycloud_base_pos(1,1)+n-1,Xcloud_base_pos(1,1):Xcloud_base_pos(1,1)+m-1) ;
    
figure(2)
    
cmap = hsv(22);
cmap(1,:)=[1 1 1];
    imagesc(image)
    colormap(cmap)
    hold on;

%     hold off;
    clear img
end

hold on;
for row = 1 : 500 : 4000
  line([1, 4000], [row, row], 'Color', 'k');
end
for col = 1 : 500 : 4000
  line([col, col], [1, 4000], 'Color', 'k');
end
for jj = 2:2:len
        plot(com_x(jj), com_y(jj), '.', 'Color',cmap(jj+1,:),'MarkerSize', 20);
        plot(airmspi_x(jj), airmspi_y(jj), '+','Color',cmap(jj+1,:), 'MarkerSize', 20);
end
axis equal
xlim([500 3000])
ylim([500 2000])
%  close all
% imshow(image)
%     hold on;
% plot(com_x, com_y, 'r.', 'MarkerSize', 2);
% hold off;
%%
function finalThreshold = isodataAlgorithm(grayImage)
grayImage =  grayImage(:);
% The itial threshol is equal the mean of grayscale image
initialTheta = mean(grayImage);
%     initialTheta = round(initialTheta); % Rounding
i = 1;
threshold(i) = initialTheta;
% Gray levels are greater than or equal to the threshold
foregroundLevel =  grayImage(find((grayImage >= initialTheta)));
meanForeground = mean(foregroundLevel(:));
% Gray levels are less than or equal to the threshold
backgroundLevel = grayImage(find((grayImage < initialTheta)));
meanBackground = mean(backgroundLevel(:));
% Setup new threshold
i = 2;
threshold(2) = round((meanForeground + meanBackground)/2);
%Loop: Consider condition for threshold
while abs(threshold(i)-threshold(i-1))>=1
    
    % Gray levels are greater than or equal to the threshold
    foregroundLevel =  grayImage(find((grayImage >= threshold(i))));
    meanForeground = (mean(foregroundLevel(:)));
    % Gray levels are less than or equal to the threshold
    backgroundLevel = grayImage(find((grayImage < threshold(i))));
    meanBackground = (mean(backgroundLevel(:)));
    i = i+1;
    % Setup new threshold
    threshold(i) = round((meanForeground + meanBackground)/2);
    
end
finalThreshold = threshold(end);
end