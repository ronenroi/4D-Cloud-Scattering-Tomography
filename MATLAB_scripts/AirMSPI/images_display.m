close all
M = 0.1;


load('sigma_60_3.mat')
input_image3=input_image;
dynamic_image3=interp_image;

figure;
imshow(input_image3,[0 M])
colormap('gray')
axis off
figure;
imshow(squeeze(dynamic_image3),[0 M])
colormap('gray')
axis off

load('sigma_60_10.mat')
input_image10=input_image;
dynamic_image10=interp_image;


figure;
imshow(input_image10,[0 M])
colormap('gray')
axis off
figure;
imshow(squeeze(dynamic_image10),[0 M])
colormap('gray')
axis off



load('sigma_60_17.mat')
input_image17=input_image;
dynamic_image17=interp_image;


figure;
imshow(input_image17,[0 M])
colormap('gray')
axis off
figure;
imshow(squeeze(dynamic_image17),[0 M])
colormap('gray')
axis off


load('static_3.mat')
static_image3=estimated_image;

figure;
imshow(squeeze(static_image3),[0 M])
colormap('gray')
axis off


load('static_10.mat')
static_image10=estimated_image;
figure;
imshow(squeeze(static_image10),[0 M])
colormap('gray')
axis off

load('static_17.mat')
static_image17=estimated_image;
figure;
imshow(squeeze(static_image17),[0 M])
colormap('gray')
axis off

dynamic_mse(1) = 0.5* norm(input_image3(:)-dynamic_image3(:))^2;
dynamic_mse(2) = 0.5* norm(input_image10(:)-dynamic_image10(:))^2;
dynamic_mse(3) = 0.5* norm(input_image17(:)-dynamic_image17(:))^2;
mean(dynamic_mse)

static_mse(1) = 0.5* norm(input_image3(:)-static_image3(:))^2;
static_mse(2) = 0.5* norm(input_image10(:)-static_image10(:))^2;
static_mse(3) = 0.5* norm(input_image17(:)-static_image17(:))^2;
mean(static_mse)
%%
figure
imshow(imadjust(squeeze(input_image10)))
axis off
figure
imshow(imadjust(squeeze(dynamic_image10)))
axis off
figure
imshow(imadjust(squeeze(static_image10)))
axis off