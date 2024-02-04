%{
    本代码用于对雷达的回波数据，利用wka算法进行成像，利用电平饱和法以及直方图均衡法，
    提高成像质量。
    2023/11/26 21:16
%}
close all;
%% 数据读取
% 加载数据
echo1 = importdata('CDdata1.mat');
echo2 = importdata('CDdata2.mat');
% 将回波拼装在一起
echo = double([echo1;echo2]);
% 加载参数
para = importdata('CD_run_params.mat');
Fr = para.Fr;   % 距离向采样率
Fa = para.PRF;  % 方位向采样率
f0 = para.f0;   % 中心频率
Tr = para.Tr;   % 脉冲持续时间
R0 = para.R0;   % 最近点斜距
Kr = -para.Kr;   % 线性调频率
c = para.c;     % 光速
% 以下参数来自课本附录A
Vr = 7062;      % 等效雷达速度
Ka = 1733;      % 方位向调频率
f_nc = -6900;   % 多普勒中心频率

%% 图像填充
% 计算参数
[Na,Nr] = size(echo);
% 按照全尺寸对图像进行补零
echo = padarray(echo,[round(Na/6), round(Nr/3)]);
% 计算参数
[Na,Nr] = size(echo);

%% 轴产生
% 距离向时间轴及频率轴
tr_axis = 2*R0/c + (-Nr/2:Nr/2-1)/Fr;   % 距离向时间轴
fr_gap = Fr/Nr;
fr_axis = fftshift(-Nr/2:Nr/2-1).*fr_gap;   % 距离向频率轴

% 方位向时间轴及频率轴
ta_axis = (-Na/2:Na/2-1)/Fa;    % 方位向时间轴
ta_gap = Fa/Na; 
fa_axis = f_nc + fftshift(-Na/2:Na/2-1).*ta_gap;    % 方位向频率轴
% 方位向对应纵轴，应该转置成列向量
ta_axis = ta_axis';
fa_axis = fa_axis';

%% 第一步 二维傅里叶变换
% 方位向下变频
echo = echo .* exp(-2i*pi*f_nc.*ta_axis);
% 二维傅里叶变换
echo_s1 = fft2(echo);
%% 第二步 参考函数相乘(一致压缩)
% 生成参考函数
theta_ft_fa = 4*pi*R0/c.*sqrt((f0+fr_axis).^2-c^2/4/Vr^2.*fa_axis.^2)+pi/Kr.*fr_axis.^2;
theta_ft_fa = exp(1i.*theta_ft_fa);
% 一致压缩
echo_s2 = echo_s1 .* theta_ft_fa;

%% 第三步 在距离域进行Stolt插值操作(补余压缩)
% 计算映射后的距离向频率
fr_new_mtx = sqrt((f0+fr_axis).^2-c^2/4/Vr^2.*fa_axis.^2)-f0;
% Stolt映射
echo_s3 = zeros(Na,Nr);
t = waitbar(0,'Stolt映射中');
for i = 1:Na
    if mod(i,10) == 0
        waitbar(i/Na,t);
    end
    echo_s3(i,:) = interp1(fr_axis,echo_s2(i,:),fr_new_mtx(i,:),"spline",0);
end
close(t);

%% 第四步 二维逆傅里叶变换
echo_s4 = ifft2(echo_s3);

%% 第五步 图像纠正
echo_s5 = circshift(echo_s4,-1800,2);
echo_s5 = circshift(echo_s5,-3365,1);
echo_s5 = flipud(echo_s5);

%% 画图
saturation = 50;
figure;
echo_s6 = abs(echo_s5);
echo_s6(echo_s6 > saturation) = saturation;
imagesc(tr_axis.*c,ta_axis.*c,echo_s6);
title('ωk处理结果(精确版本)');
% 绘制处理结果灰度图
% 做一些图像处理。。。
echo_res = gather(echo_s6 ./ saturation);
% 直方图均衡
echo_res = adapthisteq(echo_res,"ClipLimit",0.004,"Distribution","exponential","Alpha",0.5);
figure;
imshow(echo_res);