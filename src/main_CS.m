%{
    本代码用于对雷达的回波数据，利用CS算法进行成像，利用电平饱和法以及直方图均衡法，
    提高成像质量。
    2023/11/18 11:06
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
tr_axis = 2*R0/c + (0:Nr-1)/Fr;   % 距离向时间轴
fr_gap = Fr/Nr;
fr_axis = fftshift(-Nr/2:Nr/2-1).*fr_gap;   % 距离向频率轴

% 方位向时间轴及频率轴
ta_axis = (-Na/2:Na/2-1)/Fa;    % 方位向时间轴
ta_gap = Fa/Na; 
fa_axis = f_nc + fftshift(-Na/2:Na/2-1).*ta_gap;    % 方位向频率轴
% 方位向对应纵轴，应该转置成列向量
ta_axis = ta_axis';
fa_axis = fa_axis';

%% 第一步 相位相乘
% 方位向下变频
echo = echo .* exp(-2i*pi*f_nc.*ta_axis);
% 方位向傅里叶变换
echo_fft_a = fft(echo,[],1);
% 计算徙动参数
D_fa_Vr = sqrt(1-c^2*fa_axis.^2/(4*Vr^2*f0^2));  % 关于方位向频率的徙动参数矩阵
D_fnc_Vr = sqrt(1-c^2*f_nc^2/(4*Vr^2*f0^2));    % 关于参考多普勒中心的徙动参数
R0_var = c * tr_axis / 2;    % 随距离变化的最近点斜距
Km = Kr./(1-Kr.*(c*R0_var.*fa_axis.^2./(2*Vr^2*f0^3.*D_fa_Vr.^3))); % 改变后的距离向调频率
% 计算变标方程
tao_new = tr_axis - 2*R0./(c.*D_fa_Vr);   % 新的距离向时间
Ssc = exp(1i*pi*Km.*(D_fnc_Vr./D_fa_Vr - 1).*(tao_new.^2));    % 变标方程
% 补余RCMC中的Chirp Scaling操作
echo_s1 = echo_fft_a .* Ssc;

%% 第二步 相位相乘
% 距离向傅里叶变换
echo_s2 = fft(echo_s1,[],2);
% 补偿第2项
echo_d2_mf = exp(1i*pi*D_fa_Vr.*(fr_axis.^2)./(Km.*D_fnc_Vr));
% 补偿第4项
echo_d4_mf = exp(4i*pi/c*R0*(1./D_fa_Vr-1/D_fnc_Vr).*fr_axis);
% 参考函数相乘用于距离压缩、SRC和一致性RCMC 
echo_s3 = echo_s2 .* echo_d2_mf .* echo_d4_mf;

%% 第三步 相位相乘
% 距离向逆傅里叶变换
echo_s4 = ifft(echo_s3,[],2);
% 方位向匹配滤波
echo_d1_mf = exp(4i*pi*R0_var.*f0/c.*D_fa_Vr);  % 方位向匹配滤波器
% 变标相位矫正
echo_d3_mf = exp(-4i*pi*Km/c^2.*(1-D_fa_Vr./D_fnc_Vr)...
    .*(R0_var./D_fa_Vr-R0./D_fa_Vr).^2);
% 方位向逆傅里叶变换
echo_s5 = ifft(echo_s4 .* echo_d1_mf .* echo_d3_mf,[],1);

%% 数据最后的矫正
% 根据实际观感，方位向做合适的循环位移
echo_s5 = circshift((echo_s5), -3328, 1);
% 上下镜像
echo_s6 = flipud(echo_s5);
% 取模
echo_s7 = abs(echo_s6);
%% 数据可视化
% 绘制直方图
figure;
histogram(echo_s7(:),50);
% 根据直方图结果做饱和处理
saturation = 50;
echo_s7(echo_s7 > saturation) = saturation;
% 绘制处理结果热力图
figure;
imagesc(tr_axis.*c,ta_axis.*c,(echo_s7));
title('处理结果(CS算法)');
% 绘制处理结果灰度图
% 做一些图像处理。。。
echo_res = gather(echo_s7 ./ saturation);
% 直方图均衡
echo_res = adapthisteq(echo_res,"ClipLimit",0.004,"Distribution","exponential","Alpha",0.5);
figure;
imshow(echo_res);